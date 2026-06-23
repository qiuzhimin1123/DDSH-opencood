# -*- coding: utf-8 -*-
"""Profile DDSH-VoxelNeXt token, communication, memory and latency stats.

This script is intended for paper tables. It runs forward passes with
``ddsh_collect_debug=True`` and saves detailed per-batch CSV plus aggregate
summary files. It does not train the model and does not compute AP.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../..')))

import torch
import tqdm
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import ddsh_stats_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile DDSH sparse token communication statistics.')
    parser.add_argument('--hypes_yaml', type=str, default='',
                        help='DDSH yaml config. Optional when --model_dir '
                             'contains config.yaml.')
    parser.add_argument('--model_dir', type=str, default='',
                        help='Optional checkpoint/log directory to load.')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'train'],
                        help='Dataset split to profile.')
    parser.add_argument('--max_batches', type=int, default=-1,
                        help='Maximum number of batches to profile. '
                             'Use -1 for the full split.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for profiling. Paper statistics '
                             'usually use 1.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Dataloader workers. Use 0 for deterministic '
                             'debug/profile runs.')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory for CSV/JSON outputs. Default: '
                             '<model_dir>/ddsh_statistics or '
                             'opencood/logs/ddsh_profile/<config_name>.')
    parser.add_argument('--prefix', type=str, default='ddsh_statistics',
                        help='Output file prefix.')
    parser.add_argument('--no_latency_sync', action='store_true',
                        help='Disable CUDA synchronization for latency timing.')
    parser.add_argument('--no_memory_profile', action='store_true',
                        help='Disable CUDA peak memory profiling.')
    return parser.parse_args()


def _load_hypes(opt):
    if not opt.hypes_yaml and not opt.model_dir:
        raise ValueError('Either --hypes_yaml or --model_dir is required.')
    yaml_file = opt.hypes_yaml or os.path.join(opt.model_dir, 'config.yaml')
    return yaml_utils.load_yaml(yaml_file, opt)


def _default_output_dir(opt, hypes):
    if opt.output_dir:
        return opt.output_dir
    if opt.model_dir:
        return os.path.join(opt.model_dir, 'ddsh_statistics')
    config_name = hypes.get('name', 'ddsh_profile')
    return os.path.join('opencood', 'logs', 'ddsh_profile', config_name)


def main():
    opt = parse_args()
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils

    hypes = _load_hypes(opt)
    hypes.setdefault('train_params', {})['batch_size'] = opt.batch_size

    train_split = opt.split == 'train'
    dataset = build_dataset(hypes, visualize=False, train=train_split)
    collate_fn = dataset.collate_batch_train if train_split else \
        dataset.collate_batch_test
    data_loader = DataLoader(dataset,
                             batch_size=opt.batch_size,
                             num_workers=opt.num_workers,
                             collate_fn=collate_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    model = train_utils.create_model(hypes)
    if opt.model_dir:
        _, model = train_utils.load_saved_model(opt.model_dir, model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    output_dir = _default_output_dir(opt, hypes)
    os.makedirs(output_dir, exist_ok=True)
    detail_csv = os.path.join(
        output_dir, '%s_%s_detail.csv' % (opt.prefix, opt.split))
    summary_csv = os.path.join(
        output_dir, '%s_%s_summary.csv' % (opt.prefix, opt.split))

    rows = []
    use_latency_sync = not opt.no_latency_sync
    use_memory_profile = not opt.no_memory_profile

    print('DDSH profile output dir:', output_dir)
    print('Profiling split=%s batch_size=%d max_batches=%d' %
          (opt.split, opt.batch_size, opt.max_batches))

    iterator = enumerate(data_loader)
    total = len(data_loader) if opt.max_batches < 0 else min(
        len(data_loader), opt.max_batches)
    for batch_idx, batch_data in tqdm.tqdm(iterator, total=total,
                                           desc='DDSH statistics profile'):
        if opt.max_batches >= 0 and batch_idx >= opt.max_batches:
            break
        batch_data = train_utils.to_device(batch_data, device)
        ego_batch = batch_data['ego']
        ego_batch['ddsh_collect_debug'] = True

        ddsh_stats_utils.maybe_reset_peak_memory(use_memory_profile, device)
        ddsh_stats_utils.maybe_sync_cuda(use_latency_sync)
        start = time.perf_counter()
        try:
            with torch.no_grad():
                output_dict = model(ego_batch)
        finally:
            ego_batch.pop('ddsh_collect_debug', None)
        ddsh_stats_utils.maybe_sync_cuda(use_latency_sync)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        peak_memory = ddsh_stats_utils.peak_memory_mb(device) \
            if use_memory_profile else 0.0

        row = ddsh_stats_utils.build_record(
            output_dict,
            hypes,
            phase='profile_%s' % opt.split,
            epoch=-1,
            iteration=batch_idx,
            global_step=batch_idx,
            batch_idx=batch_idx,
            record_len=ego_batch.get('record_len', None),
            loss=None,
            lr=None,
            elapsed_ms=elapsed_ms,
            peak_memory=peak_memory)
        rows.append(row)

    ddsh_stats_utils.write_csv(detail_csv, rows)
    summary = ddsh_stats_utils.write_summary(summary_csv, rows)
    print('Detailed DDSH statistics saved to:', detail_csv)
    print('Summary DDSH statistics saved to:', summary_csv)
    print('Summary JSON saved to:', os.path.splitext(summary_csv)[0] + '.json')
    for item in summary:
        if item['metric'] in ['total_bytes', 'num_supply_tokens',
                              'num_fused_tokens', 'elapsed_ms',
                              'peak_memory_mb']:
            print('%s mean=%.4f p50=%.4f p90=%.4f' %
                  (item['metric'], item['mean'], item['p50'], item['p90']))


if __name__ == '__main__':
    main()
