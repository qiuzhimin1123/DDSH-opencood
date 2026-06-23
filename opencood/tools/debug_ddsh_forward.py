# -*- coding: utf-8 -*-
"""Run one DDSH-VoxelNeXt forward pass and print sparse debug stats."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../..')))

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='Debug one DDSH-VoxelNeXt sparse forward pass.')
    parser.add_argument('--hypes_yaml', type=str, required=True,
                        help='Path to DDSH yaml config.')
    parser.add_argument('--model_dir', type=str, default='',
                        help='Optional checkpoint directory to load.')
    parser.add_argument('--train', action='store_true',
                        help='Use train split/collate. Default is val/test.')
    return parser.parse_args()


def _shape(tensor):
    return tuple(tensor.shape) if tensor is not None else None


def _token_count(tokens):
    if tokens is None:
        return 0
    if 'features' in tokens:
        return int(tokens['features'].shape[0])
    if 'feats' in tokens:
        return int(tokens['feats'].shape[0])
    if 'coords' in tokens:
        return int(tokens['coords'].shape[0])
    return 0


def _register_hooks(model, capture):
    handles = []

    def backbone_hook(module, inputs, output):
        encoded = output.get('encoded_spconv_tensor', None) \
            if isinstance(output, dict) else None
        if encoded is not None:
            capture['encoded_indices_shape'] = _shape(encoded.indices)
            capture['encoded_features_shape'] = _shape(encoded.features)

    def sparse_bev_hook(module, inputs, output):
        tokens = output.get('ddsh_bev_tokens', None) \
            if isinstance(output, dict) else None
        capture['sparse_bev_tokens'] = _token_count(tokens)

    def ddsh_hook(module, inputs, output):
        if isinstance(output, tuple) and len(output) == 2:
            fused_tokens, stats = output
            capture['fused_tokens'] = _token_count(fused_tokens)
            capture['ddsh_stats'] = stats

    raw_model = model.module if hasattr(model, 'module') else model
    if hasattr(raw_model, 'backbone_3d'):
        handles.append(raw_model.backbone_3d.register_forward_hook(
            backbone_hook))
    if hasattr(raw_model, 'sparse_height_compression'):
        handles.append(
            raw_model.sparse_height_compression.register_forward_hook(
                sparse_bev_hook))
    if hasattr(raw_model, 'ddsh'):
        handles.append(raw_model.ddsh.register_forward_hook(ddsh_hook))
    return handles


def _run_debug_paper_vis(output_dict, hypes):
    try:
        from opencood.visualization import ddsh_paper_vis
        if 'ddsh_debug' not in output_dict:
            print('Warning: debug paper vis skipped, ddsh_debug missing.')
            return
        cfg = dict(hypes.get('paper_vis', {}))
        cfg['enable'] = True
        cfg.setdefault('save_png', True)
        cfg.setdefault('save_svg', True)
        cfg.setdefault('save_pdf', True)
        cfg.setdefault('dpi', 600)
        stage = output_dict['ddsh_debug'].get('stage', 'unknown')
        scene_id = int(output_dict['ddsh_debug'].get('scene_id', 0))
        stage_dir = ddsh_paper_vis.STAGE_DIRS.get(stage, str(stage))
        save_dir = os.path.join('opencood', 'logs', 'debug_ddsh_forward',
                                cfg.get('save_dir', 'paper_figures'),
                                stage_dir)
        prefix = '%s_epoch000_iter0000_scene%d' % (stage_dir, scene_id)
        ddsh_paper_vis.visualize_ddsh_debug(
            output_dict['ddsh_debug'],
            save_dir=save_dir,
            prefix=prefix,
            writer=None,
            global_step=0,
            cfg=cfg)
        print('paper figures saved to:', save_dir)
    except Exception as exc:
        print('Warning: debug DDSH paper visualization failed: %s' % exc)


def main():
    opt = parse_args()

    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    hypes.setdefault('train_params', {})['batch_size'] = 1

    print('Building dataset with batch_size=1, num_workers=0')
    dataset = build_dataset(hypes, visualize=False, train=opt.train)
    collate_fn = dataset.collate_batch_train if opt.train else \
        dataset.collate_batch_test
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             num_workers=0,
                             collate_fn=collate_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating model')
    model = train_utils.create_model(hypes)
    if opt.model_dir:
        _, model = train_utils.load_saved_model(opt.model_dir, model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    batch_data = next(iter(data_loader))
    batch_data = train_utils.to_device(batch_data, device)
    ego_batch = batch_data['ego']

    print('batch_data.keys():', list(batch_data.keys()))
    print('batch_dict.keys():', list(ego_batch.keys()))
    print('record_len:', ego_batch.get('record_len'))
    print('pairwise_t_matrix exists:', 'pairwise_t_matrix' in ego_batch)

    capture = {}
    handles = _register_hooks(model, capture)
    ego_batch['ddsh_collect_debug'] = True
    try:
        with torch.no_grad():
            output_dict = model(ego_batch)
    finally:
        ego_batch.pop('ddsh_collect_debug', None)
        for handle in handles:
            handle.remove()

    stats = output_dict.get('ddsh_stats', capture.get('ddsh_stats', {}))
    print('encoded_spconv_tensor indices shape:',
          capture.get('encoded_indices_shape'))
    print('encoded_spconv_tensor features shape:',
          capture.get('encoded_features_shape'))
    print('sparse BEV token count:', capture.get('sparse_bev_tokens', 0))
    print('demand token count:', stats.get('num_demand_tokens', 0))
    print('supply token count:', stats.get('num_supply_tokens', 0))
    print('fused token count:', stats.get(
        'num_fused_tokens', capture.get('fused_tokens', 0)))
    if 'ddsh_debug' in output_dict:
        comm = output_dict['ddsh_debug'].get('communication', {}) or {}
        print('demand bytes:', comm.get('demand_bytes', 0))
        print('supply bytes:', comm.get('supply_bytes', 0))
        print('total communication bytes:', comm.get('total_bytes', 0))
        print('per-helper supply bytes:',
              comm.get('per_helper_supply_bytes', []))
    print('output keys:', list(output_dict.keys()))
    _run_debug_paper_vis(output_dict, hypes)


if __name__ == '__main__':
    main()
