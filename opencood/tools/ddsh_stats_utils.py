# -*- coding: utf-8 -*-
"""Utilities for logging DDSH-VoxelNeXt paper statistics.

The helpers in this file intentionally avoid importing heavy model modules.
They only flatten already computed ``output_dict`` fields into CSV-friendly
records, so they can be reused by training, debug, and offline profiling.
"""

import csv
import json
import math
import os
from collections import OrderedDict

import torch


CSV_FIELDS = [
    'phase',
    'stage',
    'epoch',
    'iteration',
    'global_step',
    'batch_idx',
    'scene_id',
    'record_len',
    'loss',
    'lr',
    'elapsed_ms',
    'peak_memory_mb',
    'num_helpers',
    'num_ego_tokens',
    'num_helper_tokens',
    'num_demand_tokens',
    'num_supply_tokens',
    'num_fused_tokens',
    'fusion_input_tokens',
    'fusion_match_token_count',
    'per_cav_token_count',
    'helper_aligned_token_count',
    'helper_supply_token_count',
    'demand_bytes',
    'supply_bytes',
    'total_bytes',
    'per_helper_supply_bytes',
    'feature_dim',
    'demand_match_distance_min',
    'demand_match_distance_mean',
    'demand_match_distance_max',
]


SUMMARY_FIELDS = [
    'metric',
    'count',
    'mean',
    'std',
    'min',
    'p50',
    'p90',
    'max',
]


SUMMARY_METRICS = [
    'loss',
    'elapsed_ms',
    'peak_memory_mb',
    'num_ego_tokens',
    'num_helper_tokens',
    'num_demand_tokens',
    'num_supply_tokens',
    'num_fused_tokens',
    'fusion_input_tokens',
    'fusion_match_token_count',
    'demand_bytes',
    'supply_bytes',
    'total_bytes',
    'feature_dim',
    'demand_match_distance_min',
    'demand_match_distance_mean',
    'demand_match_distance_max',
]


def is_rank0(opt):
    """Return True when this process should write shared statistics files."""
    if opt is None or not getattr(opt, 'distributed', False):
        return True
    try:
        from opencood.tools import multi_gpu_utils
        return multi_gpu_utils.get_dist_info()[0] == 0
    except Exception:
        return True


def cfg_enabled(hypes):
    """Check whether DDSH CSV statistics are enabled in YAML."""
    cfg = hypes.get('ddsh_stats', {})
    return bool(cfg.get('enable', False))


def should_collect(hypes, global_step, opt=None):
    """Check interval and distributed rank before collecting debug stats."""
    if not cfg_enabled(hypes):
        return False
    if not is_rank0(opt):
        return False
    cfg = hypes.get('ddsh_stats', {})
    interval = max(1, int(cfg.get('interval', 100)))
    return int(global_step) % interval == 0


def output_path(saved_path, hypes, filename=None):
    """Return the CSV path for training-time DDSH statistics."""
    cfg = hypes.get('ddsh_stats', {})
    save_dir = os.path.join(saved_path, cfg.get('save_dir', 'ddsh_statistics'))
    if filename is None:
        filename = cfg.get('train_csv', 'train_ddsh_statistics.csv')
    return os.path.join(save_dir, filename)


def maybe_reset_peak_memory(enabled, device):
    """Reset CUDA peak memory only for sampled iterations."""
    if not enabled or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def maybe_sync_cuda(enabled):
    """Synchronize CUDA only when latency profiling is requested."""
    if enabled and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def peak_memory_mb(device):
    """Return current CUDA peak memory in MB, or 0 when CUDA is unavailable."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
    except Exception:
        return 0.0


def _to_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return float(value.detach().cpu().reshape(-1)[0])
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value, default=0):
    return int(round(_to_float(value, float(default))))


def _record_len_string(record_len):
    if record_len is None:
        return ''
    if isinstance(record_len, torch.Tensor):
        values = record_len.detach().cpu().reshape(-1).tolist()
    elif isinstance(record_len, (list, tuple)):
        values = list(record_len)
    else:
        values = [record_len]
    return ';'.join(str(int(v)) for v in values)


def _list_counts(items, count_key='count'):
    if not items:
        return ''
    values = []
    for item in items:
        if isinstance(item, dict):
            values.append(str(item.get(count_key, 0)))
        else:
            values.append(str(item))
    return ';'.join(values)


def _list_numbers(values):
    if values is None:
        return ''
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().reshape(-1).tolist()
    return ';'.join(str(v) for v in values)


def _distance_summary(helper_debug, key):
    values = []
    for item in helper_debug or []:
        value = item.get(key, None)
        if value is not None:
            values.append(float(value))
    if len(values) == 0:
        return None
    if key.endswith('_min'):
        return min(values)
    if key.endswith('_max'):
        return max(values)
    return sum(values) / float(len(values))


def build_record(output_dict, hypes, phase='train', epoch=-1, iteration=-1,
                 global_step=-1, batch_idx=-1, record_len=None, loss=None,
                 lr=None, elapsed_ms=None, peak_memory=None):
    """Flatten DDSH output into a stable CSV row."""
    output_dict = output_dict or {}
    stats = output_dict.get('ddsh_stats', {}) or {}
    debug = output_dict.get('ddsh_debug', {}) or {}
    comm = debug.get('communication', {}) or {}
    helper_debug = stats.get('helper_debug', []) or []

    row = OrderedDict()
    for field in CSV_FIELDS:
        row[field] = ''

    row['phase'] = phase
    row['stage'] = stats.get('stage',
                             debug.get('stage',
                                       hypes.get('ddsh', {}).get('stage', '')))
    row['epoch'] = int(epoch)
    row['iteration'] = int(iteration)
    row['global_step'] = int(global_step)
    row['batch_idx'] = int(batch_idx)
    row['scene_id'] = int(debug.get('scene_id', 0))
    row['record_len'] = _record_len_string(record_len)
    row['loss'] = _to_float(loss, 0.0) if loss is not None else ''
    row['lr'] = _to_float(lr, 0.0) if lr is not None else ''
    row['elapsed_ms'] = _to_float(elapsed_ms, 0.0) \
        if elapsed_ms is not None else ''
    row['peak_memory_mb'] = _to_float(peak_memory, 0.0) \
        if peak_memory is not None else ''

    row['num_helpers'] = _to_int(stats.get('num_helpers', 0))
    row['num_ego_tokens'] = _to_int(comm.get(
        'num_ego_tokens', stats.get('num_ego_tokens', 0)))
    row['num_helper_tokens'] = _to_int(comm.get('num_helper_tokens', 0))
    row['num_demand_tokens'] = _to_int(comm.get(
        'num_demand_tokens', stats.get('num_demand_tokens', 0)))
    row['num_supply_tokens'] = _to_int(comm.get(
        'num_supply_tokens', stats.get('num_supply_tokens', 0)))
    row['num_fused_tokens'] = _to_int(comm.get(
        'num_fused_tokens', stats.get('num_fused_tokens', 0)))
    row['fusion_input_tokens'] = _to_int(stats.get('fusion_input_tokens', 0))
    row['fusion_match_token_count'] = _to_int(
        stats.get('fusion_match_token_count', 0))
    row['per_cav_token_count'] = _list_numbers(
        stats.get('per_cav_token_count', []))
    row['helper_aligned_token_count'] = _list_counts(
        stats.get('helper_aligned_token_count', []))
    row['helper_supply_token_count'] = _list_counts(
        stats.get('helper_supply_token_count', []))

    row['demand_bytes'] = _to_int(comm.get('demand_bytes', 0))
    row['supply_bytes'] = _to_int(comm.get('supply_bytes', 0))
    row['total_bytes'] = _to_int(comm.get('total_bytes', 0))
    row['per_helper_supply_bytes'] = _list_numbers(
        comm.get('per_helper_supply_bytes', []))
    row['feature_dim'] = _to_int(comm.get('feature_dim', 0))
    row['demand_match_distance_min'] = _distance_summary(
        helper_debug, 'demand_match_distance_min')
    row['demand_match_distance_mean'] = _distance_summary(
        helper_debug, 'demand_match_distance_mean')
    row['demand_match_distance_max'] = _distance_summary(
        helper_debug, 'demand_match_distance_max')

    for key, value in list(row.items()):
        if value is None:
            row[key] = ''
    return row


def append_csv(path, row, fields=None):
    """Append one row to a CSV file, creating parent directories as needed."""
    fields = fields or CSV_FIELDS
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in fields})


def write_csv(path, rows, fields=None):
    """Write all rows to a CSV file."""
    fields = fields or CSV_FIELDS
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fields})


def _numeric_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key, '')
        if value == '' or value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if not math.isnan(value):
            values.append(value)
    return values


def _percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = (len(values) - 1) * float(q)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(values[low])
    weight = pos - low
    return float(values[low] * (1.0 - weight) + values[high] * weight)


def summarize(rows, metrics=None):
    """Return summary rows for numeric DDSH statistics."""
    metrics = metrics or SUMMARY_METRICS
    summary = []
    for key in metrics:
        values = _numeric_values(rows, key)
        if not values:
            continue
        mean = sum(values) / float(len(values))
        if len(values) > 1:
            var = sum((v - mean) ** 2 for v in values) / \
                float(len(values) - 1)
            std = math.sqrt(max(var, 0.0))
        else:
            std = 0.0
        summary.append(OrderedDict([
            ('metric', key),
            ('count', len(values)),
            ('mean', mean),
            ('std', std),
            ('min', min(values)),
            ('p50', _percentile(values, 0.50)),
            ('p90', _percentile(values, 0.90)),
            ('max', max(values)),
        ]))
    return summary


def write_summary(summary_csv_path, rows):
    """Write CSV and JSON summaries next to the detailed CSV."""
    summary = summarize(rows)
    write_csv(summary_csv_path, summary, SUMMARY_FIELDS)
    json_path = os.path.splitext(summary_csv_path)[0] + '.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as json_file:
        json.dump(summary, json_file, indent=2)
    return summary

