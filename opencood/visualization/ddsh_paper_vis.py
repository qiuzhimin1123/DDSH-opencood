# -*- coding: utf-8 -*-
"""Paper-quality DDSH-VoxelNeXt visualizations.

All plotting functions are defensive by design: they detach tensors to CPU
before plotting, skip missing fields with warnings, and close figures after
saving to avoid memory leaks during training.
"""

from __future__ import print_function

import os
import warnings

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon


STAGE_DIRS = {
    'sparse_single': 'stage0_sparse_single',
    'sparse_all_token': 'stage1_sparse_all_token',
    'sparse_topk': 'stage2_sparse_topk',
    'demand_supply': 'stage3_demand_supply',
    'sparse_attention': 'stage4_sparse_attention',
    'hybrid_late': 'stage5_hybrid_late',
}


def _warn(message):
    warnings.warn('[DDSH paper vis] %s' % message)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def setup_paper_style(cfg=None):
    """Set global matplotlib style for clean paper figures."""
    font_family = _cfg_get(cfg, 'font_family', 'Times New Roman')
    plt.rcParams.update({
        'font.family': font_family,
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def _ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def save_paper_figure(fig, save_dir, filename, writer=None, tag=None,
                      global_step=None, cfg=None):
    """Save one figure as PNG/SVG/PDF and optionally to TensorBoard."""
    setup_paper_style(cfg)
    _ensure_dir(save_dir)
    dpi = int(_cfg_get(cfg, 'dpi', 600))
    save_png = bool(_cfg_get(cfg, 'save_png', True))
    save_svg = bool(_cfg_get(cfg, 'save_svg', True))
    save_pdf = bool(_cfg_get(cfg, 'save_pdf', True))
    tensorboard = bool(_cfg_get(cfg, 'tensorboard', True))
    root, _ = os.path.splitext(filename)
    paths = []
    try:
        if writer is not None and tensorboard and tag is not None:
            try:
                writer.add_figure(tag, fig, global_step=global_step)
            except Exception as exc:
                _warn('TensorBoard add_figure failed for %s: %s' %
                      (tag, exc))
        for ext, enabled in [('png', save_png), ('svg', save_svg),
                             ('pdf', save_pdf)]:
            if not enabled:
                continue
            path = os.path.join(save_dir, '%s.%s' % (root, ext))
            try:
                fig.savefig(path, dpi=dpi, facecolor='white')
                paths.append(path)
            except Exception as exc:
                _warn('failed to save %s: %s' % (path, exc))
        return paths
    finally:
        plt.close(fig)


def _to_cpu(obj):
    """Recursively detach tensors and move them to CPU."""
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(value) for value in obj)
    return obj


def detach_debug_info(ddsh_debug):
    """Return a CPU-only copy of the DDSH debug dictionary."""
    return _to_cpu(ddsh_debug or {})


def _tokens(debug_info, key, fallback=None):
    value = debug_info.get(key, fallback)
    return value if isinstance(value, dict) else None


def _features(tokens):
    if not isinstance(tokens, dict):
        return None
    return tokens.get('features', tokens.get('feats', None))


def _coords(tokens):
    if not isinstance(tokens, dict):
        return None
    coords = tokens.get('coords', None)
    if coords is None:
        return None
    if hasattr(coords, 'detach'):
        coords = coords.detach().cpu().numpy()
    else:
        coords = np.asarray(coords)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return coords.reshape((-1, coords.shape[-1]))


def _scores(tokens, names=None):
    if not isinstance(tokens, dict):
        return None
    names = names or ['score', 'scores', 'demand_scores']
    for name in names:
        if name in tokens and tokens[name] is not None:
            score = tokens[name]
            if hasattr(score, 'detach'):
                score = score.detach().cpu().numpy()
            else:
                score = np.asarray(score)
            return score.reshape(-1)
    return None


def _sample(coords, values=None, max_points=30000):
    if coords is None:
        return coords, values
    if coords.shape[0] <= max_points:
        return coords, values
    idx = np.linspace(0, coords.shape[0] - 1, max_points).astype(np.int64)
    coords = coords[idx]
    values = values[idx] if values is not None and len(values) >= len(idx) \
        else values
    return coords, values


def _yx(coords):
    if coords is None or coords.shape[0] == 0:
        return None, None
    return coords[:, 2], coords[:, 1]


def _has_points(tokens):
    coords = _coords(tokens)
    return coords is not None and coords.shape[0] > 0


def _new_ax(title, xlabel='X index', ylabel='Y index', figsize=(5.2, 4.2),
            cfg=None):
    setup_paper_style(cfg)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.25)
    return fig, ax


def _scatter_tokens(ax, tokens, label, color, size=4, alpha=0.7,
                    marker='o', edgecolors='none', scores=None,
                    cmap='Reds', max_points=30000):
    coords = _coords(tokens)
    if coords is None or coords.shape[0] == 0:
        return None
    if scores is None:
        scores = _scores(tokens)
    coords, scores = _sample(coords, scores, max_points=max_points)
    x, y = _yx(coords)
    if scores is not None and len(scores) == coords.shape[0]:
        sc = ax.scatter(x, y, c=scores, s=size, alpha=alpha, marker=marker,
                        cmap=cmap, edgecolors=edgecolors, linewidths=0.4,
                        label=label)
        return sc
    return ax.scatter(x, y, c=color, s=size, alpha=alpha, marker=marker,
                      edgecolors=edgecolors, linewidths=0.4, label=label)


def _set_common_limits(ax, token_list, margin=8):
    coords_list = [_coords(tokens) for tokens in token_list
                   if isinstance(tokens, dict)]
    coords_list = [coords for coords in coords_list
                   if coords is not None and coords.shape[0] > 0]
    if not coords_list:
        return
    coords = np.concatenate(coords_list, axis=0)
    x = coords[:, 2]
    y = coords[:, 1]
    ax.set_xlim(float(x.min()) - margin, float(x.max()) + margin)
    ax.set_ylim(float(y.max()) + margin, float(y.min()) - margin)


def _maybe_save(fig, save_dir, filename, writer=None, tag=None,
                global_step=None, cfg=None):
    if save_dir is None or filename is None:
        return fig
    return save_paper_figure(fig, save_dir, filename, writer=writer,
                             tag=tag, global_step=global_step, cfg=cfg)


def visualize_sparse_tokens(debug_info=None, save_dir=None, filename=None,
                            writer=None, tag=None, global_step=None,
                            cfg=None, ego_tokens=None, helper_tokens=None,
                            fused_tokens=None):
    """Visualize sparse BEV token distributions."""
    debug_info = detach_debug_info(debug_info)
    ego_tokens = ego_tokens or _tokens(debug_info, 'ego_tokens')
    helper_tokens = helper_tokens or _tokens(debug_info,
                                             'helper_tokens_aligned',
                                             _tokens(debug_info,
                                                     'helper_tokens_original'))
    fused_tokens = fused_tokens or _tokens(debug_info, 'fused_tokens')
    if not any(_has_points(tokens) for tokens in
               [ego_tokens, helper_tokens, fused_tokens]):
        _warn('skip sparse_tokens: no sparse token coordinates available')
        return None
    fig, ax = _new_ax('Sparse BEV Token Distribution', cfg=cfg)
    _scatter_tokens(ax, ego_tokens, 'Ego sparse tokens', '#C8C8C8',
                    size=3, alpha=0.55)
    _scatter_tokens(ax, helper_tokens, 'Helper sparse tokens', '#4C78A8',
                    size=3, alpha=0.55)
    _scatter_tokens(ax, fused_tokens, 'Fused sparse tokens', '#7A3E9D',
                    size=5, alpha=0.75)
    _set_common_limits(ax, [ego_tokens, helper_tokens, fused_tokens])
    ax.legend(loc='best')
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def visualize_demand_tokens(debug_info=None, save_dir=None, filename=None,
                            writer=None, tag=None, global_step=None,
                            cfg=None, ego_tokens=None, demand_tokens=None):
    """Visualize ego demand tokens."""
    debug_info = detach_debug_info(debug_info)
    ego_tokens = ego_tokens or _tokens(debug_info, 'ego_tokens')
    demand_tokens = demand_tokens or _tokens(debug_info, 'demand_tokens')
    if not _has_points(ego_tokens) and not _has_points(demand_tokens):
        _warn('skip demand_tokens: no ego or demand tokens available')
        return None
    fig, ax = _new_ax('Ego Demand Token Generation', cfg=cfg)
    _scatter_tokens(ax, ego_tokens, 'Ego all tokens', '#C8C8C8',
                    size=3, alpha=0.45)
    score = _scores(demand_tokens, ['demand_scores', 'score', 'scores'])
    sc = _scatter_tokens(ax, demand_tokens, 'Demand tokens', '#D62728',
                         size=14, alpha=0.95, marker='o',
                         edgecolors='black', scores=score, cmap='Reds')
    if score is not None and sc is not None:
        cb = fig.colorbar(sc, ax=ax, shrink=0.82)
        cb.set_label('Demand score')
    _set_common_limits(ax, [ego_tokens, demand_tokens])
    ax.legend(loc='best')
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def visualize_supply_tokens(debug_info=None, save_dir=None, filename=None,
                            writer=None, tag=None, global_step=None,
                            cfg=None, helper_tokens_aligned=None,
                            supply_tokens=None, demand_tokens=None):
    """Visualize selected helper supply tokens in the ego frame."""
    debug_info = detach_debug_info(debug_info)
    helper_tokens_aligned = helper_tokens_aligned or _tokens(
        debug_info, 'helper_tokens_aligned')
    supply_tokens = supply_tokens or _tokens(debug_info, 'supply_tokens')
    demand_tokens = demand_tokens or _tokens(debug_info, 'demand_tokens')
    if not any(_has_points(tokens) for tokens in
               [helper_tokens_aligned, supply_tokens, demand_tokens]):
        _warn('skip supply_tokens: no aligned helper/supply/demand tokens')
        return None
    fig, ax = _new_ax('Demand-aware Helper Supply Selection', cfg=cfg)
    _scatter_tokens(ax, helper_tokens_aligned, 'Aligned helper tokens',
                    '#9ECAE1', size=3, alpha=0.55)
    _scatter_tokens(ax, supply_tokens, 'Selected supply tokens', '#F28E2B',
                    size=12, alpha=0.9)
    _scatter_tokens(ax, demand_tokens, 'Ego demand tokens', '#D62728',
                    size=22, alpha=0.95, marker='x')
    _set_common_limits(ax, [helper_tokens_aligned, supply_tokens,
                            demand_tokens])
    ax.legend(loc='best')
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def visualize_alignment(debug_info=None, save_dir=None, filename=None,
                        writer=None, tag=None, global_step=None, cfg=None,
                        helper_tokens_original=None,
                        helper_tokens_aligned=None, demand_tokens=None,
                        supply_tokens=None):
    """Visualize pose-aware helper token alignment before selection."""
    debug_info = detach_debug_info(debug_info)
    helper_tokens_original = helper_tokens_original or _tokens(
        debug_info, 'helper_tokens_original')
    helper_tokens_aligned = helper_tokens_aligned or _tokens(
        debug_info, 'helper_tokens_aligned')
    demand_tokens = demand_tokens or _tokens(debug_info, 'demand_tokens')
    supply_tokens = supply_tokens or _tokens(debug_info, 'supply_tokens')
    if not any(_has_points(tokens) for tokens in
               [helper_tokens_original, helper_tokens_aligned,
                demand_tokens, supply_tokens]):
        _warn('skip alignment: no alignment tokens available')
        return None
    fig, ax = _new_ax('Pose-aware Sparse Token Alignment', cfg=cfg)
    _scatter_tokens(ax, helper_tokens_original, 'Helper original tokens',
                    '#4C78A8', size=3, alpha=0.42)
    _scatter_tokens(ax, helper_tokens_aligned, 'Helper aligned tokens',
                    '#59A14F', size=4, alpha=0.55)
    _scatter_tokens(ax, demand_tokens, 'Ego demand tokens', '#D62728',
                    size=20, alpha=0.95, marker='x')
    _scatter_tokens(ax, supply_tokens, 'Selected supply tokens', '#F28E2B',
                    size=13, alpha=0.95)
    _set_common_limits(ax, [helper_tokens_original, helper_tokens_aligned,
                            demand_tokens, supply_tokens])
    ax.legend(loc='best')
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def visualize_demand_supply_matching(debug_info=None, save_dir=None,
                                      filename=None, writer=None, tag=None,
                                      global_step=None, cfg=None,
                                      demand_tokens=None,
                                      supply_tokens=None):
    """Visualize selected supply tokens and nearest ego demand tokens."""
    debug_info = detach_debug_info(debug_info)
    demand_tokens = demand_tokens or _tokens(debug_info, 'demand_tokens')
    supply_tokens = supply_tokens or _tokens(debug_info, 'supply_tokens')
    if not _has_points(demand_tokens) or not _has_points(supply_tokens):
        _warn('skip demand_supply_matching: demand or supply tokens missing')
        return None
    fig, ax = _new_ax('Demand-Supply Sparse Token Matching', cfg=cfg)
    _scatter_tokens(ax, demand_tokens, 'Ego demand tokens', '#D62728',
                    size=22, alpha=0.95, marker='x')
    _scatter_tokens(ax, supply_tokens, 'Selected supply tokens', '#F28E2B',
                    size=14, alpha=0.92)

    demand = _coords(demand_tokens)
    supply = _coords(supply_tokens)
    max_lines = int(_cfg_get(cfg, 'max_match_lines', 40))
    if demand is not None and supply is not None and demand.shape[0] > 0:
        s_yx = supply[:, 1:3].astype(np.float32)
        d_yx = demand[:, 1:3].astype(np.float32)
        dist = np.abs(s_yx[:, None, :] - d_yx[None, :, :]).sum(axis=2)
        nearest = dist.argmin(axis=1)
        if supply.shape[0] > max_lines:
            line_idx = np.linspace(0, supply.shape[0] - 1,
                                   max_lines).astype(np.int64)
        else:
            line_idx = np.arange(supply.shape[0])
        for idx in line_idx:
            sy, sx = supply[idx, 1], supply[idx, 2]
            dy, dx = demand[nearest[idx], 1], demand[nearest[idx], 2]
            ax.plot([sx, dx], [sy, dy], color='#888888', alpha=0.22,
                    linewidth=0.6)
        mean_dist = float(dist.min(axis=1).mean())
    else:
        mean_dist = None

    kd = 0 if demand is None else demand.shape[0]
    ks = 0 if supply is None else supply.shape[0]
    if mean_dist is None:
        text = r'$K_d$=%d, $K_s$=%d' % (kd, ks)
    else:
        text = r'$K_d$=%d, $K_s$=%d, mean dist=%.2f' % (kd, ks, mean_dist)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='0.75', alpha=0.92))
    _set_common_limits(ax, [demand_tokens, supply_tokens])
    ax.legend(loc='best')
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def visualize_topk_comparison(debug_info=None, save_dir=None, filename=None,
                              writer=None, tag=None, global_step=None,
                              cfg=None, topk_tokens=None,
                              demand_supply_tokens=None,
                              helper_tokens=None, demand_tokens=None):
    """Compare Stage2 Top-K supply and Stage3 Demand-Supply selection."""
    debug_info = detach_debug_info(debug_info)
    topk_tokens = topk_tokens or _tokens(debug_info, 'topk_tokens')
    demand_supply_tokens = demand_supply_tokens or _tokens(
        debug_info, 'supply_tokens')
    helper_tokens = helper_tokens or _tokens(debug_info,
                                             'helper_tokens_aligned')
    demand_tokens = demand_tokens or _tokens(debug_info, 'demand_tokens')
    if not _has_points(topk_tokens) and not _has_points(demand_supply_tokens):
        _warn('skip topk_comparison: topk and demand-supply tokens missing')
        return None
    setup_paper_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True,
                             sharey=True)
    for ax, selected, title in [
            (axes[0], topk_tokens, 'Stage2 Top-K Supply'),
            (axes[1], demand_supply_tokens, 'Stage3 Demand-Supply')]:
        ax.set_title(title)
        ax.set_xlabel('X index')
        ax.set_ylabel('Y index')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.25)
        _scatter_tokens(ax, helper_tokens, 'Aligned helper tokens',
                        '#D6EAF8', size=3, alpha=0.45)
        _scatter_tokens(ax, selected, 'Selected tokens', '#F28E2B',
                        size=12, alpha=0.9)
        _scatter_tokens(ax, demand_tokens, 'Demand tokens', '#D62728',
                        size=18, alpha=0.95, marker='x')
        _set_common_limits(ax, [helper_tokens, topk_tokens,
                                demand_supply_tokens, demand_tokens])
        ax.legend(loc='best')
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def visualize_sparse_attention_neighbors(debug_info=None, save_dir=None,
                                         filename=None, writer=None, tag=None,
                                         global_step=None, cfg=None,
                                         attention_info=None):
    """Visualize local sparse attention neighborhoods with a fallback view."""
    debug_info = detach_debug_info(debug_info)
    attention_info = attention_info or debug_info.get('attention_info', None)
    ego_tokens = _tokens(debug_info, 'ego_tokens')
    helper_tokens = _tokens(debug_info, 'helper_tokens_aligned')
    supply_tokens = _tokens(debug_info, 'supply_tokens')
    radius = None
    if isinstance(attention_info, dict):
        radius = attention_info.get('radius', None)
    if radius is None:
        radius = _cfg_get(cfg, 'attention_radius', 3)
    if not _has_points(ego_tokens) and not _has_points(helper_tokens):
        _warn('skip sparse_attention_neighbors: no attention tokens')
        return None
    fig, ax = _new_ax('Local Sparse Attention Neighborhood', cfg=cfg)
    _scatter_tokens(ax, ego_tokens, 'Ego query tokens', '#C8C8C8',
                    size=4, alpha=0.5)
    _scatter_tokens(ax, helper_tokens, 'Helper tokens in ego frame',
                    '#9ECAE1', size=4, alpha=0.55)
    _scatter_tokens(ax, supply_tokens, 'Highlighted neighbors', '#F28E2B',
                    size=13, alpha=0.9)

    query_coords = _coords(ego_tokens)
    if query_coords is not None and query_coords.shape[0] > 0:
        q = query_coords[query_coords.shape[0] // 2]
        ax.scatter([q[2]], [q[1]], c='#7A3E9D', s=45, marker='*',
                   label='Example query')
        ax.add_patch(Circle((q[2], q[1]), radius=float(radius),
                            edgecolor='#7A3E9D', facecolor='none',
                            linestyle='--', linewidth=1.2, alpha=0.8))
    _set_common_limits(ax, [ego_tokens, helper_tokens, supply_tokens])
    ax.legend(loc='best')
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def visualize_fusion_tokens(debug_info=None, save_dir=None, filename=None,
                            writer=None, tag=None, global_step=None,
                            cfg=None, ego_tokens=None, supply_tokens=None,
                            fused_tokens=None):
    """Visualize sparse tokens before and after fusion."""
    debug_info = detach_debug_info(debug_info)
    ego_tokens = ego_tokens or _tokens(debug_info, 'ego_tokens')
    supply_tokens = supply_tokens or _tokens(debug_info, 'supply_tokens')
    fused_tokens = fused_tokens or _tokens(debug_info, 'fused_tokens')
    if not any(_has_points(tokens) for tokens in
               [ego_tokens, supply_tokens, fused_tokens]):
        _warn('skip fusion_tokens: no fusion token coordinates available')
        return None
    fig, ax = _new_ax('Sparse Token Fusion', cfg=cfg)
    _scatter_tokens(ax, ego_tokens, 'Ego tokens', '#C8C8C8', size=3,
                    alpha=0.45)
    _scatter_tokens(ax, supply_tokens, 'Supply tokens', '#F28E2B', size=10,
                    alpha=0.88)
    _scatter_tokens(ax, fused_tokens, 'Fused tokens', '#7A3E9D', size=5,
                    alpha=0.72)
    _set_common_limits(ax, [ego_tokens, supply_tokens, fused_tokens])
    ax.legend(loc='best')
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def _boxes_np(boxes):
    if boxes is None:
        return None
    if hasattr(boxes, 'detach'):
        boxes = boxes.detach().cpu().numpy()
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    if boxes.ndim == 3 and boxes.shape[1:] == (8, 3):
        corners = boxes[:, :, :2]
        return corners
    if boxes.ndim == 2 and boxes.shape[1] >= 7:
        return boxes[:, :7]
    return None


def _draw_boxes(ax, boxes, label, color, linestyle='-', linewidth=1.2):
    boxes = _boxes_np(boxes)
    if boxes is None or boxes.shape[0] == 0:
        return False
    added = False
    for box in boxes[:100]:
        if box.ndim == 2:
            xy = box[:, :2]
        else:
            x, y, _, dx, dy, _, yaw = box[:7]
            c, s = np.cos(yaw), np.sin(yaw)
            local = np.array([[dx / 2, dy / 2], [dx / 2, -dy / 2],
                              [-dx / 2, -dy / 2], [-dx / 2, dy / 2]])
            rot = np.array([[c, -s], [s, c]])
            xy = local.dot(rot.T) + np.array([x, y])
        patch = Polygon(xy, closed=True, fill=False, edgecolor=color,
                        linestyle=linestyle, linewidth=linewidth,
                        label=label if not added else None)
        ax.add_patch(patch)
        added = True
    return added


def visualize_detection_result(debug_info=None, save_dir=None, filename=None,
                               writer=None, tag=None, global_step=None,
                               cfg=None, gt_boxes=None,
                               ego_pred_boxes=None,
                               fused_pred_boxes=None, late_boxes=None):
    """Visualize GT and prediction boxes when available."""
    debug_info = detach_debug_info(debug_info)
    gt_boxes = gt_boxes if gt_boxes is not None else debug_info.get('gt_boxes')
    ego_pred_boxes = ego_pred_boxes if ego_pred_boxes is not None else \
        debug_info.get('ego_pred_boxes')
    fused_pred_boxes = fused_pred_boxes if fused_pred_boxes is not None else \
        debug_info.get('fused_pred_boxes', debug_info.get('pred_boxes'))
    late_boxes = late_boxes if late_boxes is not None else \
        debug_info.get('late_boxes')
    if all(_boxes_np(boxes) is None or _boxes_np(boxes).shape[0] == 0
           for boxes in [gt_boxes, ego_pred_boxes, fused_pred_boxes,
                         late_boxes]):
        _warn('skip detection_result: no GT or prediction boxes available')
        return None
    fig, ax = _new_ax('Detection Results', xlabel='X (m)', ylabel='Y (m)',
                      cfg=cfg)
    any_box = False
    any_box |= _draw_boxes(ax, gt_boxes, 'GT boxes', 'black', '-')
    any_box |= _draw_boxes(ax, ego_pred_boxes, 'Ego-only prediction',
                           '#4C78A8', '--')
    any_box |= _draw_boxes(ax, fused_pred_boxes, 'DDSH fused prediction',
                           '#D62728', '-')
    any_box |= _draw_boxes(ax, late_boxes, 'Helper late boxes',
                           '#59A14F', ':')
    if not any_box:
        plt.close(fig)
        _warn('skip detection_result: box format is unsupported')
        return None
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best')
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def visualize_late_compensation(debug_info=None, save_dir=None, filename=None,
                                writer=None, tag=None, global_step=None,
                                cfg=None, intermediate_boxes=None,
                                late_boxes=None, final_boxes=None):
    """Visualize Stage5 late compensation boxes."""
    debug_info = detach_debug_info(debug_info)
    intermediate_boxes = intermediate_boxes if intermediate_boxes is not None \
        else debug_info.get('pred_boxes')
    late_boxes = late_boxes if late_boxes is not None else \
        debug_info.get('late_boxes')
    final_boxes = final_boxes if final_boxes is not None else \
        debug_info.get('final_boxes')
    if all(_boxes_np(boxes) is None or _boxes_np(boxes).shape[0] == 0
           for boxes in [intermediate_boxes, late_boxes, final_boxes]):
        _warn('skip late_compensation: required boxes are unavailable')
        return None
    fig, ax = _new_ax('Hybrid Late Compensation', xlabel='X (m)',
                      ylabel='Y (m)', cfg=cfg)
    _draw_boxes(ax, intermediate_boxes, 'Sparse intermediate prediction',
                '#4C78A8', '--')
    _draw_boxes(ax, late_boxes, 'Helper late boxes', '#59A14F', ':')
    _draw_boxes(ax, final_boxes, 'Final boxes after compensation',
                '#D62728', '-')
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='best')
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def plot_token_statistics(debug_info=None, save_dir=None, filename=None,
                          writer=None, tag=None, global_step=None, cfg=None):
    """Plot token count changes across DDSH stages."""
    debug_info = detach_debug_info(debug_info)
    comm = debug_info.get('communication', {}) or {}
    labels = ['Ego', 'Helper', 'Demand', 'Supply', 'Fused']
    values = [
        int(comm.get('num_ego_tokens', 0)),
        int(comm.get('num_helper_tokens', 0)),
        int(comm.get('num_demand_tokens', 0)),
        int(comm.get('num_supply_tokens', 0)),
        int(comm.get('num_fused_tokens', 0)),
    ]
    if sum(values) == 0:
        _warn('skip token_statistics: all token counts are zero/missing')
        return None
    setup_paper_style(cfg)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.bar(labels, values, color=['#BDBDBD', '#9ECAE1', '#D62728',
                                  '#F28E2B', '#7A3E9D'])
    ax.set_title('Sparse Token Statistics')
    ax.set_ylabel('Number of tokens')
    ax.grid(True, axis='y', alpha=0.25)
    for idx, val in enumerate(values):
        ax.text(idx, val, str(val), ha='center', va='bottom', fontsize=9)
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def plot_communication_statistics(debug_info=None, save_dir=None,
                                  filename=None, writer=None, tag=None,
                                  global_step=None, cfg=None):
    """Plot DDSH communication bytes."""
    debug_info = detach_debug_info(debug_info)
    comm = debug_info.get('communication', {}) or {}
    helper_bytes = comm.get('per_helper_supply_bytes', None)
    if helper_bytes is not None:
        helper_bytes = np.asarray(helper_bytes, dtype=np.float64)
    labels = ['Demand request', 'Supply feature', 'Total']
    values = [
        float(comm.get('demand_bytes', 0.0)),
        float(comm.get('supply_bytes', 0.0)),
        float(comm.get('total_bytes', 0.0)),
    ]
    if helper_bytes is None and sum(values) == 0:
        _warn('skip communication_statistics: communication fields missing')
        return None
    setup_paper_style(cfg)
    fig, axes = plt.subplots(1, 2 if helper_bytes is not None else 1,
                             figsize=(8.0, 3.5)
                             if helper_bytes is not None else (5.0, 3.5))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes[0].bar(labels, values, color=['#D62728', '#F28E2B', '#7A3E9D'])
    axes[0].set_title('Communication Budget')
    axes[0].set_ylabel('Bytes')
    axes[0].grid(True, axis='y', alpha=0.25)
    if helper_bytes is not None:
        h_labels = ['H%d' % (idx + 1) for idx in range(len(helper_bytes))]
        axes[1].bar(h_labels, helper_bytes, color='#4C78A8')
        axes[1].set_title('Per-helper Supply Bytes')
        axes[1].set_ylabel('Bytes')
        axes[1].grid(True, axis='y', alpha=0.25)
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def plot_stage_comparison(stage_records=None, save_dir=None, filename=None,
                          writer=None, tag=None, global_step=None, cfg=None):
    """Plot stage-level comparison for AP, communication or token counts."""
    records = stage_records or []
    if isinstance(records, dict):
        records = records.get('stage_records', [])
    if not records:
        _warn('skip stage_comparison: no stage records available')
        return None
    stages = [str(item.get('stage', idx)) for idx, item in enumerate(records)]
    ap = [item.get('AP', item.get('ap', None)) for item in records]
    comm = [float(item.get('communication_bytes', 0.0)) for item in records]
    tokens = [float(item.get('token_count', 0.0)) for item in records]
    setup_paper_style(cfg)
    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(stages))
    if any(v is not None for v in ap):
        ap_values = [0.0 if v is None else float(v) for v in ap]
        ax1.plot(x, ap_values, color='#D62728', marker='o', label='AP')
        ax1.set_ylabel('AP')
    else:
        ax1.plot(x, tokens, color='#7A3E9D', marker='o',
                 label='Token count')
        ax1.set_ylabel('Token count')
    ax2 = ax1.twinx()
    ax2.plot(x, comm, color='#4C78A8', marker='s',
             label='Communication bytes')
    ax2.set_ylabel('Communication bytes')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, rotation=20, ha='right')
    ax1.set_title('DDSH Stage Comparison')
    ax1.grid(True, alpha=0.25)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
    return _maybe_save(fig, save_dir, filename, writer, tag, global_step, cfg)


def _filename(prefix, figure_type):
    return '%s_%s' % (prefix, figure_type)


def visualize_ddsh_debug(debug_info, save_dir, prefix, writer=None,
                         global_step=None, cfg=None):
    """Generate all enabled DDSH paper figures for one debug payload."""
    cfg = cfg or {}
    if not bool(_cfg_get(cfg, 'enable', True)):
        return []
    debug_info = detach_debug_info(debug_info)
    generated = []

    def call(flag, figure_type, fn, *args):
        if not bool(_cfg_get(cfg, flag, True)):
            return
        try:
            result = fn(debug_info, save_dir=save_dir,
                        filename=_filename(prefix, figure_type),
                        writer=writer,
                        tag='DDSH/%s' % figure_type,
                        global_step=global_step,
                        cfg=cfg, *args)
            if result is not None:
                generated.append(figure_type)
        except Exception as exc:
            _warn('%s failed: %s' % (figure_type, exc))

    call('draw_sparse_tokens', 'sparse_tokens', visualize_sparse_tokens)
    call('draw_demand', 'demand_tokens', visualize_demand_tokens)
    call('draw_supply', 'supply_tokens', visualize_supply_tokens)
    call('draw_alignment', 'alignment', visualize_alignment)
    call('draw_demand_supply_matching', 'demand_supply_matching',
         visualize_demand_supply_matching)
    call('draw_topk_comparison', 'topk_comparison',
         visualize_topk_comparison)
    call('draw_attention_neighbors', 'attention_neighbors',
         visualize_sparse_attention_neighbors)
    call('draw_fusion_tokens', 'fusion_tokens', visualize_fusion_tokens)
    call('draw_detection_result', 'detection_result',
         visualize_detection_result)
    call('draw_late_compensation', 'late_compensation',
         visualize_late_compensation)
    call('draw_token_statistics', 'token_statistics',
         plot_token_statistics)
    call('draw_communication_statistics', 'communication_statistics',
         plot_communication_statistics)
    if bool(_cfg_get(cfg, 'draw_stage_comparison', True)):
        try:
            records = debug_info.get('stage_records', None)
            result = plot_stage_comparison(
                records, save_dir=save_dir,
                filename=_filename(prefix, 'stage_comparison'),
                writer=writer, tag='DDSH/stage_comparison',
                global_step=global_step, cfg=cfg)
            if result is not None:
                generated.append('stage_comparison')
        except Exception as exc:
            _warn('stage_comparison failed: %s' % exc)
    return generated
