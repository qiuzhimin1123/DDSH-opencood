import torch
import torch.nn as nn

from opencood.models.sub_modules.ddsh.demand_generator import (
    SparseDemandGenerator,
)
from opencood.models.sub_modules.ddsh.pose_align import SparsePoseAligner
from opencood.models.sub_modules.ddsh.sparse_fusion import SparseTokenFusion
from opencood.models.sub_modules.ddsh.sparse_local_attention import (
    SparseLocalAttention,
)
from opencood.models.sub_modules.ddsh.sparse_tensor_utils import (
    make_tokens,
    select_batch,
)
from opencood.models.sub_modules.ddsh.supply_selector import (
    SparseSupplySelector,
)


class DemandDrivenSparseHybrid(nn.Module):
    VALID_STAGES = {
        'sparse_single',
        'sparse_all_token',
        'sparse_topk',
        'demand_supply',
        'sparse_attention',
        'hybrid_late',
    }

    def __init__(self, in_channels, voxel_size, lidar_range, feature_stride,
                 model_cfg):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.feature_stride = int(feature_stride)
        self.stage = self.model_cfg.get('stage', 'demand_supply')
        if self.stage not in self.VALID_STAGES:
            raise ValueError('Unsupported DDSH stage "%s". Valid stages: %s' %
                             (self.stage, sorted(self.VALID_STAGES)))
        self.use_demand = bool(self.model_cfg.get('use_demand', True))
        self.use_helpers = bool(self.model_cfg.get('use_helpers', True))
        self.demand_generator = SparseDemandGenerator(
            in_channels, self.model_cfg.get('demand', {}))
        self.pose_aligner = SparsePoseAligner(
            voxel_size, lidar_range, self.feature_stride,
            self.model_cfg.get('pose_align', {}))
        self.supply_selector = SparseSupplySelector(
            in_channels, self.model_cfg.get('supply', {}))
        self.fusion = SparseTokenFusion(in_channels,
                                        self.model_cfg.get('fusion', {}))
        self.local_attention = SparseLocalAttention(
            in_channels, self.model_cfg.get('local_attention', {}))

    @staticmethod
    def _record_len_to_list(record_len):
        if isinstance(record_len, torch.Tensor):
            return [int(x) for x in record_len.detach().cpu().tolist()]
        return [int(x) for x in record_len]

    @staticmethod
    def _require_pairwise(pairwise_t_matrix, batch_size, max_cav,
                          need_helpers=True):
        if not need_helpers:
            return
        if pairwise_t_matrix is None:
            if max_cav > 1:
                raise KeyError(
                    'DDSH requires pairwise_t_matrix for helper alignment '
                    'when record_len contains helpers.')
            return
        if pairwise_t_matrix.dim() != 5 or pairwise_t_matrix.shape[-2:] != (4, 4):
            raise ValueError(
                'DDSH expects pairwise_t_matrix with shape '
                '[B, max_cav, max_cav, 4, 4], got %s.' %
                (tuple(pairwise_t_matrix.shape),))
        if pairwise_t_matrix.shape[0] < batch_size:
            raise ValueError('pairwise_t_matrix batch dimension is smaller '
                             'than record_len length.')

    def _stage_policy(self):
        policy = {
            'use_helpers': self.use_helpers,
            'use_demand': self.use_demand,
            'supply_mode': self.supply_selector.mode,
            'force_attention': False,
        }

        if self.stage == 'sparse_single':
            policy.update({
                'use_helpers': False,
                'use_demand': False,
                'supply_mode': 'none',
            })
        elif self.stage == 'sparse_all_token':
            policy.update({
                'use_helpers': True,
                'use_demand': False,
                'supply_mode': 'all',
            })
        elif self.stage == 'sparse_topk':
            policy.update({
                'use_helpers': True,
                'use_demand': False,
                'supply_mode': 'topk',
            })
        elif self.stage == 'demand_supply':
            policy.update({
                'use_helpers': True,
                'use_demand': True,
                'supply_mode': 'demand_radius',
            })
        elif self.stage == 'sparse_attention':
            policy.update({
                'use_helpers': True,
                'use_demand': False,
                'supply_mode': 'all',
                'force_attention': True,
            })
        elif self.stage == 'hybrid_late':
            policy.update({
                'use_helpers': True,
                'use_demand': True,
                'supply_mode': 'demand_radius',
            })
        return policy

    def _count_tokens_by_cav(self, bev_tokens, total_cav):
        coords = bev_tokens['coords']
        return [int((coords[:, 0] == cav_idx).sum().item())
                for cav_idx in range(total_cav)]

    @staticmethod
    def _coord_range(coords):
        if coords is None or coords.numel() == 0:
            return {'min': None, 'max': None}
        coords = coords.detach()
        return {
            'min': [int(v) for v in coords.min(dim=0).values.cpu().tolist()],
            'max': [int(v) for v in coords.max(dim=0).values.cpu().tolist()],
        }

    @staticmethod
    def _append_token(store, key, tokens):
        if store is None:
            return
        if tokens is None:
            return
        if key not in store:
            store[key] = []
        store[key].append(tokens)

    @staticmethod
    def _merge_token_list(token_list, spatial_shape, batch_size, reference):
        token_list = [tokens for tokens in token_list
                      if tokens is not None and
                      tokens['features'].shape[0] > 0]
        if len(token_list) == 0:
            channels = reference['features'].shape[1]
            features = reference['features'].new_zeros((0, channels))
            coords = reference['coords'].new_zeros((0, 3))
            return make_tokens(features, coords, spatial_shape, batch_size)
        features = torch.cat([tokens['features'] for tokens in token_list],
                             dim=0)
        coords = torch.cat([tokens['coords'] for tokens in token_list], dim=0)
        merged = make_tokens(features, coords, spatial_shape, batch_size)
        for score_key in ['demand_scores', 'score', 'scores']:
            values = [tokens[score_key] for tokens in token_list
                      if score_key in tokens and tokens[score_key] is not None]
            if len(values) == len(token_list):
                merged[score_key] = torch.cat(values, dim=0)
        return merged

    @staticmethod
    def _estimate_bytes(num_tokens, feature_dim, dtype, coord_dim=3):
        if hasattr(dtype, 'itemsize'):
            elem_size = dtype.itemsize
        elif dtype in [torch.float16, torch.bfloat16]:
            elem_size = 2
        elif dtype in [torch.float64, torch.long, torch.int64]:
            elem_size = 8
        else:
            elem_size = 4
        coord_size = 4
        return int(num_tokens) * (int(feature_dim) * elem_size +
                                  int(coord_dim) * coord_size)

    def _finalize_debug(self, debug_store, fused, stats, scene_batch_size,
                        bev_tokens):
        if debug_store is None:
            return None
        feature_dim = int(bev_tokens['features'].shape[1])
        dtype = bev_tokens['features'].dtype
        demand_tokens = self._merge_token_list(
            debug_store.get('demand_tokens', []), bev_tokens['spatial_shape'],
            scene_batch_size, bev_tokens)
        supply_tokens = self._merge_token_list(
            debug_store.get('supply_tokens', []), bev_tokens['spatial_shape'],
            scene_batch_size, bev_tokens)
        helper_original = self._merge_token_list(
            debug_store.get('helper_tokens_original', []),
            bev_tokens['spatial_shape'], scene_batch_size, bev_tokens)
        helper_aligned = self._merge_token_list(
            debug_store.get('helper_tokens_aligned', []),
            bev_tokens['spatial_shape'], scene_batch_size, bev_tokens)
        ego_tokens = self._merge_token_list(
            debug_store.get('ego_tokens', []), bev_tokens['spatial_shape'],
            scene_batch_size, bev_tokens)
        topk_tokens = supply_tokens if self.stage == 'sparse_topk' else None
        demand_bytes = self._estimate_bytes(
            demand_tokens['features'].shape[0], 1, dtype, coord_dim=3)
        supply_bytes = self._estimate_bytes(
            supply_tokens['features'].shape[0], feature_dim, dtype,
            coord_dim=3)
        per_helper_supply_bytes = [
            self._estimate_bytes(count, feature_dim, dtype, coord_dim=3)
            for count in debug_store.get('per_helper_supply_counts', [])
        ]
        attention_info = {
            'radius': self.local_attention.radius
            if hasattr(self.local_attention, 'radius') else None,
            'note': 'Fallback visualization uses sparse ego/helper tokens.',
        } if self.stage in ['sparse_attention', 'hybrid_late'] else None
        return {
            'stage': self.stage,
            'scene_id': 0,
            'ego_tokens': ego_tokens,
            'helper_tokens_original': helper_original,
            'helper_tokens_aligned': helper_aligned,
            'demand_tokens': demand_tokens,
            'supply_tokens': supply_tokens,
            'fused_tokens': fused,
            'topk_tokens': topk_tokens,
            'attention_info': attention_info,
            'communication': {
                'demand_bytes': demand_bytes,
                'supply_bytes': supply_bytes,
                'total_bytes': demand_bytes + supply_bytes,
                'per_helper_supply_bytes': per_helper_supply_bytes,
                'num_ego_tokens': stats.get('num_ego_tokens', 0),
                'num_helper_tokens': sum(
                    debug_store.get('per_helper_raw_counts', [])),
                'num_demand_tokens': stats.get('num_demand_tokens', 0),
                'num_supply_tokens': stats.get('num_supply_tokens', 0),
                'num_fused_tokens': stats.get('num_fused_tokens', 0),
                'feature_dim': feature_dim,
            },
            'pred_boxes': None,
            'ego_pred_boxes': None,
            'fused_pred_boxes': None,
            'late_boxes': None,
            'final_boxes': None,
            'gt_boxes': None,
            'stage_records': [{
                'stage': self.stage,
                'token_count': stats.get('num_fused_tokens', 0),
                'communication_bytes': demand_bytes + supply_bytes,
            }],
        }

    def forward(self, bev_tokens, record_len, pairwise_t_matrix=None,
                collect_debug=False):
        record_len_list = self._record_len_to_list(record_len)
        scene_batch_size = len(record_len_list)
        max_cav = max(record_len_list) if record_len_list else 0
        total_cav = sum(record_len_list)
        policy = self._stage_policy()
        self._require_pairwise(pairwise_t_matrix, scene_batch_size, max_cav,
                               need_helpers=policy['use_helpers'])

        token_groups = []
        debug_store = None
        if collect_debug:
            debug_store = {
                'ego_tokens': [],
                'demand_tokens': [],
                'helper_tokens_original': [],
                'helper_tokens_aligned': [],
                'supply_tokens': [],
                'per_helper_supply_counts': [],
                'per_helper_raw_counts': [],
            }
        stats = {
            'stage': self.stage,
            'per_cav_token_count': self._count_tokens_by_cav(
                bev_tokens, total_cav),
            'ego_demand_token_count': [],
            'helper_supply_token_count': [],
            'helper_aligned_token_count': [],
            'helper_debug': [],
            'ego_demand_coord_range': [],
            'num_ego_tokens': 0,
            'num_demand_tokens': 0,
            'num_supply_tokens': 0,
            'num_helpers': 0,
            'fusion_input_tokens': 0,
            'fusion_match_token_count': 0,
        }

        cav_offset = 0
        for scene_idx, cav_count in enumerate(record_len_list):
            if cav_count <= 0:
                cav_offset += cav_count
                continue

            ego_tokens = select_batch(bev_tokens, cav_offset,
                                      out_batch_idx=scene_idx)
            if policy['use_demand']:
                demand_tokens = self.demand_generator(ego_tokens)
            else:
                demand_tokens = make_tokens(
                    ego_tokens['features'][:0],
                    ego_tokens['coords'][:0],
                    ego_tokens['spatial_shape'],
                    ego_tokens['batch_size'])
            token_groups.append(ego_tokens)
            self._append_token(debug_store, 'ego_tokens', ego_tokens)
            stats['num_ego_tokens'] += int(ego_tokens['features'].shape[0])
            demand_count = int(demand_tokens['features'].shape[0])
            stats['num_demand_tokens'] += demand_count
            stats['ego_demand_token_count'].append({
                'scene_idx': scene_idx,
                'count': demand_count,
            })
            demand_coord_range = self._coord_range(demand_tokens['coords'])
            stats['ego_demand_coord_range'].append({
                'scene_idx': scene_idx,
                'min': demand_coord_range['min'],
                'max': demand_coord_range['max'],
            })
            self._append_token(debug_store, 'demand_tokens', demand_tokens)

            helper_range = range(1, cav_count) if policy['use_helpers'] else []
            for helper_local_idx in helper_range:
                helper_global_idx = cav_offset + helper_local_idx
                helper_tokens = select_batch(bev_tokens, helper_global_idx)
                if helper_tokens['features'].shape[0] == 0:
                    continue
                matrix = pairwise_t_matrix[scene_idx, helper_local_idx, 0]
                # Explicit DDSH coordinate order:
                # helper tokens are first aligned into ego BEV coordinates;
                # supply selection and demand matching then run in ego frame.
                helper_tokens_ego = self.pose_aligner(
                    helper_tokens, matrix, scene_idx,
                    batch_size=demand_tokens['batch_size'])
                self._append_token(debug_store, 'helper_tokens_original',
                                   helper_tokens)
                self._append_token(debug_store, 'helper_tokens_aligned',
                                   helper_tokens_ego)
                supply_tokens = self.supply_selector(
                    helper_tokens_ego, demand_tokens,
                    mode_override=policy['supply_mode'])
                aligned_count = int(supply_tokens.get('aligned_count', 0))
                supply_count = int(supply_tokens['features'].shape[0])
                stats['helper_aligned_token_count'].append({
                    'scene_idx': scene_idx,
                    'helper_idx': helper_local_idx,
                    'count': aligned_count,
                })
                stats['helper_supply_token_count'].append({
                    'scene_idx': scene_idx,
                    'helper_idx': helper_local_idx,
                    'count': supply_count,
                })
                stats['helper_debug'].append({
                    'scene_idx': scene_idx,
                    'helper_idx': helper_local_idx,
                    'raw_count': int(helper_tokens['features'].shape[0]),
                    'aligned_count': aligned_count,
                    'supply_count': supply_count,
                    'ego_demand_coords_min': demand_coord_range['min'],
                    'ego_demand_coords_max': demand_coord_range['max'],
                    'helper_original_coords_min': self._coord_range(
                        helper_tokens['coords'])['min'],
                    'helper_original_coords_max': self._coord_range(
                        helper_tokens['coords'])['max'],
                    'helper_ego_coords_min': self._coord_range(
                        helper_tokens_ego['coords'])['min'],
                    'helper_ego_coords_max': self._coord_range(
                        helper_tokens_ego['coords'])['max'],
                    'demand_match_distance_min': supply_tokens.get(
                        'demand_match_distance_min'),
                    'demand_match_distance_mean': supply_tokens.get(
                        'demand_match_distance_mean'),
                    'demand_match_distance_max': supply_tokens.get(
                        'demand_match_distance_max'),
                })
                if debug_store is not None:
                    debug_store['per_helper_raw_counts'].append(
                        int(helper_tokens['features'].shape[0]))
                    debug_store['per_helper_supply_counts'].append(
                        supply_count)
                self._append_token(debug_store, 'supply_tokens',
                                   supply_tokens)
                if supply_tokens['features'].shape[0] > 0:
                    token_groups.append(supply_tokens)
                    stats['num_supply_tokens'] += int(
                        supply_tokens['features'].shape[0])
                stats['num_helpers'] += 1

            cav_offset += cav_count

        stats['fusion_input_tokens'] = sum(
            int(tokens['features'].shape[0]) for tokens in token_groups)

        if len(token_groups) == 0:
            channels = bev_tokens['features'].shape[1]
            empty_features = bev_tokens['features'].new_zeros((0, channels))
            empty_coords = bev_tokens['coords'].new_zeros((0, 3))
            fused = make_tokens(empty_features, empty_coords,
                                bev_tokens['spatial_shape'],
                                scene_batch_size)
        else:
            fused = self.fusion(token_groups, bev_tokens['spatial_shape'],
                                scene_batch_size)

        fused = self.local_attention(
            fused, force=policy['force_attention'])
        stats['num_fused_tokens'] = int(fused['features'].shape[0])
        stats['fusion_match_token_count'] = max(
            0, stats['fusion_input_tokens'] - stats['num_fused_tokens'])
        if collect_debug:
            stats['ddsh_debug'] = self._finalize_debug(
                debug_store, fused, stats, scene_batch_size, bev_tokens)
        return fused, stats
