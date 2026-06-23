# -*- coding: utf-8 -*-
# DDSH-VoxelNeXt: Demand-Driven Sparse Hybrid VoxelNeXt.

import torch
import torch.nn as nn

from opencood.models.sub_modules.ddsh import (
    DemandDrivenSparseHybrid,
    DdshVoxelNeXtSparseHead,
    LateBoxCompensation,
    SparseHeightCompression,
)
from opencood.models.sub_modules.ddsh.sparse_tensor_utils import (
    assert_no_dense_bev,
    find_dense_bev_keys,
    tokens_to_sparse_tensor,
)
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.voxelnext_backbone import (
    VoxelResBackBone8xVoxelNeXt,
)


class DDSHVoxelNeXt(nn.Module):
    def __init__(self, args):
        super(DDSHVoxelNeXt, self).__init__()
        self.args = args
        self.num_point_features = args.get('num_point_features', 4)
        self.voxel_size = args['voxel_size']
        self.lidar_range = args['lidar_range']
        self.grid_size = args['grid_size']
        self.class_names = args.get('class_names', ['vehicle'])
        self.ddsh_cfg = args.get('ddsh', {})
        self.stage = self.ddsh_cfg.get('stage', 'demand_supply')
        self.debug = bool(self.ddsh_cfg.get('debug', False))
        if len(self.class_names) != 1 and not args.get('class_labels_key'):
            raise ValueError(
                'OpenCOOD batches in this repository do not provide per-box '
                'class labels. Configure one class or add class_labels_key.')

        self.mean_vfe = MeanVFE(args.get('mean_vfe', {}),
                                self.num_point_features)
        self.backbone_3d = VoxelResBackBone8xVoxelNeXt(
            args['backbone_3d'],
            self.num_point_features,
            self.grid_size)

        out_channel = args['backbone_3d'].get('OUT_CHANNEL', 128)
        sparse_head_cfg = args.get('sparse_head', {})
        target_cfg = sparse_head_cfg.get('TARGET_ASSIGNER_CONFIG', {})
        self.feature_stride = int(target_cfg.get(
            'FEATURE_MAP_STRIDE', args.get('feature_stride', 8)))

        self.sparse_height_compression = SparseHeightCompression(
            args.get('sparse_height_compression', {}))
        self.ddsh = DemandDrivenSparseHybrid(
            out_channel,
            self.voxel_size,
            self.lidar_range,
            self.feature_stride,
            self.ddsh_cfg)
        self.sparse_head = DdshVoxelNeXtSparseHead(
            sparse_head_cfg,
            input_channels=out_channel,
            num_class=len(self.class_names),
            class_names=self.class_names,
            grid_size=self.grid_size,
            point_cloud_range=self.lidar_range,
            voxel_size=self.voxel_size,
            predict_boxes_when_training=bool(
                sparse_head_cfg.get('PREDICT_BOXES_WHEN_TRAINING', False)),
        )
        late_cfg = dict(args.get('late_compensation', {}))
        if self.stage == 'hybrid_late':
            late_cfg['enabled'] = True
        self.late_compensation = LateBoxCompensation(late_cfg)
        self.box_order = args.get('box_order', 'hwl')
        self.class_labels_key = args.get('class_labels_key', None)

    @staticmethod
    def _require_processed_lidar(data_dict):
        if 'processed_lidar' not in data_dict:
            raise KeyError('DDSH-VoxelNeXt requires processed_lidar.')
        processed = data_dict['processed_lidar']
        required = ['voxel_features', 'voxel_coords', 'voxel_num_points']
        missing = [key for key in required if key not in processed]
        if missing:
            raise KeyError('processed_lidar is missing keys: %s' % missing)
        return processed

    @staticmethod
    def _pairwise_matrix(data_dict):
        for key in ['pairwise_t_matrix', 'pairwise_transformation_matrix',
                    't_matrix']:
            if key in data_dict:
                return data_dict[key]
        return None

    @staticmethod
    def _record_len_to_list(record_len):
        if isinstance(record_len, torch.Tensor):
            return [int(x) for x in record_len.detach().cpu().tolist()]
        return [int(x) for x in record_len]

    @staticmethod
    def _collect_pred_boxes(final_box_dicts):
        if not final_box_dicts:
            return None
        boxes = []
        for item in final_box_dicts:
            if isinstance(item, dict) and 'pred_boxes' in item:
                cur_boxes = item['pred_boxes']
                if cur_boxes is not None and cur_boxes.shape[0] > 0:
                    boxes.append(cur_boxes)
        if len(boxes) == 0:
            return None
        return torch.cat(boxes, dim=0)

    def _debug_print(self, record_len, ddsh_stats, dense_keys):
        if not self.debug:
            return
        record_len_list = self._record_len_to_list(record_len)
        print('[DDSH][debug] stage: %s' % self.stage)
        print('[DDSH][debug] record_len: %s' % record_len_list)
        print('[DDSH][debug] dense BEV fields: %s' %
              (dense_keys if dense_keys else 'none'))

        cav_counts = ddsh_stats.get('per_cav_token_count', [])
        cav_offset = 0
        for scene_idx, cav_count in enumerate(record_len_list):
            for local_idx in range(cav_count):
                global_idx = cav_offset + local_idx
                role = 'ego' if local_idx == 0 else 'helper'
                count = cav_counts[global_idx] \
                    if global_idx < len(cav_counts) else 0
                print('[DDSH][debug] scene %d cav %d (%s) sparse tokens: %d'
                      % (scene_idx, local_idx, role, count))
            cav_offset += cav_count

        for item in ddsh_stats.get('ego_demand_token_count', []):
            print('[DDSH][debug] scene %d ego demand tokens: %d' %
                  (item['scene_idx'], item['count']))
        for item in ddsh_stats.get('ego_demand_coord_range', []):
            print('[DDSH][debug] scene %d ego_demand coords min/max: %s / %s'
                  % (item['scene_idx'], item.get('min'), item.get('max')))
        for item in ddsh_stats.get('helper_aligned_token_count', []):
            print('[DDSH][debug] scene %d helper %d aligned tokens: %d' %
                  (item['scene_idx'], item['helper_idx'], item['count']))
        for item in ddsh_stats.get('helper_supply_token_count', []):
            print('[DDSH][debug] scene %d helper %d supply tokens: %d' %
                  (item['scene_idx'], item['helper_idx'], item['count']))
        print('[DDSH][debug] fusion input tokens: %d' %
              ddsh_stats.get('fusion_input_tokens', 0))
        print('[DDSH][debug] fusion matched tokens: %d' %
              ddsh_stats.get('fusion_match_token_count', 0))
        print('[DDSH][debug] fused sparse tokens: %d' %
              ddsh_stats.get('num_fused_tokens', 0))
        for item in ddsh_stats.get('helper_debug', []):
            print('[DDSH][debug] scene %d helper %d original coords min/max: '
                  '%s / %s' %
                  (item['scene_idx'], item['helper_idx'],
                   item.get('helper_original_coords_min'),
                   item.get('helper_original_coords_max')))
            print('[DDSH][debug] scene %d helper %d ego coords min/max: '
                  '%s / %s' %
                  (item['scene_idx'], item['helper_idx'],
                   item.get('helper_ego_coords_min'),
                   item.get('helper_ego_coords_max')))
            print('[DDSH][debug] scene %d helper %d demand matching '
                  'distance min/mean/max: %s / %s / %s' %
                  (item['scene_idx'], item['helper_idx'],
                   item.get('demand_match_distance_min'),
                   item.get('demand_match_distance_mean'),
                   item.get('demand_match_distance_max')))
            print('[DDSH][debug] scene %d helper %d selected supply token '
                  'count: %d' %
                  (item['scene_idx'], item['helper_idx'],
                   item.get('supply_count', 0)))

    def _build_gt_boxes(self, data_dict):
        if 'object_bbx_center' not in data_dict or 'object_bbx_mask' not in \
                data_dict:
            raise KeyError(
                'DDSH sparse head training requires object_bbx_center and '
                'object_bbx_mask from the OpenCOOD dataset.')

        boxes = data_dict['object_bbx_center'].float()
        masks = data_dict['object_bbx_mask'].bool()
        if boxes.dim() != 3 or boxes.shape[-1] != 7:
            raise ValueError(
                'object_bbx_center must have shape [B, M, 7], got %s.' %
                (tuple(boxes.shape),))

        if self.box_order == 'hwl':
            boxes_lwh = boxes[:, :, [0, 1, 2, 5, 4, 3, 6]]
        elif self.box_order == 'lwh':
            boxes_lwh = boxes
        else:
            raise ValueError('Unsupported DDSH box_order: %s' %
                             self.box_order)

        if self.class_labels_key is not None:
            if self.class_labels_key not in data_dict:
                raise KeyError('class_labels_key "%s" not found in data_dict.'
                               % self.class_labels_key)
            labels = data_dict[self.class_labels_key].long()
        else:
            labels = torch.ones(boxes.shape[:2], device=boxes.device,
                                dtype=torch.long)
        labels = labels.masked_fill(~masks, 0).unsqueeze(-1).float()
        boxes_lwh = boxes_lwh.masked_fill(~masks.unsqueeze(-1), 0)
        return torch.cat([boxes_lwh, labels], dim=-1)

    def forward(self, data_dict):
        processed = self._require_processed_lidar(data_dict)
        if 'record_len' not in data_dict:
            raise KeyError('DDSH-VoxelNeXt requires record_len.')
        record_len = data_dict['record_len']

        batch_dict = {
            'voxel_features': processed['voxel_features'],
            'voxel_coords': processed['voxel_coords'],
            'voxel_num_points': processed['voxel_num_points'],
            'batch_size': int(torch.sum(record_len).item()),
            'record_len': record_len,
        }

        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        dense_keys = find_dense_bev_keys(batch_dict)
        if self.debug:
            print('[DDSH][debug] dense BEV fields after backbone: %s' %
                  (dense_keys if dense_keys else 'none'))
        assert_no_dense_bev(batch_dict)
        batch_dict = self.sparse_height_compression(batch_dict)

        pairwise_t_matrix = self._pairwise_matrix(data_dict)
        # OpenCOOD builds pairwise_t_matrix[i, j] as the transform from CAV i
        # to CAV j. DDSH fusion therefore uses [helper_local_idx, 0] as the
        # helper -> ego matrix; no inverse is applied in this model.
        collect_debug = bool(data_dict.get('ddsh_collect_debug', False))
        fused_tokens, ddsh_stats = self.ddsh(
            batch_dict['ddsh_bev_tokens'], record_len, pairwise_t_matrix,
            collect_debug=collect_debug)
        self._debug_print(record_len, ddsh_stats, dense_keys)
        fused_sparse_tensor = tokens_to_sparse_tensor(fused_tokens)

        compute_loss = bool(self.training or
                            data_dict.get('ddsh_compute_loss', False))

        head_batch_dict = {
            'encoded_spconv_tensor': fused_sparse_tensor,
            'encoded_spconv_tensor_stride': self.feature_stride,
            'batch_size': int(len(record_len)),
            'record_len': record_len,
        }
        if compute_loss:
            head_batch_dict['gt_boxes'] = self._build_gt_boxes(data_dict)

        head_batch_dict = self.sparse_head(head_batch_dict)
        output_dict = {
            'ddsh_stats': ddsh_stats,
            'encoded_spconv_tensor_stride': self.feature_stride,
        }
        if collect_debug and ddsh_stats.get('ddsh_debug', None) is not None:
            output_dict['ddsh_debug'] = ddsh_stats['ddsh_debug']

        if 'final_box_dicts' in head_batch_dict:
            final_box_dicts = self.late_compensation(
                head_batch_dict['final_box_dicts'], data_dict,
                pairwise_t_matrix)
            output_dict['final_box_dicts'] = final_box_dicts
            if 'ddsh_debug' in output_dict:
                pred_boxes = self._collect_pred_boxes(final_box_dicts)
                output_dict['ddsh_debug']['pred_boxes'] = pred_boxes
                output_dict['ddsh_debug']['fused_pred_boxes'] = pred_boxes
                output_dict['ddsh_debug']['final_boxes'] = pred_boxes

        if compute_loss:
            loss, tb_dict = self.sparse_head.get_loss()
            output_dict['loss'] = loss
            output_dict['tb_dict'] = tb_dict
            if 'ddsh_debug' in output_dict:
                output_dict['ddsh_debug']['gt_boxes'] = head_batch_dict.get(
                    'gt_boxes', None)

        return output_dict
