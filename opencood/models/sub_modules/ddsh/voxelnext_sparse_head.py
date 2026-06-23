import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

from opencood.models.sub_modules.spconv_utils import spconv
from opencood.utils import box_utils


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _gaussian_radius(height, width, min_overlap=0.5):
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = torch.sqrt(torch.clamp(b1 ** 2 - 4 * a1 * c1, min=0))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = torch.sqrt(torch.clamp(b2 ** 2 - 4 * a2 * c2, min=0))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = torch.sqrt(torch.clamp(b3 ** 2 - 4 * a3 * c3, min=0))
    r3 = (b3 + sq3) / 2

    return torch.min(torch.min(r1, r2), r3)


def _draw_gaussian_to_sparse_heatmap(heatmap, distances, radius):
    diameter = 2 * radius + 1
    sigma = max(float(diameter) / 6.0, 1e-3)
    gaussian = torch.exp(-distances / (2 * sigma * sigma))
    torch.max(heatmap, gaussian, out=heatmap)


def _neg_loss_sparse(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(pred.clamp_min(1e-6)) * torch.pow(1 - pred, 2) * \
        pos_inds
    neg_loss = torch.log((1 - pred).clamp_min(1e-6)) * torch.pow(pred, 2) * \
        neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def _reg_loss_sparse(output, mask, ind, target, batch_index):
    pred = []
    batch_size = mask.shape[0]
    for bs_idx in range(batch_size):
        batch_inds = batch_index == bs_idx
        cur_output = output[batch_inds]
        if cur_output.shape[0] == 0:
            pred.append(target.new_zeros(target[bs_idx].shape))
            continue
        safe_ind = ind[bs_idx].clamp(max=max(cur_output.shape[0] - 1, 0))
        pred.append(cur_output[safe_ind])
    pred = torch.stack(pred, dim=0)

    valid_mask = mask.unsqueeze(2).expand_as(target).float()
    valid_mask = valid_mask * (~torch.isnan(target)).float()
    diff = torch.abs((pred - target) * valid_mask)
    loss = diff.transpose(2, 0).sum(dim=2).sum(dim=1)
    normalizer = torch.clamp_min(mask.float().sum(), min=1.0)
    return loss / normalizer


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, kernel_size,
                 init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for _ in range(num_conv - 1):
                fc_list.append(spconv.SparseSequential(
                    spconv.SubMConv2d(input_channels, input_channels,
                                      kernel_size,
                                      padding=int(kernel_size // 2),
                                      bias=use_bias,
                                      indice_key='ddsh_%s' % cur_name),
                    nn.BatchNorm1d(input_channels),
                    nn.ReLU(),
                ))
            fc_list.append(spconv.SubMConv2d(
                input_channels, output_channels, 1, bias=True,
                indice_key='ddsh_%s_out' % cur_name))
            fc = nn.Sequential(*fc_list)

            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for module in fc.modules():
                    if isinstance(module, spconv.SubMConv2d):
                        kaiming_normal_(module.weight.data)
                        if hasattr(module, 'bias') and module.bias is not None:
                            nn.init.constant_(module.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x).features
        return ret_dict


class DdshVoxelNeXtSparseHead(nn.Module):
    """
    Sparse VoxelNeXt head adapted for OpenCOOD's batch format.
    """
    def __init__(self, model_cfg, input_channels, num_class, class_names,
                 grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=False):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.num_class = int(num_class)
        self.grid_size = grid_size
        self.class_names = list(class_names)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}

        self.register_buffer('point_cloud_range',
                             torch.tensor(point_cloud_range,
                                          dtype=torch.float32),
                             persistent=False)
        self.register_buffer('voxel_size',
                             torch.tensor(voxel_size, dtype=torch.float32),
                             persistent=False)

        target_cfg = _cfg_get(self.model_cfg, 'TARGET_ASSIGNER_CONFIG', {})
        self.feature_map_stride = int(_cfg_get(target_cfg,
                                               'FEATURE_MAP_STRIDE', 8))
        self.num_max_objs = int(_cfg_get(target_cfg, 'NUM_MAX_OBJS', 500))
        self.gaussian_overlap = float(_cfg_get(target_cfg,
                                               'GAUSSIAN_OVERLAP', 0.1))
        self.min_radius = int(_cfg_get(target_cfg, 'MIN_RADIUS', 2))
        self.gaussian_ratio = float(_cfg_get(self.model_cfg,
                                             'GAUSSIAN_RATIO', 1.0))
        self.gaussian_type = _cfg_get(self.model_cfg, 'GAUSSIAN_TYPE',
                                      ['nearst', 'gt_center'])

        class_names_each_head = _cfg_get(
            self.model_cfg, 'CLASS_NAMES_EACH_HEAD', [self.class_names])
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        for head_idx, cur_class_names in enumerate(class_names_each_head):
            valid_names = [name for name in cur_class_names
                           if name in self.class_names]
            self.class_names_each_head.append(valid_names)
            mapping = torch.from_numpy(np.array([
                self.class_names.index(name) for name in valid_names
            ], dtype=np.int64))
            self.register_buffer('class_id_mapping_head_%d' % head_idx,
                                 mapping, persistent=False)
            self.class_id_mapping_each_head.append(mapping)

        total_classes = sum(len(names) for names in self.class_names_each_head)
        if total_classes != len(self.class_names):
            raise ValueError(
                'Sparse head class split does not cover class_names. '
                'class_names=%s, CLASS_NAMES_EACH_HEAD=%s' %
                (self.class_names, self.class_names_each_head))

        self.separate_head_cfg = _cfg_get(self.model_cfg,
                                          'SEPARATE_HEAD_CFG', {})
        self.head_order = _cfg_get(self.separate_head_cfg, 'HEAD_ORDER',
                                   ['center', 'center_z', 'dim', 'rot'])
        default_head_dict = {
            'center': {'out_channels': 2, 'num_conv': 2},
            'center_z': {'out_channels': 1, 'num_conv': 2},
            'dim': {'out_channels': 3, 'num_conv': 2},
            'rot': {'out_channels': 2, 'num_conv': 2},
        }
        self.head_dict = _cfg_get(self.separate_head_cfg, 'HEAD_DICT',
                                  default_head_dict)

        kernel_size_head = int(_cfg_get(self.model_cfg, 'KERNEL_SIZE_HEAD', 3))
        shared_channels = int(_cfg_get(self.model_cfg,
                                       'SHARED_CONV_CHANNEL',
                                       input_channels))
        if shared_channels != input_channels:
            raise ValueError(
                'DDSH sparse head expects SHARED_CONV_CHANNEL to match input '
                'channels. Got %d vs %d.' % (shared_channels, input_channels))

        self.heads_list = nn.ModuleList()
        for cur_class_names in self.class_names_each_head:
            cur_head_dict = copy.deepcopy(self.head_dict)
            cur_head_dict['hm'] = {
                'out_channels': len(cur_class_names),
                'num_conv': int(_cfg_get(self.model_cfg, 'NUM_HM_CONV', 2)),
            }
            self.heads_list.append(SeparateHead(
                input_channels=shared_channels,
                sep_head_dict=cur_head_dict,
                kernel_size=kernel_size_head,
                init_bias=-2.19,
                use_bias=bool(_cfg_get(self.model_cfg,
                                       'USE_BIAS_BEFORE_NORM', False)),
            ))

    def assign_targets(self, gt_boxes, num_voxels, spatial_indices,
                       spatial_shape):
        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'gt_boxes': [],
        }
        all_names = np.array(['bg'] + self.class_names)

        for head_idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list = []
            target_boxes_list = []
            inds_list = []
            masks_list = []
            gt_boxes_list = []

            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                if cur_gt_boxes.shape[0] > 0:
                    max_cls_id = int(cur_gt_boxes[:, -1].max().item())
                    if max_cls_id >= len(all_names):
                        raise ValueError(
                            'gt_boxes class id %d exceeds configured classes '
                            '%s.' % (max_cls_id, self.class_names))
                    gt_class_names = all_names[
                        cur_gt_boxes[:, -1].detach().cpu().long().numpy()]
                else:
                    gt_class_names = []

                gt_boxes_single_head = []
                for obj_idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[obj_idx].clone()
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head,
                                                     dim=0)

                heatmap, ret_boxes, inds, mask = \
                    self.assign_target_of_single_head(
                        num_classes=len(cur_class_names),
                        gt_boxes=gt_boxes_single_head,
                        num_voxels=int(num_voxels[bs_idx]),
                        spatial_indices=spatial_indices[bs_idx],
                        spatial_shape=spatial_shape,
                    )
                heatmap_list.append(heatmap)
                target_boxes_list.append(ret_boxes)
                inds_list.append(inds)
                masks_list.append(mask)
                gt_boxes_list.append(gt_boxes_single_head[:, :-1])

            ret_dict['heatmaps'].append(
                torch.cat(heatmap_list, dim=1).permute(1, 0)
                if len(heatmap_list) > 0 else gt_boxes.new_zeros((0, 0)))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list,
                                                        dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
            ret_dict['gt_boxes'].append(gt_boxes_list)

        return ret_dict

    @staticmethod
    def distance(voxel_indices, center):
        return ((voxel_indices - center.unsqueeze(0)) ** 2).sum(-1)

    def assign_target_of_single_head(self, num_classes, gt_boxes, num_voxels,
                                     spatial_indices, spatial_shape):
        heatmap = gt_boxes.new_zeros(num_classes, num_voxels)
        ret_boxes = gt_boxes.new_zeros((self.num_max_objs, 8))
        inds = gt_boxes.new_zeros(self.num_max_objs).long()
        mask = gt_boxes.new_zeros(self.num_max_objs).long()

        if num_voxels == 0 or gt_boxes.shape[0] == 0:
            return heatmap, ret_boxes, inds, mask

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / \
            self.feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / \
            self.feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=spatial_shape[1] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=spatial_shape[0] - 0.5)

        center = torch.stack((coord_x, coord_y), dim=-1)
        center_int = center.int()
        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx_feat = dx / self.voxel_size[0] / self.feature_map_stride
        dy_feat = dy / self.voxel_size[1] / self.feature_map_stride
        radius = _gaussian_radius(dx_feat, dy_feat, self.gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=self.min_radius)

        max_objs = min(self.num_max_objs, gt_boxes.shape[0])
        for obj_idx in range(max_objs):
            if dx[obj_idx] <= 0 or dy[obj_idx] <= 0 or dz[obj_idx] <= 0:
                continue
            if not (0 <= center_int[obj_idx][0] < spatial_shape[1] and
                    0 <= center_int[obj_idx][1] < spatial_shape[0]):
                continue

            cur_class_id = (gt_boxes[obj_idx, -1] - 1).long()
            distances = self.distance(spatial_indices, center[obj_idx])
            inds[obj_idx] = distances.argmin()
            mask[obj_idx] = 1

            cur_radius = int(radius[obj_idx].item() * self.gaussian_ratio)
            if 'gt_center' in self.gaussian_type:
                _draw_gaussian_to_sparse_heatmap(
                    heatmap[cur_class_id], distances, cur_radius)
            if 'nearst' in self.gaussian_type:
                nearest_distance = self.distance(
                    spatial_indices, spatial_indices[inds[obj_idx]].float())
                _draw_gaussian_to_sparse_heatmap(
                    heatmap[cur_class_id], nearest_distance, cur_radius)

            ret_boxes[obj_idx, 0:2] = center[obj_idx] - \
                spatial_indices[inds[obj_idx]][:2]
            ret_boxes[obj_idx, 2] = z[obj_idx]
            ret_boxes[obj_idx, 3:6] = gt_boxes[obj_idx, 3:6].log()
            ret_boxes[obj_idx, 6] = torch.cos(gt_boxes[obj_idx, 6])
            ret_boxes[obj_idx, 7] = torch.sin(gt_boxes[obj_idx, 6])

        return heatmap, ret_boxes, inds, mask

    @staticmethod
    def sigmoid(x):
        return torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        batch_index = self.forward_ret_dict['batch_index']

        loss_weights = _cfg_get(_cfg_get(self.model_cfg, 'LOSS_CONFIG', {}),
                                'LOSS_WEIGHTS', {})
        cls_weight = float(loss_weights.get('cls_weight', 1.0))
        loc_weight = float(loss_weights.get('loc_weight', 1.0))
        code_weights = pred_dicts[0]['center'].new_tensor(
            loss_weights.get('code_weights',
                             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        tb_dict = {}
        total_loss = pred_dicts[0]['center'].new_tensor(0.0)
        for head_idx, pred_dict in enumerate(pred_dicts):
            pred_hm = self.sigmoid(pred_dict['hm'])
            hm_loss = _neg_loss_sparse(
                pred_hm, target_dicts['heatmaps'][head_idx])
            hm_loss = hm_loss * cls_weight

            pred_boxes = torch.cat(
                [pred_dict[head_name] for head_name in self.head_order],
                dim=1)
            reg_loss = _reg_loss_sparse(
                pred_boxes, target_dicts['masks'][head_idx],
                target_dicts['inds'][head_idx],
                target_dicts['target_boxes'][head_idx],
                batch_index,
            )
            loc_loss = (reg_loss * code_weights).sum() * loc_weight
            total_loss = total_loss + hm_loss + loc_loss

            tb_dict['hm_loss_head_%d' % head_idx] = float(
                hm_loss.detach().cpu())
            tb_dict['loc_loss_head_%d' % head_idx] = float(
                loc_loss.detach().cpu())

        tb_dict['rpn_loss'] = float(total_loss.detach().cpu())
        return total_loss, tb_dict

    def _get_voxel_infos(self, x):
        spatial_shape = list(x.spatial_shape)
        voxel_indices = x.indices.long()
        batch_size = int(x.batch_size)
        batch_index = voxel_indices[:, 0]
        spatial_indices = []
        num_voxels = []

        for bs_idx in range(batch_size):
            batch_inds = batch_index == bs_idx
            cur_indices = voxel_indices[batch_inds]
            spatial_indices.append(cur_indices[:, [2, 1]].float())
            num_voxels.append(int(batch_inds.sum().item()))

        return spatial_shape, batch_index, voxel_indices, spatial_indices, \
            num_voxels

    def _decode_single_head(self, pred_dict, voxel_indices, head_idx,
                            batch_size):
        post_cfg = _cfg_get(self.model_cfg, 'POST_PROCESSING', {})
        score_thresh = float(_cfg_get(post_cfg, 'SCORE_THRESH', 0.1))
        max_obj = int(_cfg_get(post_cfg, 'MAX_OBJ_PER_SAMPLE', 500))
        limit_range = torch.tensor(
            _cfg_get(post_cfg, 'POST_CENTER_LIMIT_RANGE',
                     self.point_cloud_range.detach().cpu().tolist()),
            device=voxel_indices.device,
            dtype=torch.float32)

        ret = []
        mapping = self.class_id_mapping_each_head[head_idx].to(
            voxel_indices.device)
        hm = pred_dict['hm'].sigmoid()
        dim = torch.exp(torch.clamp(pred_dict['dim'], min=-5, max=5))
        angle = torch.atan2(pred_dict['rot'][:, 1:2],
                            pred_dict['rot'][:, 0:1])

        for bs_idx in range(batch_size):
            batch_mask = voxel_indices[:, 0] == bs_idx
            if batch_mask.sum().item() == 0:
                ret.append({
                    'pred_boxes': voxel_indices.new_zeros((0, 7)).float(),
                    'pred_scores': voxel_indices.new_zeros((0,)).float(),
                    'pred_labels': voxel_indices.new_zeros((0,)).long(),
                })
                continue

            cur_hm = hm[batch_mask]
            flat_scores = cur_hm.reshape(-1)
            topk = min(max_obj, flat_scores.numel())
            scores, flat_idx = torch.topk(flat_scores, k=topk)
            class_count = cur_hm.shape[1]
            token_idx = torch.div(flat_idx, class_count,
                                  rounding_mode='floor')
            class_idx = flat_idx % class_count

            cur_voxel_indices = voxel_indices[batch_mask][token_idx]
            cur_center = pred_dict['center'][batch_mask][token_idx]
            cur_center_z = pred_dict['center_z'][batch_mask][token_idx]
            cur_dim = dim[batch_mask][token_idx]
            cur_angle = angle[batch_mask][token_idx]

            xs = (cur_voxel_indices[:, 2:3].float() + cur_center[:, 0:1]) * \
                self.feature_map_stride * self.voxel_size[0] + \
                self.point_cloud_range[0]
            ys = (cur_voxel_indices[:, 1:2].float() + cur_center[:, 1:2]) * \
                self.feature_map_stride * self.voxel_size[1] + \
                self.point_cloud_range[1]
            boxes = torch.cat([xs, ys, cur_center_z, cur_dim, cur_angle],
                              dim=1)

            keep = (scores >= score_thresh)
            keep &= (boxes[:, :3] >= limit_range[:3]).all(dim=1)
            keep &= (boxes[:, :3] <= limit_range[3:]).all(dim=1)
            boxes = boxes[keep]
            scores_keep = scores[keep]
            labels = mapping[class_idx[keep]].long() + 1

            ret.append({
                'pred_boxes': boxes,
                'pred_scores': scores_keep,
                'pred_labels': labels,
            })

        return ret

    def _nms_prediction(self, pred_dict):
        post_cfg = _cfg_get(self.model_cfg, 'POST_PROCESSING', {})
        nms_cfg = _cfg_get(post_cfg, 'NMS_CONFIG', {})
        nms_thresh = float(_cfg_get(nms_cfg, 'NMS_THRESH', 0.1))
        post_max_size = int(_cfg_get(nms_cfg, 'NMS_POST_MAXSIZE', 500))

        boxes = pred_dict['pred_boxes']
        scores = pred_dict['pred_scores']
        labels = pred_dict['pred_labels']
        if boxes.shape[0] == 0:
            return pred_dict

        corners = box_utils.boxes_to_corners_3d(boxes, order='lwh')
        keep = box_utils.nms_rotated(corners, scores, nms_thresh)
        keep = torch.as_tensor(keep, device=boxes.device,
                               dtype=torch.long)[:post_max_size]
        return {
            'pred_boxes': boxes[keep],
            'pred_scores': scores[keep],
            'pred_labels': labels[keep],
        }

    def generate_predicted_boxes(self, batch_size, pred_dicts, voxel_indices):
        per_head = [
            self._decode_single_head(pred_dict, voxel_indices, head_idx,
                                     batch_size)
            for head_idx, pred_dict in enumerate(pred_dicts)
        ]

        ret = []
        for bs_idx in range(batch_size):
            boxes = torch.cat([head[bs_idx]['pred_boxes']
                               for head in per_head], dim=0)
            scores = torch.cat([head[bs_idx]['pred_scores']
                                for head in per_head], dim=0)
            labels = torch.cat([head[bs_idx]['pred_labels']
                                for head in per_head], dim=0)
            ret.append(self._nms_prediction({
                'pred_boxes': boxes,
                'pred_scores': scores,
                'pred_labels': labels,
            }))
        return ret

    def forward(self, data_dict):
        if 'encoded_spconv_tensor' not in data_dict:
            raise KeyError(
                'DdshVoxelNeXtSparseHead requires encoded_spconv_tensor.')
        x = data_dict['encoded_spconv_tensor']
        spatial_shape, batch_index, voxel_indices, spatial_indices, \
            num_voxels = self._get_voxel_infos(x)

        pred_dicts = [head(x) for head in self.heads_list]
        self.forward_ret_dict = {
            'batch_index': batch_index,
            'pred_dicts': pred_dicts,
            'voxel_indices': voxel_indices,
        }

        if 'gt_boxes' in data_dict:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], num_voxels, spatial_indices,
                spatial_shape)
            self.forward_ret_dict['target_dicts'] = target_dict

        if not self.training or self.predict_boxes_when_training:
            data_dict['final_box_dicts'] = self.generate_predicted_boxes(
                int(data_dict['batch_size']), pred_dicts, voxel_indices)

        return data_dict
