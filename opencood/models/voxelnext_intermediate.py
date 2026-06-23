# -*- coding: utf-8 -*-
# Author: OpenCOOD contributors, adapted for VoxelNeXt backbone
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn

from opencood.models.fuse_modules.self_attn import AttFusion
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.voxelnext_backbone import \
    VoxelResBackBone8xVoxelNeXt


class VoxelNextIntermediate(nn.Module):
    def __init__(self, args):
        super(VoxelNextIntermediate, self).__init__()

        self.num_point_features = args.get('num_point_features', 4)
        self.mean_vfe = MeanVFE(args.get('mean_vfe', {}),
                                self.num_point_features)

        self.backbone_3d = VoxelResBackBone8xVoxelNeXt(
            args['backbone_3d'],
            self.num_point_features,
            args['grid_size'])

        out_channel = args['backbone_3d'].get('OUT_CHANNEL', 128)
        self.fusion_net = AttFusion(out_channel)

        self.cls_head = nn.Conv2d(out_channel,
                                  args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(out_channel,
                                  7 * args['anchor_num'],
                                  kernel_size=1)

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'batch_size': int(torch.sum(record_len).item()),
            'record_len': record_len,
        }

        batch_dict = self.mean_vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)

        spatial_features_2d = \
            batch_dict['encoded_spconv_tensor'].dense()
        spatial_features_2d = self.fusion_net(spatial_features_2d,
                                              record_len)

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        return {
            'psm': psm,
            'rm': rm,
        }
