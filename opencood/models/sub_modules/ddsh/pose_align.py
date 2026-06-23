import math

import torch
import torch.nn as nn

from opencood.models.sub_modules.ddsh.sparse_tensor_utils import make_tokens


class SparsePoseAligner(nn.Module):
    def __init__(self, voxel_size, lidar_range, feature_stride, model_cfg=None):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.feature_stride = int(feature_stride)
        self.register_buffer('voxel_size_tensor',
                             torch.tensor(voxel_size, dtype=torch.float32),
                             persistent=False)
        self.register_buffer('lidar_range_tensor',
                             torch.tensor(lidar_range, dtype=torch.float32),
                             persistent=False)

    @property
    def xy_resolution(self):
        return self.voxel_size_tensor[:2] * float(self.feature_stride)

    def coords_to_metric_xy(self, coords):
        coords = coords.float()
        res = self.xy_resolution.to(coords.device)
        origin = self.lidar_range_tensor[:2].to(coords.device)
        x = (coords[:, 2] + 0.5) * res[0] + origin[0]
        y = (coords[:, 1] + 0.5) * res[1] + origin[1]
        return torch.stack([x, y], dim=1)

    def metric_xy_to_coords(self, xy, batch_idx, spatial_shape):
        res = self.xy_resolution.to(xy.device)
        origin = self.lidar_range_tensor[:2].to(xy.device)
        x_idx = torch.floor((xy[:, 0] - origin[0]) / res[0]).long()
        y_idx = torch.floor((xy[:, 1] - origin[1]) / res[1]).long()

        h, w = int(spatial_shape[0]), int(spatial_shape[1])
        valid = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
        coords = torch.stack([
            torch.full_like(y_idx, int(batch_idx)),
            y_idx,
            x_idx,
        ], dim=1)
        return coords, valid

    def transform_xy(self, xy, matrix):
        if matrix.shape != (4, 4):
            raise ValueError('Expected a 4x4 transform matrix, got %s.' %
                             (tuple(matrix.shape),))
        z = torch.zeros((xy.shape[0], 1), device=xy.device, dtype=xy.dtype)
        ones = torch.ones((xy.shape[0], 1), device=xy.device, dtype=xy.dtype)
        points = torch.cat([xy, z, ones], dim=1)
        projected = torch.matmul(
            points,
            matrix.to(device=xy.device, dtype=xy.dtype).t())
        return projected[:, :2]

    def project_coords(self, coords, matrix, out_batch_idx, spatial_shape):
        if coords.shape[0] == 0:
            return coords.clone(), coords.new_zeros((0,), dtype=torch.bool)
        xy = self.coords_to_metric_xy(coords)
        xy = self.transform_xy(xy, matrix)
        return self.metric_xy_to_coords(xy, out_batch_idx, spatial_shape)

    def forward(self, tokens, matrix, out_batch_idx, batch_size=None):
        """Align helper sparse BEV tokens from helper frame into ego frame.

        ``tokens['coords']`` are expected to be in the helper local BEV grid.
        The returned ``coords`` are quantized in the ego BEV grid and filtered
        to tokens inside the ego point cloud range. No dense BEV tensor is
        created.
        """
        coords_ego, valid = self.project_coords(
            tokens['coords'], matrix, out_batch_idx, tokens['spatial_shape'])
        valid_idx = valid.nonzero(as_tuple=False).view(-1)
        out_batch_size = tokens['batch_size'] if batch_size is None else \
            int(batch_size)
        return make_tokens(
            tokens['features'][valid_idx],
            coords_ego[valid_idx],
            tokens['spatial_shape'],
            out_batch_size,
            original_coords=tokens['coords'][valid_idx],
            align_valid_index=valid_idx,
            aligned_count=int(valid_idx.numel()),
        )

    @staticmethod
    def yaw_from_matrix(matrix):
        return math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
