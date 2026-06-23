import torch
import torch.nn as nn


class PoseSparseAligner(nn.Module):
    """
    Align sparse helper BEV tokens into ego coordinates using helper->ego pose.
    """
    def __init__(self, voxel_size, point_cloud_range, feature_stride=1):
        super(PoseSparseAligner, self).__init__()
        self.feature_stride = float(feature_stride)
        self.register_buffer('voxel_size',
                             torch.tensor(voxel_size, dtype=torch.float32),
                             persistent=False)
        self.register_buffer('point_cloud_range',
                             torch.tensor(point_cloud_range,
                                          dtype=torch.float32),
                             persistent=False)

    def _infer_spatial_shape(self):
        """Infer BEV [H, W] from lidar range and effective voxel size."""
        eff = self.voxel_size[:2] * self.feature_stride
        pc_range = self.point_cloud_range
        width = torch.round((pc_range[3] - pc_range[0]) / eff[0]).long()
        height = torch.round((pc_range[4] - pc_range[1]) / eff[1]).long()
        return int(height.item()), int(width.item())

    def _coords_to_metric(self, coords):
        """Convert sparse [batch,y,x] indices to metric xy center points."""
        eff = self.voxel_size[:2].to(coords.device) * self.feature_stride
        origin = self.point_cloud_range[:2].to(coords.device)
        x = (coords[:, 2].float() + 0.5) * eff[0] + origin[0]
        y = (coords[:, 1].float() + 0.5) * eff[1] + origin[1]
        return torch.stack([x, y], dim=1)

    def _metric_to_coords(self, xy, batch_ids, spatial_shape):
        """Quantize ego metric xy points back to sparse BEV coordinates."""
        eff = self.voxel_size[:2].to(xy.device) * self.feature_stride
        origin = self.point_cloud_range[:2].to(xy.device)
        x_idx = torch.floor((xy[:, 0] - origin[0]) / eff[0]).long()
        y_idx = torch.floor((xy[:, 1] - origin[1]) / eff[1]).long()
        h, w = int(spatial_shape[0]), int(spatial_shape[1])
        valid = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
        coords = torch.stack([batch_ids.long(), y_idx, x_idx], dim=1)
        return coords, valid

    def forward(self, supply_tokens, transformation_matrix,
                target_batch_idx=None, spatial_shape=None):
        """
        Apply helper->ego 4x4 transform and keep feats/score synchronized.
        """
        coords = supply_tokens['coords'].long()
        feats = supply_tokens.get('feats', supply_tokens.get('features'))
        if feats is None:
            raise KeyError('supply_tokens must contain "feats" or "features".')
        score = supply_tokens.get(
            'score', feats.new_ones((feats.shape[0],), dtype=feats.dtype))
        if transformation_matrix.shape != (4, 4):
            raise ValueError('transformation_matrix must be helper->ego [4,4].')

        if spatial_shape is None:
            spatial_shape = self._infer_spatial_shape()
        batch_ids = coords[:, 0].clone()
        if target_batch_idx is not None:
            batch_ids.fill_(int(target_batch_idx))

        xy = self._coords_to_metric(coords)
        zeros = xy.new_zeros((xy.shape[0], 1))
        ones = xy.new_ones((xy.shape[0], 1))
        points = torch.cat([xy, zeros, ones], dim=1)

        matrix = transformation_matrix.to(device=xy.device, dtype=xy.dtype)
        ego_points = torch.matmul(points, matrix.t())[:, :2]
        aligned_coords, valid = self._metric_to_coords(
            ego_points, batch_ids, spatial_shape)

        return {
            'coords': aligned_coords[valid],
            'feats': feats[valid],
            'score': score[valid],
        }
