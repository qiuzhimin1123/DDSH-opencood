import torch
import torch.nn as nn


class SparseHeightCompression(nn.Module):
    """
    Collapse sparse 3D voxel tokens into sparse BEV tokens without dense().
    """
    def __init__(self, model_cfg=None):
        super(SparseHeightCompression, self).__init__()
        model_cfg = model_cfg or {}
        sparse_bev_cfg = model_cfg.get('sparse_bev', model_cfg)
        self.reduce = sparse_bev_cfg.get('reduce', 'max')
        if self.reduce not in ['max', 'mean', 'sum']:
            raise ValueError('Unsupported sparse BEV reduce mode: %s' %
                             self.reduce)

    @staticmethod
    def _unpack_sparse_input(sparse_input):
        """Read features/indices/spatial_shape/batch_size from dict or spconv."""
        if isinstance(sparse_input, dict):
            features = sparse_input.get('features', sparse_input.get('feats'))
            indices = sparse_input.get('indices', sparse_input.get('coords'))
            spatial_shape = sparse_input['spatial_shape']
            batch_size = sparse_input['batch_size']
        else:
            features = sparse_input.features
            indices = sparse_input.indices
            spatial_shape = sparse_input.spatial_shape
            batch_size = sparse_input.batch_size
        if features is None or indices is None:
            raise KeyError('SparseHeightCompression needs features and indices.')
        return features, indices.long(), list(spatial_shape), int(batch_size)

    @staticmethod
    def _spatial_shape_2d(spatial_shape, indices):
        """Infer BEV [H, W] from a 3D or already-compressed sparse shape."""
        if indices.shape[1] == 4:
            if len(spatial_shape) != 3:
                raise ValueError('3D indices [B,Z,Y,X] require spatial_shape '
                                 '[D,H,W].')
            return [int(spatial_shape[1]), int(spatial_shape[2])]
        if indices.shape[1] == 3:
            if len(spatial_shape) != 2:
                raise ValueError('2D indices [B,Y,X] require spatial_shape '
                                 '[H,W].')
            return [int(spatial_shape[0]), int(spatial_shape[1])]
        raise ValueError('indices must be [N,4] or [N,3], got %s.' %
                         (tuple(indices.shape),))

    @staticmethod
    def _reduce_sum_mean(features, inverse, out_size, reduce):
        """Segment-reduce sparse features by sum or mean on CPU/CUDA tensors."""
        out = features.new_zeros((out_size, features.shape[1]))
        out.index_add_(0, inverse, features)
        if reduce == 'mean':
            counts = features.new_zeros((out_size, 1))
            ones = features.new_ones((features.shape[0], 1))
            counts.index_add_(0, inverse, ones)
            out = out / counts.clamp_min(1.0)
        return out

    @staticmethod
    def _reduce_max(features, inverse, out_size):
        """Segment max for sparse features without constructing a dense grid."""
        out = features.new_full((out_size, features.shape[1]), -float('inf'))
        if hasattr(out, 'scatter_reduce_'):
            expand_inverse = inverse.view(-1, 1).expand(-1, features.shape[1])
            out.scatter_reduce_(0, expand_inverse, features, reduce='amax',
                                include_self=True)
        else:
            for token_idx in range(features.shape[0]):
                out[inverse[token_idx]] = torch.maximum(
                    out[inverse[token_idx]], features[token_idx])
        out[out == -float('inf')] = 0
        return out

    def forward(self, sparse_input):
        """
        Convert [batch,z,y,x] sparse voxels to [batch,y,x] sparse BEV tokens.
        """
        features, indices, spatial_shape, batch_size = \
            self._unpack_sparse_input(sparse_input)
        spatial_shape_2d = self._spatial_shape_2d(spatial_shape, indices)

        if indices.shape[1] == 4:
            coords_2d = indices[:, [0, 2, 3]]
        else:
            coords_2d = indices[:, [0, 1, 2]]

        unique_coords, inverse = torch.unique(coords_2d.long(),
                                              dim=0,
                                              sorted=True,
                                              return_inverse=True)
        if self.reduce in ['sum', 'mean']:
            feats_2d = self._reduce_sum_mean(features, inverse,
                                             unique_coords.shape[0],
                                             self.reduce)
        else:
            feats_2d = self._reduce_max(features, inverse,
                                        unique_coords.shape[0])

        return {
            'coords': unique_coords.long(),
            'feats': feats_2d,
            'batch_size': batch_size,
            'spatial_shape': spatial_shape_2d,
        }
