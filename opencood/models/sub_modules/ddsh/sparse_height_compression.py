import torch
import torch.nn as nn

from opencood.models.sub_modules.ddsh.sparse_tensor_utils import (
    make_tokens,
    tokens_from_sparse_tensor,
    unique_reduce,
)


class SparseHeightCompression(nn.Module):
    """
    Sparse-only height collapse.

    The integrated VoxelNeXt backbone in this repository already emits a 2D
    SparseConvTensor in encoded_spconv_tensor. This module accepts that tensor
    directly. If a future backbone exposes a 3D SparseConvTensor, it can also
    collapse z by sparse coordinate reduction without calling dense().
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.source_key = self.model_cfg.get('source_key',
                                             'encoded_spconv_tensor')
        self.reduce = self.model_cfg.get('reduce', 'sum')

    def forward(self, batch_dict):
        if self.source_key not in batch_dict:
            raise KeyError(
                'SparseHeightCompression expected "%s" in batch_dict. '
                'Available keys: %s' %
                (self.source_key, sorted(batch_dict.keys()))
            )

        sp_tensor = batch_dict[self.source_key]
        spatial_shape = list(sp_tensor.spatial_shape)
        indices = sp_tensor.indices.long()

        if len(spatial_shape) == 2:
            tokens = tokens_from_sparse_tensor(sp_tensor)
        elif len(spatial_shape) == 3 and indices.shape[1] == 4:
            coords_2d = indices[:, [0, 2, 3]]
            tokens = unique_reduce(sp_tensor.features, coords_2d,
                                   spatial_shape[1:], sp_tensor.batch_size,
                                   reduce=self.reduce)
        else:
            raise ValueError(
                'SparseHeightCompression supports 2D [B,Y,X] or 3D '
                '[B,Z,Y,X] SparseConvTensor only. Got spatial_shape=%s, '
                'indices shape=%s.' %
                (spatial_shape, tuple(indices.shape))
            )

        batch_dict['ddsh_bev_tokens'] = make_tokens(
            tokens['features'],
            tokens['coords'],
            tokens['spatial_shape'],
            tokens['batch_size'],
        )
        return batch_dict
