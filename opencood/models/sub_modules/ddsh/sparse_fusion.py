import torch
import torch.nn as nn

from opencood.models.sub_modules.ddsh.sparse_tensor_utils import (
    make_tokens,
    unique_reduce,
)


class SparseTokenFusion(nn.Module):
    def __init__(self, in_channels, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.method = self.model_cfg.get('method', 'gated_mean')
        if self.method == 'gated_mean':
            self.gate = nn.Sequential(
                nn.Linear(in_channels, max(16, in_channels // 2)),
                nn.ReLU(inplace=True),
                nn.Linear(max(16, in_channels // 2), 1),
                nn.Sigmoid(),
            )
        elif self.method in ['mean', 'sum', 'max']:
            self.gate = None
        else:
            raise ValueError('Unsupported DDSH sparse fusion method: %s' %
                             self.method)

    def forward(self, token_list, spatial_shape, batch_size):
        token_list = [tokens for tokens in token_list
                      if tokens is not None and tokens['features'].shape[0] > 0]
        if len(token_list) == 0:
            raise RuntimeError('DDSH fusion received no sparse tokens.')

        features = torch.cat([tokens['features'] for tokens in token_list],
                             dim=0)
        coords = torch.cat([tokens['coords'] for tokens in token_list], dim=0)

        if self.method == 'gated_mean':
            weights = self.gate(features)
            return unique_reduce(features, coords, spatial_shape, batch_size,
                                 reduce='mean', weights=weights)

        reduce = self.method
        return unique_reduce(features, coords, spatial_shape, batch_size,
                             reduce=reduce)
