import math

import torch
import torch.nn as nn

from opencood.models.sub_modules.ddsh.sparse_tensor_utils import make_tokens


class SparseLocalAttention(nn.Module):
    def __init__(self, in_channels, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.enabled = bool(self.model_cfg.get('enabled', False))
        self.radius = int(self.model_cfg.get('radius', 2))
        self.max_neighbors = int(self.model_cfg.get('max_neighbors', 32))
        self.max_attention_tokens = int(
            self.model_cfg.get('max_attention_tokens', 4096))
        self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.out = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, tokens, force=False):
        if (not self.enabled and not force) or tokens['features'].shape[0] == 0:
            return tokens

        features = tokens['features']
        coords = tokens['coords']
        if features.shape[0] > self.max_attention_tokens:
            return tokens

        q = self.q(features)
        k = self.k(features)
        v = self.v(features)
        updated = torch.zeros_like(features)

        for batch_idx in range(int(tokens['batch_size'])):
            mask = coords[:, 0] == batch_idx
            idx = mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            cur_coords = coords[idx, 1:3]
            cur_q = q[idx]
            cur_k = k[idx]
            cur_v = v[idx]
            cur_out = []
            for token_i in range(idx.numel()):
                delta = (cur_coords - cur_coords[token_i]).abs()
                neigh_mask = (delta[:, 0] <= self.radius) & \
                    (delta[:, 1] <= self.radius)
                neigh_idx = neigh_mask.nonzero(as_tuple=False).view(-1)
                if neigh_idx.numel() > self.max_neighbors:
                    manhattan = delta[neigh_idx].sum(dim=1)
                    _, order = torch.topk(-manhattan.float(),
                                          k=self.max_neighbors)
                    neigh_idx = neigh_idx[order]
                logits = (cur_k[neigh_idx] * cur_q[token_i:token_i + 1]).sum(
                    dim=1) / math.sqrt(cur_q.shape[1])
                attn = torch.softmax(logits, dim=0).unsqueeze(1)
                cur_out.append((attn * cur_v[neigh_idx]).sum(dim=0))
            updated[idx] = torch.stack(cur_out, dim=0)

        features = self.norm(features + self.out(updated))
        return make_tokens(features, coords, tokens['spatial_shape'],
                           tokens['batch_size'])
