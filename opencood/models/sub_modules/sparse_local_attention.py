import math

import torch
import torch.nn as nn


class SparseLocalAttentionFusion(nn.Module):
    """
    Pose-aware local sparse cross-attention from ego queries to helper tokens.

    TODO: replace the per-token neighbor loop with a fused sparse neighborhood
    kernel when one is available. The current implementation keeps the public
    interface real and sparse, while favoring clarity over speed.
    """
    def __init__(self, in_channels, model_cfg=None):
        super(SparseLocalAttentionFusion, self).__init__()
        self.model_cfg = model_cfg or {}
        attn_cfg = self.model_cfg.get('sparse_attention', self.model_cfg)
        self.enable = bool(attn_cfg.get('enable',
                                        attn_cfg.get('enabled', True)))
        self.radius = int(attn_cfg.get('radius', 3))
        self.num_heads = int(attn_cfg.get('num_heads', 4))
        self.hidden_dim = int(attn_cfg.get('hidden_dim', 128))
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads.')
        self.head_dim = self.hidden_dim // self.num_heads
        self.q_proj = nn.Linear(in_channels, self.hidden_dim)
        self.k_proj = nn.Linear(in_channels, self.hidden_dim)
        self.v_proj = nn.Linear(in_channels, self.hidden_dim)
        self.rel_bias = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.num_heads),
        )
        self.out_proj = nn.Linear(self.hidden_dim, in_channels)
        self.norm = nn.LayerNorm(in_channels)

    @staticmethod
    def _parts(tokens):
        """Extract sparse token components with compatible feat names."""
        feats = tokens.get('feats', tokens.get('features'))
        if feats is None:
            raise KeyError('tokens must contain "feats" or "features".')
        return tokens['coords'].long(), feats

    def _attend_one_scene(self, ego_coords, ego_feats, helper_coords,
                          helper_feats):
        """Fuse one scene worth of ego/helper tokens using local attention."""
        if helper_coords.shape[0] == 0 or ego_coords.shape[0] == 0:
            return ego_feats

        q = self.q_proj(ego_feats).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(helper_feats).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(helper_feats).view(-1, self.num_heads, self.head_dim)
        output = ego_feats.clone()

        helper_yx = helper_coords[:, 1:3].float()
        ego_yx = ego_coords[:, 1:3].float()
        for ego_idx in range(ego_coords.shape[0]):
            rel = helper_yx - ego_yx[ego_idx:ego_idx + 1]
            mask = rel.abs().max(dim=1).values <= self.radius
            if not mask.any():
                continue
            rel_neigh = rel[mask]
            k_neigh = k[mask]
            v_neigh = v[mask]
            logits = (q[ego_idx:ego_idx + 1] * k_neigh).sum(dim=2) / \
                math.sqrt(float(self.head_dim))
            logits = logits.transpose(0, 1)
            logits = logits + self.rel_bias(rel_neigh).transpose(0, 1)
            attn = torch.softmax(logits, dim=1)
            context = torch.einsum('hn,nhd->hd', attn, v_neigh)
            update = self.out_proj(context.reshape(1, -1))
            output[ego_idx:ego_idx + 1] = self.norm(
                ego_feats[ego_idx:ego_idx + 1] + update)
        return output

    def forward(self, ego_tokens, helper_tokens):
        """
        Return ego-coordinate fused tokens; helper-only tokens are not added.
        """
        if not self.enable:
            return ego_tokens

        ego_coords, ego_feats = self._parts(ego_tokens)
        helper_coords, helper_feats = self._parts(helper_tokens)
        fused_feats = ego_feats.clone()

        batch_ids = torch.unique(ego_coords[:, 0]) if ego_coords.numel() > 0 \
            else ego_coords.new_zeros((0,))
        for batch_idx in batch_ids.tolist():
            ego_mask = ego_coords[:, 0] == int(batch_idx)
            helper_mask = helper_coords[:, 0] == int(batch_idx)
            fused_feats[ego_mask] = self._attend_one_scene(
                ego_coords[ego_mask], ego_feats[ego_mask],
                helper_coords[helper_mask], helper_feats[helper_mask])

        return {
            'coords': ego_coords,
            'feats': fused_feats,
            'score': ego_tokens.get('score'),
            'batch_size': ego_tokens.get('batch_size'),
            'spatial_shape': ego_tokens.get('spatial_shape'),
        }
