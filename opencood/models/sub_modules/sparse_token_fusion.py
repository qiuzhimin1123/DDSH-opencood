import torch
import torch.nn as nn

from opencood.models.sub_modules.ddsh_sparse_utils import (
    build_coord_hash,
    intersect_hash,
)


class SparseTokenFusion(nn.Module):
    """
    Fuse ego and helper sparse BEV tokens by coordinate hash matching.
    """
    def __init__(self, in_channels, model_cfg=None):
        super(SparseTokenFusion, self).__init__()
        self.model_cfg = model_cfg or {}
        fusion_cfg = self.model_cfg.get('sparse_fusion', self.model_cfg)
        self.fusion_type = fusion_cfg.get('type', 'hash_mlp')
        self.allow_helper_only = bool(
            fusion_cfg.get('allow_helper_only', False))
        hidden_dim = int(fusion_cfg.get('hidden_dim', 128))
        if self.fusion_type not in ['hash_mlp', 'weighted_sum']:
            raise ValueError('Unsupported sparse fusion type: %s' %
                             self.fusion_type)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels),
        )

    @staticmethod
    def _parts(tokens):
        """Extract coords, feats, and score with compatible field names."""
        feats = tokens.get('feats', tokens.get('features'))
        if feats is None:
            raise KeyError('tokens must contain "feats" or "features".')
        score = tokens.get('score', feats.new_ones((feats.shape[0],)))
        return tokens['coords'].long(), feats, score.reshape(-1)

    @staticmethod
    def _reduce_helper(coords, feats, score, spatial_shape):
        """Merge duplicate helper coordinates by score-weighted averaging."""
        if coords.shape[0] == 0:
            return coords, feats, score
        coord_hash = build_coord_hash(coords, spatial_shape)
        unique_hash, inverse = torch.unique(coord_hash,
                                            sorted=True,
                                            return_inverse=True)
        unique_coords = coords.new_zeros((unique_hash.shape[0],
                                          coords.shape[1]))
        unique_score = score.new_zeros((unique_hash.shape[0],))
        unique_feats = feats.new_zeros((unique_hash.shape[0],
                                        feats.shape[1]))
        weights = score.clamp_min(1e-4)
        unique_score.index_add_(0, inverse, score)
        denom = feats.new_zeros((unique_hash.shape[0], 1))
        denom.index_add_(0, inverse, weights.reshape(-1, 1))
        unique_feats.index_add_(0, inverse, feats * weights.reshape(-1, 1))
        unique_feats = unique_feats / denom.clamp_min(1e-6)
        for token_idx in range(coords.shape[0]):
            unique_coords[inverse[token_idx]] = coords[token_idx]
        counts = score.new_zeros((unique_hash.shape[0],))
        counts.index_add_(0, inverse, torch.ones_like(score))
        unique_score = unique_score / counts.clamp_min(1.0)
        return unique_coords, unique_feats, unique_score

    def _fuse_hash_mlp(self, ego_feats, helper_feats, helper_score):
        """Run MLP over [ego_feat, helper_feat, helper_score]."""
        fusion_input = torch.cat(
            [ego_feats, helper_feats, helper_score.reshape(-1, 1)], dim=1)
        return self.mlp(fusion_input)

    def forward(self, ego_tokens, helper_tokens):
        """
        Fuse helper tokens into ego tokens without creating dense BEV maps.
        """
        ego_coords, ego_feats, ego_score = self._parts(ego_tokens)
        helper_coords, helper_feats, helper_score = self._parts(helper_tokens)
        spatial_shape = ego_tokens.get('spatial_shape',
                                       helper_tokens.get('spatial_shape'))
        if spatial_shape is None:
            raise KeyError('sparse fusion requires spatial_shape metadata.')

        helper_coords, helper_feats, helper_score = self._reduce_helper(
            helper_coords, helper_feats, helper_score, spatial_shape)

        ego_hash = build_coord_hash(ego_coords, spatial_shape)
        helper_hash = build_coord_hash(helper_coords, spatial_shape)
        idx_ego, idx_helper = intersect_hash(ego_hash, helper_hash)

        fused_feats = ego_feats.clone()
        fused_score = ego_score.clone()
        if idx_ego.numel() > 0:
            if self.fusion_type == 'hash_mlp':
                fused_feats[idx_ego] = self._fuse_hash_mlp(
                    ego_feats[idx_ego], helper_feats[idx_helper],
                    helper_score[idx_helper])
            else:
                weight = helper_score[idx_helper].reshape(-1, 1).clamp_min(0)
                fused_feats[idx_ego] = (
                    ego_feats[idx_ego] + weight * helper_feats[idx_helper]) / \
                    (1.0 + weight)
            fused_score[idx_ego] = torch.maximum(
                fused_score[idx_ego], helper_score[idx_helper])

        if self.allow_helper_only and helper_coords.shape[0] > 0:
            matched_helper = torch.zeros(helper_coords.shape[0],
                                         device=helper_coords.device,
                                         dtype=torch.bool)
            matched_helper[idx_helper] = True
            helper_only = ~matched_helper
            if helper_only.any():
                fused_coords = torch.cat([ego_coords,
                                          helper_coords[helper_only]], dim=0)
                fused_feats = torch.cat([fused_feats,
                                         helper_feats[helper_only]], dim=0)
                fused_score = torch.cat([fused_score,
                                         helper_score[helper_only]], dim=0)
            else:
                fused_coords = ego_coords
        else:
            fused_coords = ego_coords

        return {
            'coords': fused_coords,
            'feats': fused_feats,
            'score': fused_score,
            'batch_size': ego_tokens.get('batch_size'),
            'spatial_shape': spatial_shape,
            'matched_token_count': int(idx_ego.numel()),
        }
