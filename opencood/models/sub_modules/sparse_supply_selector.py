import torch
import torch.nn as nn

from opencood.models.sub_modules.ddsh_sparse_utils import safe_topk


class SparseSupplySelector(nn.Module):
    """
    Select helper sparse tokens for communication.
    """
    def __init__(self, in_channels, model_cfg=None):
        super(SparseSupplySelector, self).__init__()
        self.model_cfg = model_cfg or {}
        comm_cfg = self.model_cfg.get('communication', self.model_cfg)
        self.supply_mode = comm_cfg.get('supply_mode', 'all')
        self.supply_topk = comm_cfg.get('supply_topk', 1024)
        self.demand_radius = float(comm_cfg.get('demand_radius', 4))
        self.demand_match_weight = float(
            comm_cfg.get('demand_match_weight', 1.0))
        if self.supply_mode not in ['all', 'topk', 'demand_supply']:
            raise ValueError('Unsupported supply_mode: %s' % self.supply_mode)
        self.quality_head = nn.Linear(in_channels, 1)

    @staticmethod
    def _token_parts(tokens):
        """Extract coords and feats from a sparse token dictionary."""
        feats = tokens.get('feats', tokens.get('features'))
        if feats is None:
            raise KeyError('tokens must contain "feats" or "features".')
        return tokens['coords'], feats

    def _quality_score(self, feats):
        """Compute learned helper token quality scores."""
        return torch.sigmoid(self.quality_head(feats)).squeeze(-1)

    def _demand_match_score(self, helper_coords, demand_coords):
        """Score helper tokens by distance to nearest ego demand token."""
        if demand_coords.shape[0] == 0 or helper_coords.shape[0] == 0:
            return helper_coords.new_zeros((helper_coords.shape[0],),
                                           dtype=torch.float32)
        helper_yx = helper_coords[:, 1:3].float()
        demand_yx = demand_coords[:, 1:3].float()
        dist = torch.cdist(helper_yx, demand_yx, p=1).min(dim=1).values
        return (1.0 - dist / max(self.demand_radius, 1e-6)).clamp(min=0.0)

    def forward(self, helper_tokens, ego_demand_tokens=None):
        """
        Return selected helper supply tokens with score aligned to feats.
        """
        coords, feats = self._token_parts(helper_tokens)
        quality_score = self._quality_score(feats)

        if self.supply_mode == 'all':
            idx = torch.arange(coords.shape[0], device=coords.device)
            final_score = quality_score
        elif self.supply_mode == 'topk':
            final_score = quality_score
            _, idx = safe_topk(final_score, self.supply_topk)
        else:
            if ego_demand_tokens is None:
                demand_coords = coords[:0]
            else:
                demand_coords = ego_demand_tokens['coords']
            demand_match = self._demand_match_score(coords, demand_coords)
            final_score = quality_score + \
                self.demand_match_weight * demand_match
            _, idx = safe_topk(final_score, self.supply_topk)

        return {
            'coords': coords[idx],
            'feats': feats[idx],
            'score': final_score[idx],
        }
