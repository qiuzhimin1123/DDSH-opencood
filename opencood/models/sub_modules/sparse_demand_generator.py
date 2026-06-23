import torch
import torch.nn as nn

from opencood.models.sub_modules.ddsh_sparse_utils import safe_topk


class SparseDemandGenerator(nn.Module):
    """
    Generate ego demand tokens from sparse BEV tokens.
    """
    def __init__(self, in_channels, model_cfg=None):
        super(SparseDemandGenerator, self).__init__()
        self.model_cfg = model_cfg or {}
        comm_cfg = self.model_cfg.get('communication', self.model_cfg)
        self.demand_topk = comm_cfg.get('demand_topk', 1024)
        self.score_head = nn.Linear(in_channels, 1)

    def forward(self, ego_tokens):
        """
        Select top-K ego tokens by learned demand score.
        """
        coords = ego_tokens['coords']
        feats = ego_tokens.get('feats', ego_tokens.get('features'))
        if feats is None:
            raise KeyError('ego_tokens must contain "feats" or "features".')

        demand_score = torch.sigmoid(self.score_head(feats)).squeeze(-1)

        # Reserved terms for future DDSH scoring variants.
        uncertainty_score = torch.zeros_like(demand_score)
        low_density_score = torch.zeros_like(demand_score)
        distance_score = torch.zeros_like(demand_score)
        foreground_prior = torch.zeros_like(demand_score)

        _, idx = safe_topk(demand_score, self.demand_topk)
        return {
            'coords': coords[idx],
            'feats': feats[idx],
            'score': demand_score[idx],
            'score_terms': {
                'uncertainty_score': uncertainty_score[idx],
                'low_density_score': low_density_score[idx],
                'distance_score': distance_score[idx],
                'foreground_prior': foreground_prior[idx],
            },
        }
