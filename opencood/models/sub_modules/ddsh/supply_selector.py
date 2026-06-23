import torch
import torch.nn as nn

from opencood.models.sub_modules.ddsh.sparse_tensor_utils import make_tokens


class SparseSupplySelector(nn.Module):
    def __init__(self, in_channels, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.mode = self.model_cfg.get('mode', 'demand_radius')
        self.radius = float(self.model_cfg.get('radius', 3))
        self.max_tokens_per_helper = self.model_cfg.get('max_tokens_per_helper',
                                                        4096)
        self.fallback = self.model_cfg.get('fallback', 'topk')
        self.score_mode = self.model_cfg.get('score_mode', 'norm')
        hidden = self.model_cfg.get('hidden_dim', max(16, in_channels // 2))

        if self.score_mode == 'learned':
            self.score_net = nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )
        elif self.score_mode in ['norm', 'ones']:
            self.score_net = None
        else:
            raise ValueError('Unsupported DDSH supply score_mode: %s' %
                             self.score_mode)

    def _score(self, features):
        if features.shape[0] == 0:
            return features.new_zeros((0,))
        if self.score_mode == 'learned':
            return self.score_net(features).squeeze(-1)
        if self.score_mode == 'norm':
            return torch.norm(features.detach(), dim=1)
        return features.new_ones((features.shape[0],))

    def _cap_by_score(self, features, idx):
        if self.max_tokens_per_helper is None:
            return idx
        cap = int(self.max_tokens_per_helper)
        if idx.numel() <= cap:
            return idx
        scores = self._score(features[idx])
        _, order = torch.topk(scores, k=cap)
        return idx[order]

    @staticmethod
    def _distance_stats(distances):
        if distances is None or distances.numel() == 0:
            return {
                'demand_match_distance_min': None,
                'demand_match_distance_mean': None,
                'demand_match_distance_max': None,
            }
        distances = distances.detach()
        return {
            'demand_match_distance_min': float(distances.min().cpu()),
            'demand_match_distance_mean': float(distances.mean().cpu()),
            'demand_match_distance_max': float(distances.max().cpu()),
        }

    def forward(self, helper_tokens_ego, demand_tokens, mode_override=None):
        """Select helper supply tokens after pose alignment.

        ``helper_tokens_ego`` must already be transformed into the ego BEV
        coordinate system. Demand matching distances are computed only between
        ego-frame helper coordinates and ego-frame demand coordinates.
        """
        num_aligned = int(helper_tokens_ego['features'].shape[0])
        all_idx = torch.arange(num_aligned,
                               device=helper_tokens_ego['features'].device)
        distance_stats = self._distance_stats(None)

        if num_aligned == 0:
            return make_tokens(helper_tokens_ego['features'][:0],
                               helper_tokens_ego['coords'][:0],
                               helper_tokens_ego['spatial_shape'],
                               demand_tokens['batch_size'],
                               aligned_count=0,
                               supply_count=0,
                               **distance_stats)

        mode = self.mode if mode_override is None else mode_override
        if mode == 'none':
            selected = all_idx[:0]
        elif mode == 'all':
            selected = all_idx
        elif mode == 'topk':
            selected = self._cap_by_score(helper_tokens_ego['features'],
                                          all_idx)
        elif mode == 'demand_radius':
            if demand_tokens['coords'].shape[0] == 0:
                selected = self._fallback(all_idx,
                                          helper_tokens_ego['features'])
            else:
                helper_yx = helper_tokens_ego['coords'][:, 1:3].float()
                demand_yx = demand_tokens['coords'][:, 1:3].float()
                dist = torch.cdist(helper_yx, demand_yx, p=1)
                nearest = dist.min(dim=1).values
                distance_stats = self._distance_stats(nearest)
                keep = nearest <= self.radius
                selected = all_idx[keep]
                if selected.numel() == 0:
                    selected = self._fallback(
                        all_idx, helper_tokens_ego['features'])
                selected = self._cap_by_score(
                    helper_tokens_ego['features'], selected)
        else:
            raise ValueError('Unsupported DDSH supply mode: %s' % mode)

        original_coords = helper_tokens_ego.get(
            'original_coords', helper_tokens_ego['coords'])[selected]
        return make_tokens(helper_tokens_ego['features'][selected],
                           helper_tokens_ego['coords'][selected],
                           helper_tokens_ego['spatial_shape'],
                           demand_tokens['batch_size'],
                           aligned_count=num_aligned,
                           supply_count=int(selected.numel()),
                           original_coords=original_coords,
                           **distance_stats)

    def _fallback(self, valid_idx, features):
        if self.fallback == 'none':
            return valid_idx[:0]
        if self.fallback == 'all':
            return valid_idx
        return self._cap_by_score(features, valid_idx)
