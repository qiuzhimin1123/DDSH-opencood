import torch
import torch.nn as nn

from opencood.models.sub_modules.ddsh.sparse_tensor_utils import make_tokens


class SparseDemandGenerator(nn.Module):
    def __init__(self, in_channels, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg or {}
        hidden = self.model_cfg.get('hidden_dim', max(16, in_channels // 2))
        self.score_mode = self.model_cfg.get('score_mode', 'learned')
        self.mode = self.model_cfg.get('mode', 'topk')
        self.topk = self.model_cfg.get('topk', 2048)
        self.ratio = self.model_cfg.get('ratio', None)
        self.min_tokens = self.model_cfg.get('min_tokens', 1)
        self.score_threshold = self.model_cfg.get('score_threshold', None)

        if self.score_mode == 'learned':
            self.score_net = nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 1),
            )
        elif self.score_mode in ['norm', 'ones']:
            self.score_net = None
        else:
            raise ValueError('Unsupported DDSH demand score_mode: %s' %
                             self.score_mode)

    def _score(self, features):
        if features.shape[0] == 0:
            return features.new_zeros((0,))
        if self.score_mode == 'learned':
            return self.score_net(features).squeeze(-1)
        if self.score_mode == 'norm':
            return torch.norm(features.detach(), dim=1)
        return features.new_ones((features.shape[0],))

    def _target_k(self, num_tokens):
        if self.mode == 'all':
            return num_tokens
        candidates = []
        if self.topk is not None:
            candidates.append(int(self.topk))
        if self.ratio is not None:
            candidates.append(int(max(1, round(float(self.ratio) *
                                               num_tokens))))
        if len(candidates) == 0:
            candidates.append(num_tokens)
        k = min(num_tokens, max(int(self.min_tokens), min(candidates)))
        return max(0, k)

    def forward(self, ego_tokens):
        features = ego_tokens['features']
        coords = ego_tokens['coords']
        scores = self._score(features)
        num_tokens = features.shape[0]

        if num_tokens == 0:
            return make_tokens(features, coords, ego_tokens['spatial_shape'],
                               ego_tokens['batch_size'], demand_scores=scores)

        if self.mode == 'threshold':
            if self.score_threshold is None:
                raise ValueError(
                    'DDSH demand mode "threshold" requires score_threshold.')
            keep = torch.sigmoid(scores) >= float(self.score_threshold)
            if keep.sum().item() < min(self.min_tokens, num_tokens):
                k = min(num_tokens, int(self.min_tokens))
                _, top_idx = torch.topk(scores, k=k)
                keep = torch.zeros_like(scores, dtype=torch.bool)
                keep[top_idx] = True
            idx = keep.nonzero(as_tuple=False).view(-1)
        else:
            k = self._target_k(num_tokens)
            if k == num_tokens:
                idx = torch.arange(num_tokens, device=features.device)
            else:
                _, idx = torch.topk(scores, k=k)

        return make_tokens(features[idx], coords[idx],
                           ego_tokens['spatial_shape'],
                           ego_tokens['batch_size'],
                           demand_scores=scores[idx],
                           demand_all_scores=scores)
