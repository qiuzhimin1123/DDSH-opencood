import torch
import torch.nn as nn
import torch.nn.functional as F


class DDSHAuxLoss(nn.Module):
    """
    Optional DDSH auxiliary losses.

    This module is intentionally disabled by default. It can supervise demand
    scores with BCE and softly constrain communication token budget when the
    model starts exposing those tensors in output_dict.
    """
    def __init__(self, model_cfg=None):
        super(DDSHAuxLoss, self).__init__()
        model_cfg = model_cfg or {}
        cfg = model_cfg.get('ddsh_loss', model_cfg)
        self.enable = bool(cfg.get('enable', False))
        self.demand_weight = float(cfg.get('demand_weight', 1.0))
        self.budget_weight = float(cfg.get('budget_weight', 1.0))
        self.target_budget = cfg.get('target_budget', None)
        self.loss_dict = {}

    def _zero(self, output_dict):
        """Return a device-correct zero scalar when aux loss is disabled."""
        for value in output_dict.values():
            if isinstance(value, torch.Tensor):
                return value.sum() * 0.0
        return torch.tensor(0.0)

    def forward(self, output_dict, target_dict=None):
        """
        Compute optional demand BCE and communication budget losses.
        """
        if not self.enable:
            loss = self._zero(output_dict)
            self.loss_dict = {'ddsh_aux_loss': float(loss.detach().cpu())}
            return loss

        total = self._zero(output_dict)
        demand_loss = total
        if 'demand_logits' in output_dict:
            if target_dict is None or 'demand_target' not in target_dict:
                raise KeyError('demand BCE requires target_dict[demand_target].')
            demand_loss = F.binary_cross_entropy_with_logits(
                output_dict['demand_logits'],
                target_dict['demand_target'].float())
            total = total + self.demand_weight * demand_loss

        budget_loss = total * 0.0
        if self.target_budget is not None:
            token_count = output_dict.get('communication_tokens', None)
            if token_count is None and 'ddsh_stats' in output_dict:
                token_count = output_dict['ddsh_stats'].get(
                    'num_supply_tokens', None)
            if token_count is not None:
                if not isinstance(token_count, torch.Tensor):
                    token_count = total.new_tensor(float(token_count))
                budget = total.new_tensor(float(self.target_budget))
                budget_loss = F.relu(token_count.float() - budget) / \
                    budget.clamp_min(1.0)
                total = total + self.budget_weight * budget_loss

        self.loss_dict = {
            'ddsh_aux_loss': float(total.detach().cpu()),
            'demand_bce_loss': float(demand_loss.detach().cpu()),
            'communication_budget_loss': float(budget_loss.detach().cpu()),
        }
        return total
