import torch
import torch.nn as nn


class LateBoxCompensation(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg or {}
        self.enabled = bool(self.model_cfg.get('enabled', False))
        self.require_input = bool(self.model_cfg.get('require_input', False))
        self.source_keys = self.model_cfg.get(
            'source_keys', ['helper_late_box_dicts', 'late_box_dicts'])

    def forward(self, final_box_dicts, data_dict, pairwise_t_matrix=None):
        if not self.enabled:
            return final_box_dicts

        late_dicts = None
        for key in self.source_keys:
            if key in data_dict:
                late_dicts = data_dict[key]
                break

        if late_dicts is None:
            if self.require_input:
                raise KeyError(
                    'DDSH late compensation is enabled but none of %s were '
                    'found in data_dict.' % self.source_keys)
            return final_box_dicts

        if len(final_box_dicts) != len(late_dicts):
            raise ValueError(
                'Late compensation expects one late prediction list per ego '
                'sample. Got %d fused and %d late entries.' %
                (len(final_box_dicts), len(late_dicts)))

        merged = []
        for fused, late in zip(final_box_dicts, late_dicts):
            cur = {}
            for key in ['pred_boxes', 'pred_scores', 'pred_labels']:
                if key not in fused:
                    continue
                if key in late and late[key] is not None:
                    cur[key] = torch.cat([fused[key], late[key].to(
                        fused[key].device)], dim=0)
                else:
                    cur[key] = fused[key]
            merged.append(cur)
        return merged
