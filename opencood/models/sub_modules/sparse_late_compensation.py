import torch
import torch.nn as nn


class SparseLateCompensation(nn.Module):
    """
    Optional late box compensation for hybrid_late stage.
    """
    def __init__(self, model_cfg=None):
        super(SparseLateCompensation, self).__init__()
        self.model_cfg = model_cfg or {}
        cfg = self.model_cfg.get('late_compensation', self.model_cfg)
        self.enable = bool(cfg.get('enable', cfg.get('enabled', False)))
        self.topk_boxes = int(cfg.get('topk_boxes', 10))
        self.score_thresh = float(cfg.get('score_thresh', 0.3))
        self.nms_thresh = float(cfg.get('nms_thresh', 0.15))

    @staticmethod
    def _empty_like(device):
        """Create an empty OpenCOOD-style prediction dict."""
        return {
            'pred_boxes': torch.zeros((0, 7), device=device),
            'pred_scores': torch.zeros((0,), device=device),
            'pred_labels': torch.zeros((0,), device=device, dtype=torch.long),
        }

    def _select_helper_boxes(self, helper_dict):
        """Filter one helper prediction dict by score threshold and top-k."""
        boxes = helper_dict.get('pred_boxes')
        scores = helper_dict.get('pred_scores')
        labels = helper_dict.get('pred_labels')
        if boxes is None or scores is None:
            raise KeyError('helper late boxes need pred_boxes and pred_scores.')
        if labels is None:
            labels = torch.ones_like(scores, dtype=torch.long)
        keep = scores >= self.score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if scores.numel() == 0:
            return self._empty_like(boxes.device)
        k = min(self.topk_boxes, scores.numel())
        _, idx = torch.topk(scores, k=k)
        return {
            'pred_boxes': boxes[idx],
            'pred_scores': scores[idx],
            'pred_labels': labels[idx],
        }

    def _nms(self, boxes, scores, labels):
        """Run project-level rotated NMS if boxes are available."""
        if boxes.shape[0] == 0:
            return boxes, scores, labels
        from opencood.utils import box_utils
        corners = box_utils.boxes_to_corners_3d(boxes, order='lwh')
        keep = box_utils.nms_rotated(corners, scores, self.nms_thresh)
        keep = torch.as_tensor(keep, device=boxes.device, dtype=torch.long)
        return boxes[keep], scores[keep], labels[keep]

    def forward(self, fused_prediction, helper_late_boxes=None):
        """
        Merge fused intermediate predictions with helper late boxes.

        If helper_late_boxes is missing or interfaces are unclear, this module
        returns fused_prediction unchanged so non-hybrid stages are unaffected.
        """
        if not self.enable or helper_late_boxes is None:
            return fused_prediction

        if not isinstance(fused_prediction, list):
            fused_list = [fused_prediction]
            unwrap = True
        else:
            fused_list = fused_prediction
            unwrap = False

        output = []
        for batch_idx, ego_pred in enumerate(fused_list):
            helper_entries = helper_late_boxes[batch_idx] \
                if isinstance(helper_late_boxes, list) and \
                batch_idx < len(helper_late_boxes) else helper_late_boxes
            if isinstance(helper_entries, dict):
                helper_entries = [helper_entries]

            boxes = [ego_pred.get('pred_boxes')]
            scores = [ego_pred.get('pred_scores')]
            labels = [ego_pred.get('pred_labels')]

            for helper_dict in helper_entries:
                selected = self._select_helper_boxes(helper_dict)
                boxes.append(selected['pred_boxes'].to(boxes[0].device))
                scores.append(selected['pred_scores'].to(scores[0].device))
                labels.append(selected['pred_labels'].to(labels[0].device))

            merged_boxes = torch.cat(boxes, dim=0)
            merged_scores = torch.cat(scores, dim=0)
            merged_labels = torch.cat(labels, dim=0)
            merged_boxes, merged_scores, merged_labels = self._nms(
                merged_boxes, merged_scores, merged_labels)
            output.append({
                'pred_boxes': merged_boxes,
                'pred_scores': merged_scores,
                'pred_labels': merged_labels,
            })

        return output[0] if unwrap else output
