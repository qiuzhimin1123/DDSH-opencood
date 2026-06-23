import torch

from opencood.data_utils.post_processor.base_postprocessor import (
    BasePostprocessor,
)
from opencood.utils import box_utils


class DdshVoxelNeXtPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, train):
        super(DdshVoxelNeXtPostprocessor, self).__init__(anchor_params,
                                                         train)

    def generate_anchor_box(self):
        return None

    def generate_label(self, **kwargs):
        return {}

    @staticmethod
    def collate_batch(label_batch_list):
        return {}

    def post_process(self, data_dict, output_dict):
        pred_box3d_list = []
        score_list = []

        for cav_id, cav_content in data_dict.items():
            if cav_id not in output_dict:
                raise KeyError('Missing output for cav_id=%s.' % cav_id)
            cur_output = output_dict[cav_id]
            if 'final_box_dicts' not in cur_output:
                raise KeyError(
                    'DdshVoxelNeXtPostprocessor expects final_box_dicts from '
                    'DDSH sparse head.')

            transformation_matrix = cav_content.get(
                'transformation_matrix',
                torch.eye(4, device=cur_output['final_box_dicts'][0][
                    'pred_boxes'].device))

            for final_dict in cur_output['final_box_dicts']:
                boxes = final_dict['pred_boxes']
                scores = final_dict['pred_scores']
                if boxes.shape[0] == 0:
                    continue
                corners = box_utils.boxes_to_corners_3d(boxes, order='lwh')
                projected = box_utils.project_box3d(corners,
                                                    transformation_matrix)
                pred_box3d_list.append(projected)
                score_list.append(scores)

        if len(pred_box3d_list) == 0:
            return None, None

        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        scores = torch.cat(score_list, dim=0)

        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]

        if pred_box3d_tensor.shape[0] == 0:
            return None, None

        nms_thresh = self.params.get('nms_thresh', None)
        if nms_thresh is not None:
            keep_index = box_utils.nms_rotated(pred_box3d_tensor, scores,
                                               nms_thresh)
            keep_index = torch.as_tensor(keep_index,
                                         device=pred_box3d_tensor.device,
                                         dtype=torch.long)
            pred_box3d_tensor = pred_box3d_tensor[keep_index]
            scores = scores[keep_index]

        mask = box_utils.get_mask_for_boxes_within_range_torch(
            pred_box3d_tensor)
        pred_box3d_tensor = pred_box3d_tensor[mask]
        scores = scores[mask]

        if pred_box3d_tensor.shape[0] == 0:
            return None, None
        return pred_box3d_tensor, scores
