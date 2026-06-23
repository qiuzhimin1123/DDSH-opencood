import torch.nn as nn


class DDSHVoxelNeXtLoss(nn.Module):
    def __init__(self, args):
        super(DDSHVoxelNeXtLoss, self).__init__()
        self.loss_dict = {}

    def forward(self, output_dict, target_dict):
        if 'loss' not in output_dict:
            raise KeyError(
                'DDSHVoxelNeXtLoss expects model output to contain "loss". '
                'The DDSH sparse head computes sparse targets inside the '
                'model from object_bbx_center/object_bbx_mask.')
        loss = output_dict['loss']
        self.loss_dict = output_dict.get('tb_dict', {})
        self.loss_dict['total_loss'] = float(loss.detach().cpu())
        return loss

    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        total = self.loss_dict.get('total_loss', 0.0)
        rpn = self.loss_dict.get('rpn_loss', total)
        if pbar is None:
            print('[epoch %d][%d/%d], || Loss: %.4f || RPN: %.4f' %
                  (epoch, batch_id + 1, batch_len, total, rpn))
        else:
            pbar.set_description(
                '[epoch %d][%d/%d], || Loss: %.4f || RPN: %.4f' %
                (epoch, batch_id + 1, batch_len, total, rpn))

        if writer is not None:
            step = epoch * batch_len + batch_id
            writer.add_scalar('Total_loss', total, step)
            for key, value in self.loss_dict.items():
                if key != 'total_loss':
                    writer.add_scalar(key, value, step)
