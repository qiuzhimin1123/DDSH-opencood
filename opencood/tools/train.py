# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import sys
import statistics
import time

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../..')))

import torch
import tqdm
from torch.utils.data import DataLoader, DistributedSampler

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        class SummaryWriter(object):
            def __init__(self, *args, **kwargs):
                print('Warning: tensorboardX/torch tensorboard is not '
                      'available. Training continues without tensorboard '
                      'logging.')

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import inference_utils
from opencood.tools import ddsh_stats_utils
from opencood.tools import multi_gpu_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def _infer_fusion_method(hypes):
    fusion_core = hypes['fusion']['core_method']
    if fusion_core == 'LateFusionDataset':
        return 'late'
    if fusion_core == 'EarlyFusionDataset':
        return 'early'
    if fusion_core in ['IntermediateFusionDataset',
                       'IntermediateFusionDatasetV2']:
        return 'intermediate'
    raise NotImplementedError(
        'Detection evaluation only supports late, early and intermediate '
        'fusion datasets, got %s.' % fusion_core)


def run_detection_evaluation(hypes, dataset, model, device, saved_path,
                             num_workers=8):
    fusion_method = _infer_fusion_method(hypes)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             num_workers=num_workers,
                             collate_fn=dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    model.eval()
    for batch_data in tqdm.tqdm(data_loader,
                                desc='Final detection evaluation'):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          dataset)
            elif fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           dataset)
            else:
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(
                        batch_data, model, dataset)

            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.3)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.7)

    global_sort = hypes.get('train_params', {}).get(
        'global_sort_detections', False)
    eval_utils.eval_final_results(result_stat, saved_path, global_sort)


def _paper_vis_stage_dir(stage):
    mapping = {
        'sparse_single': 'stage0_sparse_single',
        'sparse_all_token': 'stage1_sparse_all_token',
        'sparse_topk': 'stage2_sparse_topk',
        'demand_supply': 'stage3_demand_supply',
        'sparse_attention': 'stage4_sparse_attention',
        'hybrid_late': 'stage5_hybrid_late',
    }
    return mapping.get(stage, str(stage))


def _paper_vis_enabled(hypes, global_step, opt):
    cfg = hypes.get('paper_vis', {})
    if not cfg.get('enable', False):
        return False
    if opt.distributed and multi_gpu_utils.get_dist_info()[0] != 0:
        return False
    interval = max(1, int(cfg.get('interval', 500)))
    return global_step % interval == 0


def _ddsh_stats_enabled(hypes, global_step, opt):
    return ddsh_stats_utils.should_collect(hypes, global_step, opt)


def _log_ddsh_stats(output_dict, hypes, saved_path, epoch, iteration,
                    global_step, batch_data, loss, lr, elapsed_ms,
                    peak_memory):
    try:
        csv_path = ddsh_stats_utils.output_path(saved_path, hypes)
        record_len = batch_data['ego'].get('record_len', None)
        row = ddsh_stats_utils.build_record(
            output_dict,
            hypes,
            phase='train',
            epoch=epoch + 1,
            iteration=iteration,
            global_step=global_step,
            batch_idx=iteration,
            record_len=record_len,
            loss=loss,
            lr=lr,
            elapsed_ms=elapsed_ms,
            peak_memory=peak_memory)
        ddsh_stats_utils.append_csv(csv_path, row)
        print('[DDSH][stats] wrote %s | stage=%s total_bytes=%s '
              'supply_tokens=%s fused_tokens=%s' %
              (csv_path, row.get('stage'), row.get('total_bytes'),
               row.get('num_supply_tokens'), row.get('num_fused_tokens')))
    except Exception as exc:
        print('Warning: DDSH statistics logging failed: %s' % exc)


def _run_paper_vis(output_dict, hypes, saved_path, epoch, iteration,
                   global_step, writer):
    cfg = hypes.get('paper_vis', {})
    if not cfg.get('enable', False):
        return
    if 'ddsh_debug' not in output_dict:
        print('Warning: paper_vis enabled but output_dict has no ddsh_debug.')
        return
    try:
        from opencood.visualization import ddsh_paper_vis
        stage = output_dict['ddsh_debug'].get(
            'stage', hypes.get('ddsh', {}).get('stage', 'unknown'))
        scene_id = int(output_dict['ddsh_debug'].get('scene_id', 0))
        max_samples = int(cfg.get('max_samples', 3))
        if scene_id >= max_samples:
            return
        stage_dir = _paper_vis_stage_dir(stage)
        save_root = os.path.join(saved_path,
                                 cfg.get('save_dir', 'paper_figures'),
                                 stage_dir)
        prefix = '%s_epoch%03d_iter%04d_scene%d' % (
            stage_dir, epoch + 1, iteration, scene_id)
        ddsh_paper_vis.visualize_ddsh_debug(
            output_dict['ddsh_debug'],
            save_dir=save_root,
            prefix=prefix,
            writer=writer,
            global_step=global_step,
            cfg=cfg)
    except Exception as exc:
        print('Warning: DDSH paper visualization failed: %s' % exc)


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)
            global_step = epoch * len(train_loader) + i
            do_paper_vis = _paper_vis_enabled(hypes, global_step, opt)
            do_ddsh_stats = _ddsh_stats_enabled(hypes, global_step, opt)
            collect_ddsh_debug = do_paper_vis or do_ddsh_stats
            ddsh_stats_cfg = hypes.get('ddsh_stats', {})
            profile_latency = bool(ddsh_stats_cfg.get('profile_latency',
                                                      True))
            profile_memory = bool(ddsh_stats_cfg.get('profile_memory', True))
            if collect_ddsh_debug:
                batch_data['ego']['ddsh_collect_debug'] = True
            if do_ddsh_stats:
                ddsh_stats_utils.maybe_reset_peak_memory(profile_memory,
                                                         device)
                ddsh_stats_utils.maybe_sync_cuda(profile_latency)
                ddsh_stats_start = time.perf_counter()
            else:
                ddsh_stats_start = None

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            try:
                if not opt.half:
                    ouput_dict = model(batch_data['ego'])
                    # first argument is always your output dictionary,
                    # second argument is always your label dictionary.
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                else:
                    with torch.cuda.amp.autocast():
                        ouput_dict = model(batch_data['ego'])
                        final_loss = criterion(ouput_dict,
                                               batch_data['ego']['label_dict'])
            finally:
                batch_data['ego'].pop('ddsh_collect_debug', None)

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            if do_paper_vis:
                _run_paper_vis(ouput_dict, hypes, saved_path, epoch, i,
                               global_step, writer)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

            if do_ddsh_stats:
                ddsh_stats_utils.maybe_sync_cuda(profile_latency)
                elapsed_ms = (time.perf_counter() - ddsh_stats_start) * 1000.0
                peak_memory = ddsh_stats_utils.peak_memory_mb(device) \
                    if profile_memory else 0.0
                cur_lr = optimizer.param_groups[0]['lr'] \
                    if optimizer.param_groups else 0.0
                _log_ddsh_stats(ouput_dict, hypes, saved_path, epoch, i,
                                global_step, batch_data, final_loss.detach(),
                                cur_lr, elapsed_ms, peak_memory)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['ddsh_compute_loss'] = True
                    try:
                        ouput_dict = model(batch_data['ego'])
                    finally:
                        batch_data['ego'].pop('ddsh_compute_loss', None)

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

    if not opt.distributed or multi_gpu_utils.get_dist_info()[0] == 0:
        eval_after_train = hypes.get('train_params', {}).get(
            'eval_detection_after_train', True)
        if eval_after_train:
            eval_workers = hypes.get('train_params', {}).get(
                'eval_detection_num_workers', 8)
            print('---------------Final Detection Evaluation------------------')
            run_detection_evaluation(hypes,
                                     opencood_validate_dataset,
                                     model_without_ddp,
                                     device,
                                     saved_path,
                                     num_workers=eval_workers)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
