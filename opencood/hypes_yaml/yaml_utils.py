# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import re
import yaml
import os
import math

import numpy as np


def load_yaml(file, opt=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param


def load_voxel_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute `anchor_args[W][H][L]`
    """
    anchor_args = param['postprocess']['anchor_args']
    cav_lidar_range = anchor_args['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    # sometimes we just want to visualize the data without implementing model
    if 'model' in param:
        param['model']['args']['W'] = anchor_args['W']
        param['model']['args']['H'] = anchor_args['H']
        param['model']['args']['D'] = anchor_args['D']

    return param


def load_point_pillar_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['model']['args']['point_pillar_scatter']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    return param


def load_second_params(param):
    """
    Based on the lidar range and resolution of voxel, calcuate the anchor box
    and target resolution.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['model']['args']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = int(grid_size[0])
    anchor_args['H'] = int(grid_size[1])
    anchor_args['D'] = int(grid_size[2])

    param['postprocess'].update({'anchor_args': anchor_args})

    return param


def load_voxelnext_params(param):
    """
    Calculate geometry-dependent fields for VoxelNeXt backbone configs.

    VoxelNeXt keeps the OpenCOOD anchor loss/postprocess in this integration,
    so it needs both sparse-conv grid_size and VoxelPostprocessor anchor size.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['model']['args']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = int(grid_size[0])
    anchor_args['H'] = int(grid_size[1])
    anchor_args['D'] = int(grid_size[2])

    param['postprocess'].update({'anchor_args': anchor_args})

    return param


def load_ddsh_voxelnext_params(param):
    """
    Geometry helper for DDSH-VoxelNeXt.

    DDSH keeps the VoxelNeXt sparse grid but does not require dense BEV anchor
    labels. The anchor_args block is still populated so existing OpenCOOD
    dataset utilities can derive object centers and ranges without special
    cases.
    """
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']
    model_args = param['model']['args']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    model_args['grid_size'] = grid_size

    model_args['voxel_size'] = voxel_size
    model_args['lidar_range'] = cav_lidar_range
    model_args.setdefault('class_names', ['vehicle'])
    model_args.setdefault('box_order', param['postprocess'].get('order',
                                                                'hwl'))

    # The stage YAML files preserve the original VoxelNeXt OpenCOOD blocks for
    # readability. At runtime DDSH needs its sparse loss/postprocessor because
    # it does not emit dense psm/rm maps.
    param.setdefault('loss', {})['core_method'] = 'ddsh_voxelnext_loss'
    param['loss'].setdefault('args', {})
    param['postprocess']['core_method'] = 'DdshVoxelNeXtPostprocessor'

    anchor_args = param['postprocess'].setdefault('anchor_args', {})
    anchor_args['cav_lidar_range'] = anchor_args.get('cav_lidar_range',
                                                     cav_lidar_range)
    anchor_args['vw'] = voxel_size[0]
    anchor_args['vh'] = voxel_size[1]
    anchor_args['vd'] = voxel_size[2]
    anchor_args['W'] = int(grid_size[0])
    anchor_args['H'] = int(grid_size[1])
    anchor_args['D'] = int(grid_size[2])
    param['postprocess']['anchor_args'] = anchor_args

    # Mirror stage YAML blocks into model.args so ddsh_voxelnext can be
    # controlled from top-level YAML sections without repeating them.
    top_ddsh = param.get('ddsh', {})
    communication = param.get('communication', {})
    sparse_bev = param.get('sparse_bev', {})
    sparse_fusion = param.get('sparse_fusion', {})
    sparse_attention = param.get('sparse_attention', {})
    late_compensation = param.get('late_compensation', {})

    for key, value in [('ddsh', top_ddsh),
                       ('communication', communication),
                       ('sparse_bev', sparse_bev),
                       ('sparse_fusion', sparse_fusion),
                       ('sparse_attention', sparse_attention),
                       ('late_compensation', late_compensation)]:
        if value:
            model_args[key] = value

    model_args['sparse_height_compression'] = {
        'source_key': 'encoded_spconv_tensor',
        'reduce': sparse_bev.get('reduce', 'max'),
    }

    ddsh_cfg = model_args.setdefault('ddsh', {})
    ddsh_cfg.setdefault('stage', top_ddsh.get('stage', 'demand_supply'))
    ddsh_cfg.setdefault('debug', top_ddsh.get('debug', False))

    demand_cfg = ddsh_cfg.setdefault('demand', {})
    demand_cfg.setdefault('mode', 'topk')
    demand_cfg.setdefault('score_mode', 'learned')
    demand_cfg['topk'] = communication.get('demand_topk',
                                           demand_cfg.get('topk', 1024))
    demand_cfg.setdefault('min_tokens', 0)

    supply_cfg = ddsh_cfg.setdefault('supply', {})
    supply_mode = communication.get('supply_mode',
                                    supply_cfg.get('mode', 'demand_radius'))
    if supply_mode == 'demand_supply':
        supply_mode = 'demand_radius'
    supply_cfg['mode'] = supply_mode
    supply_cfg.setdefault('score_mode', 'norm')
    supply_topk = communication.get('supply_topk',
                                    supply_cfg.get('max_tokens_per_helper',
                                                   4096))
    supply_cfg['max_tokens_per_helper'] = None if int(supply_topk) < 0 \
        else int(supply_topk)
    supply_cfg['radius'] = communication.get('demand_radius',
                                             supply_cfg.get('radius', 4))
    supply_cfg['demand_match_weight'] = communication.get(
        'demand_match_weight', supply_cfg.get('demand_match_weight', 1.0))
    supply_cfg.setdefault('fallback', 'topk')

    fusion_cfg = ddsh_cfg.setdefault('fusion', {})
    fusion_type = sparse_fusion.get('type', fusion_cfg.get('method',
                                                           'gated_mean'))
    fusion_cfg['method'] = 'gated_mean' if fusion_type in [
        'none', 'hash_mlp', 'local_attention'] else fusion_type

    local_attention_cfg = ddsh_cfg.setdefault('local_attention', {})
    local_attention_cfg['enabled'] = sparse_attention.get(
        'enable', local_attention_cfg.get('enabled', False))
    local_attention_cfg['radius'] = sparse_attention.get(
        'radius', local_attention_cfg.get('radius', 2))
    local_attention_cfg['max_neighbors'] = sparse_attention.get(
        'max_neighbors', local_attention_cfg.get('max_neighbors', 32))
    local_attention_cfg.setdefault('max_attention_tokens', 4096)

    if late_compensation:
        model_args['late_compensation'] = {
            'enabled': late_compensation.get(
                'enable', late_compensation.get('enabled', False)),
            'topk_boxes': late_compensation.get('topk_boxes', 10),
            'score_thresh': late_compensation.get('score_thresh', 0.3),
            'nms_thresh': late_compensation.get('nms_thresh', 0.15),
            'require_input': late_compensation.get('require_input', False),
        }

    sparse_head = model_args.setdefault('sparse_head', {})
    sparse_head.setdefault('CLASS_AGNOSTIC', False)
    sparse_head.setdefault('CLASS_NAMES_EACH_HEAD', [
        model_args.get('class_names', ['vehicle'])])
    sparse_head.setdefault('SHARED_CONV_CHANNEL',
                           model_args['backbone_3d'].get('OUT_CHANNEL', 128))
    sparse_head.setdefault('KERNEL_SIZE_HEAD', 1)
    sparse_head.setdefault('USE_BIAS_BEFORE_NORM', True)
    sparse_head.setdefault('NUM_HM_CONV', 2)
    sep_head = sparse_head.setdefault('SEPARATE_HEAD_CFG', {})
    sep_head.setdefault('HEAD_ORDER', ['center', 'center_z', 'dim', 'rot'])
    sep_head.setdefault('HEAD_DICT', {
        'center': {'out_channels': 2, 'num_conv': 2},
        'center_z': {'out_channels': 1, 'num_conv': 2},
        'dim': {'out_channels': 3, 'num_conv': 2},
        'rot': {'out_channels': 2, 'num_conv': 2},
    })
    target_cfg = sparse_head.setdefault('TARGET_ASSIGNER_CONFIG', {})
    feature_stride = target_cfg.get('FEATURE_MAP_STRIDE',
                                    model_args.get('feature_stride', 8))
    target_cfg['FEATURE_MAP_STRIDE'] = feature_stride
    target_cfg.setdefault('NUM_MAX_OBJS', 500)
    target_cfg.setdefault('GAUSSIAN_OVERLAP', 0.1)
    target_cfg.setdefault('MIN_RADIUS', 2)
    sparse_head['TARGET_ASSIGNER_CONFIG'] = target_cfg

    loss_cfg = sparse_head.setdefault('LOSS_CONFIG', {})
    loss_cfg.setdefault('LOSS_WEIGHTS', {
        'cls_weight': 1.0,
        'loc_weight': 1.0,
        'code_weights': [1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0],
    })

    post_cfg = sparse_head.setdefault('POST_PROCESSING', {})
    post_cfg.setdefault('SCORE_THRESH', 0.1)
    post_cfg.setdefault('POST_CENTER_LIMIT_RANGE', cav_lidar_range)
    post_cfg.setdefault('MAX_OBJ_PER_SAMPLE', 500)
    post_cfg.setdefault('NMS_CONFIG', {
        'NMS_TYPE': 'nms_rotated',
        'NMS_THRESH': 0.1,
        'NMS_PRE_MAXSIZE': 4096,
        'NMS_POST_MAXSIZE': 500,
    })
    sparse_head['POST_PROCESSING'] = post_cfg

    return param


def load_bev_params(param):
    """
    Load bev related geometry parameters s.t. boundary, resolutions, input
    shape, target shape etc.

    Parameters
    ----------
    param : dict
        Original loaded parameter dictionary.

    Returns
    -------
    param : dict
        Modified parameter dictionary with new attribute `geometry_param`.

    """
    res = param["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, H2 = param["preprocess"]["cav_lidar_range"]
    downsample_rate = param["preprocess"]["args"]["downsample_rate"]

    def f(low, high, r):
        return int((high - low) / r)

    input_shape = (
        int((f(L1, L2, res))),
        int((f(W1, W2, res))),
        int((f(H1, H2, res)) + 1)
    )
    label_shape = (
        int(input_shape[0] / downsample_rate),
        int(input_shape[1] / downsample_rate),
        7
    )
    geometry_param = {
        'L1': L1,
        'L2': L2,
        'W1': W1,
        'W2': W2,
        'H1': H1,
        'H2': H2,
        "downsample_rate": downsample_rate,
        "input_shape": input_shape,
        "label_shape": label_shape,
        "res": res
    }
    param["preprocess"]["geometry_param"] = geometry_param
    param["postprocess"]["geometry_param"] = geometry_param
    param["model"]["args"]["geometry_param"] = geometry_param
    return param


def save_yaml(data, save_name):
    """
    Save the dictionary into a yaml file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """

    with open(save_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def save_yaml_wo_overwriting(data, save_name):
    """
    Save the yaml file without overwriting the existing one.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """
    if os.path.exists(save_name):
        prev_data = load_yaml(save_name)
        data = {**data, **prev_data}

    save_yaml(data, save_name)
