import torch

from opencood.models.sub_modules.spconv_utils import spconv


def ensure_2d_sparse_tensor(sp_tensor, name='encoded_spconv_tensor'):
    if not hasattr(sp_tensor, 'features') or not hasattr(sp_tensor, 'indices'):
        raise TypeError('%s must be a spconv SparseConvTensor.' % name)
    if sp_tensor.indices.dim() != 2:
        raise ValueError('%s.indices must be a 2D tensor.' % name)
    if len(sp_tensor.spatial_shape) != 2 or sp_tensor.indices.shape[1] != 3:
        raise ValueError(
            '%s must be a 2D sparse BEV tensor with indices [batch, y, x]. '
            'DDSH does not accept dense BEV tensors or 3D tensors at this '
            'stage. Got spatial_shape=%s and indices shape=%s.' %
            (name, list(sp_tensor.spatial_shape), tuple(sp_tensor.indices.shape))
        )


def make_tokens(features, coords, spatial_shape, batch_size, **extra):
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError('Sparse BEV token coords must have shape [N, 3].')
    tokens = {
        'features': features,
        'coords': coords.long(),
        'spatial_shape': list(spatial_shape),
        'batch_size': int(batch_size),
    }
    tokens.update(extra)
    return tokens


def tokens_from_sparse_tensor(sp_tensor):
    ensure_2d_sparse_tensor(sp_tensor)
    return make_tokens(
        sp_tensor.features,
        sp_tensor.indices.long(),
        sp_tensor.spatial_shape,
        sp_tensor.batch_size,
    )


def empty_tokens(reference, spatial_shape, batch_size):
    if isinstance(reference, dict):
        ref_features = reference['features']
        channels = ref_features.shape[1] if ref_features.dim() == 2 else 0
        device = ref_features.device
        dtype = ref_features.dtype
    else:
        channels = reference.shape[1]
        device = reference.device
        dtype = reference.dtype
    return make_tokens(
        torch.zeros((0, channels), device=device, dtype=dtype),
        torch.zeros((0, 3), device=device, dtype=torch.long),
        spatial_shape,
        batch_size,
    )


def select_batch(tokens, batch_idx, out_batch_idx=None):
    coords = tokens['coords']
    mask = coords[:, 0] == int(batch_idx)
    features = tokens['features'][mask]
    selected_coords = coords[mask].clone()
    if out_batch_idx is not None and selected_coords.numel() > 0:
        selected_coords[:, 0] = int(out_batch_idx)
    return make_tokens(
        features,
        selected_coords,
        tokens['spatial_shape'],
        tokens['batch_size'] if out_batch_idx is None else max(int(out_batch_idx) + 1, 1),
    )


def concatenate_tokens(token_list, spatial_shape=None, batch_size=None):
    token_list = [tokens for tokens in token_list
                  if tokens is not None and tokens['features'].shape[0] > 0]
    if len(token_list) == 0:
        if spatial_shape is None or batch_size is None:
            raise ValueError('Cannot infer empty token shape without metadata.')
        raise ValueError('Cannot concatenate an empty DDSH token list.')

    features = torch.cat([tokens['features'] for tokens in token_list], dim=0)
    coords = torch.cat([tokens['coords'] for tokens in token_list], dim=0)
    return make_tokens(
        features,
        coords,
        spatial_shape or token_list[0]['spatial_shape'],
        batch_size if batch_size is not None else token_list[0]['batch_size'],
    )


def unique_reduce(features, coords, spatial_shape, batch_size, reduce='mean',
                  weights=None):
    if features.shape[0] == 0:
        return make_tokens(features, coords, spatial_shape, batch_size)

    unique_coords, inverse = torch.unique(coords.long(),
                                          dim=0,
                                          sorted=True,
                                          return_inverse=True)
    out = features.new_zeros((unique_coords.shape[0], features.shape[1]))

    if reduce == 'max':
        out.fill_(-float('inf'))
        for idx in range(features.shape[0]):
            out[inverse[idx]] = torch.maximum(out[inverse[idx]], features[idx])
        out[out == -float('inf')] = 0
        return make_tokens(out, unique_coords, spatial_shape, batch_size)

    if weights is not None:
        weights = weights.reshape(-1, 1).type_as(features)
        out.index_add_(0, inverse, features * weights)
        denom = features.new_zeros((unique_coords.shape[0], 1))
        denom.index_add_(0, inverse, weights)
        out = out / denom.clamp_min(1e-6)
    else:
        out.index_add_(0, inverse, features)
        if reduce == 'mean':
            counts = features.new_zeros((unique_coords.shape[0], 1))
            counts.index_add_(0, inverse, torch.ones(
                (features.shape[0], 1), device=features.device,
                dtype=features.dtype))
            out = out / counts.clamp_min(1.0)
        elif reduce != 'sum':
            raise ValueError('Unsupported sparse reduce mode: %s' % reduce)

    return make_tokens(out, unique_coords, spatial_shape, batch_size)


def tokens_to_sparse_tensor(tokens):
    coords = tokens['coords'].long()
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError('DDSH fused coords must have shape [N, 3].')
    return spconv.SparseConvTensor(
        features=tokens['features'],
        indices=coords.int(),
        spatial_shape=list(tokens['spatial_shape']),
        batch_size=int(tokens['batch_size']),
    )


def assert_no_dense_bev(batch_dict):
    dense_keys = find_dense_bev_keys(batch_dict)
    if dense_keys:
        raise RuntimeError(
            'DDSH-VoxelNeXt received dense BEV fields %s. Remove dense BEV '
            'modules from this path.' % dense_keys
        )


def find_dense_bev_keys(batch_dict):
    forbidden = ['spatial_features_2d', 'spatial_features']
    dense_keys = []
    for key in forbidden:
        if key in batch_dict:
            value = batch_dict[key]
            if isinstance(value, torch.Tensor) and value.dim() == 4:
                dense_keys.append('%s%s' % (key, tuple(value.shape)))
            else:
                dense_keys.append(key)
    return dense_keys
