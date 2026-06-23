import torch


def _get_feats(tokens):
    """Return the feature tensor from a token dictionary."""
    if 'feats' in tokens:
        return tokens['feats']
    if 'features' in tokens:
        return tokens['features']
    raise KeyError('Token dict must contain "feats" or "features".')


def _make_like(tokens, coords, feats, **extra):
    """Build a token dictionary while preserving lightweight metadata."""
    output = {
        'coords': coords,
        'feats': feats,
    }
    for key in ['batch_size', 'spatial_shape', 'score']:
        if key in tokens and key not in extra:
            output[key] = tokens[key]
    output.update(extra)
    return output


def get_token_count(tokens):
    """Return the number of sparse tokens without materializing dense BEV."""
    if tokens is None:
        return 0
    if isinstance(tokens, torch.Tensor):
        return int(tokens.shape[0])
    if 'coords' in tokens:
        return int(tokens['coords'].shape[0])
    return int(_get_feats(tokens).shape[0])


def split_tokens_by_record_len(tokens, record_len):
    """
    Split CAV-level sparse tokens into scenes according to OpenCOOD record_len.

    Parameters
    ----------
    tokens : dict
        Sparse token dict with coords [N, 3] in [global_cav, y, x] format.
    record_len : torch.Tensor or list
        Shape [B], number of CAVs in each scene.

    Returns
    -------
    list
        scenes[scene_idx][local_cav_idx] is a token dict for that CAV. Coords
        keep their original global CAV index and metadata records scene/local
        indices, so callers can decide whether to reindex.
    """
    coords = tokens['coords']
    feats = _get_feats(tokens)
    scores = tokens.get('score', None)
    record = record_len.detach().cpu().tolist() if isinstance(
        record_len, torch.Tensor) else list(record_len)

    scenes = []
    cav_offset = 0
    for scene_idx, cav_count in enumerate(record):
        scene_tokens = []
        for local_idx in range(int(cav_count)):
            global_idx = cav_offset + local_idx
            mask = coords[:, 0] == global_idx
            extra = {
                'scene_idx': scene_idx,
                'local_cav_idx': local_idx,
                'global_cav_idx': global_idx,
            }
            if scores is not None:
                extra['score'] = scores[mask]
            scene_tokens.append(_make_like(tokens, coords[mask].clone(),
                                          feats[mask], **extra))
        scenes.append(scene_tokens)
        cav_offset += int(cav_count)
    return scenes


def build_coord_hash(coords, spatial_shape):
    """
    Convert sparse coordinates to collision-free linear hashes.

    coords may be [N, 3] = [batch, y, x] with spatial_shape [H, W], or a
    generic [N, D + 1] tensor with D spatial dimensions. The operation uses
    integer arithmetic on the same device as coords and never builds dense BEV.
    """
    if coords.dim() != 2:
        raise ValueError('coords must be a 2D tensor, got %s.' %
                         (tuple(coords.shape),))
    if coords.shape[1] != len(spatial_shape) + 1:
        raise ValueError('coords dimension %d does not match spatial_shape %s.'
                         % (coords.shape[1], spatial_shape))

    coords_long = coords.long()
    multipliers = [1]
    for size in reversed(list(spatial_shape)):
        multipliers.insert(0, multipliers[0] * int(size))
    multipliers = torch.tensor(multipliers, device=coords.device,
                               dtype=torch.long)
    return (coords_long * multipliers.view(1, -1)).sum(dim=1)


def intersect_hash(hash_a, hash_b):
    """
    Return matching indices between two hash tensors.

    Duplicate hashes in hash_a are all returned; duplicate hashes in hash_b map
    to the first matching sorted occurrence. This is sparse set matching only
    and does not allocate a dense grid.
    """
    if hash_a.numel() == 0 or hash_b.numel() == 0:
        empty_a = torch.zeros(0, device=hash_a.device, dtype=torch.long)
        empty_b = torch.zeros(0, device=hash_b.device, dtype=torch.long)
        return empty_a, empty_b

    sorted_b, order_b = torch.sort(hash_b.long())
    pos = torch.searchsorted(sorted_b, hash_a.long())
    valid = pos < sorted_b.numel()
    safe_pos = pos.clamp(max=max(sorted_b.numel() - 1, 0))
    valid = valid & (sorted_b[safe_pos] == hash_a.long())
    idx_a = valid.nonzero(as_tuple=False).view(-1)
    idx_b = order_b[safe_pos[idx_a]]
    return idx_a.long(), idx_b.long()


def safe_topk(score, k):
    """
    CUDA-safe top-k that gracefully handles k <= 0 and k > N.

    Returns
    -------
    tuple
        (top_values, top_indices), both empty when no token can be selected.
    """
    if score is None:
        raise ValueError('safe_topk requires a score tensor.')
    if score.numel() == 0 or int(k) <= 0:
        idx = torch.zeros(0, device=score.device, dtype=torch.long)
        return score.reshape(-1)[:0], idx
    k = min(int(k), int(score.numel()))
    return torch.topk(score.reshape(-1), k=k, largest=True, sorted=True)


def estimate_bandwidth(num_tokens, feature_dim, dtype, coord_dim=3):
    """
    Estimate sparse communication payload in bytes.

    Feature values use dtype.element_size(); coordinates are assumed to be
    int32 on the wire, which is sufficient for BEV indices in OpenCOOD.
    """
    if isinstance(dtype, torch.Tensor):
        elem_size = dtype.element_size()
    elif isinstance(dtype, torch.dtype):
        elem_size = torch.tensor([], dtype=dtype).element_size()
    elif isinstance(dtype, str):
        elem_size = torch.tensor([], dtype=getattr(torch, dtype)).element_size()
    else:
        raise TypeError('dtype must be a torch.dtype, tensor, or dtype string.')
    feature_bytes = int(num_tokens) * int(feature_dim) * elem_size
    coord_bytes = int(num_tokens) * int(coord_dim) * 4
    return feature_bytes + coord_bytes


def check_no_dense_tensor(batch_dict):
    """
    Validate that a batch dictionary does not contain dense BEV tensors.

    Raises RuntimeError on common dense BEV fields or any 4D tensor shaped like
    [B, C, H, W]. Recurses through nested dict/list containers.
    """
    dense_hits = []

    def _walk(obj, prefix):
        if isinstance(obj, torch.Tensor):
            if obj.dim() == 4:
                dense_hits.append('%s%s' % (prefix, tuple(obj.shape)))
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_prefix = '%s.%s' % (prefix, key) if prefix else str(key)
                if key in ['spatial_features', 'spatial_features_2d']:
                    dense_hits.append(key_prefix)
                _walk(value, key_prefix)
            return
        if isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                _walk(value, '%s[%d]' % (prefix, idx))

    _walk(batch_dict, '')
    if dense_hits:
        raise RuntimeError('Dense BEV tensor detected in DDSH path: %s' %
                           dense_hits)
    return True
