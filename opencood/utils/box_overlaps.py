# -*- coding: utf-8 -*-
"""Numpy fallback for the Cython box_overlaps extension.

The repository may contain a prebuilt ``box_overlaps`` shared object compiled
for a different Python ABI. Keeping this module lets imports continue to work
without rebuilding the Cython extension; Python will still prefer a compatible
extension module when one exists.
"""

import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """Compute IoU overlaps between two sets of 2D boxes.

    Parameters
    ----------
    boxes : np.ndarray
        Shape [N, 4], boxes in [x1, y1, x2, y2] format.
    query_boxes : np.ndarray
        Shape [K, 4], boxes in [x1, y1, x2, y2] format.

    Returns
    -------
    np.ndarray
        Shape [N, K], IoU overlap matrix.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    overlaps = np.zeros((n, k), dtype=np.float32)

    if n == 0 or k == 0:
        return overlaps

    box_area = (
        (boxes[:, 2] - boxes[:, 0] + 1.0) *
        (boxes[:, 3] - boxes[:, 1] + 1.0)
    )
    query_area = (
        (query_boxes[:, 2] - query_boxes[:, 0] + 1.0) *
        (query_boxes[:, 3] - query_boxes[:, 1] + 1.0)
    )

    for query_idx in range(k):
        iw = np.minimum(boxes[:, 2], query_boxes[query_idx, 2]) - \
            np.maximum(boxes[:, 0], query_boxes[query_idx, 0]) + 1.0
        ih = np.minimum(boxes[:, 3], query_boxes[query_idx, 3]) - \
            np.maximum(boxes[:, 1], query_boxes[query_idx, 1]) + 1.0
        valid = np.logical_and(iw > 0, ih > 0)
        inter = iw[valid] * ih[valid]
        union = box_area[valid] + query_area[query_idx] - inter
        overlaps[valid, query_idx] = inter / np.maximum(union, 1e-6)

    return overlaps


def bbox_intersections(boxes, query_boxes):
    """Compute intersection ratio over query box area."""
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    intersec = np.zeros((n, k), dtype=np.float32)

    if n == 0 or k == 0:
        return intersec

    query_area = (
        (query_boxes[:, 2] - query_boxes[:, 0] + 1.0) *
        (query_boxes[:, 3] - query_boxes[:, 1] + 1.0)
    )

    for query_idx in range(k):
        iw = np.minimum(boxes[:, 2], query_boxes[query_idx, 2]) - \
            np.maximum(boxes[:, 0], query_boxes[query_idx, 0]) + 1.0
        ih = np.minimum(boxes[:, 3], query_boxes[query_idx, 3]) - \
            np.maximum(boxes[:, 1], query_boxes[query_idx, 1]) + 1.0
        valid = np.logical_and(iw > 0, ih > 0)
        inter = iw[valid] * ih[valid]
        intersec[valid, query_idx] = inter / np.maximum(
            query_area[query_idx], 1e-6)

    return intersec


def box_vote(dets_nms, dets_all):
    """Weighted box voting fallback matching the Cython helper signature."""
    dets_nms = np.asarray(dets_nms, dtype=np.float32)
    dets_all = np.asarray(dets_all, dtype=np.float32)
    dets_voted = np.zeros_like(dets_nms, dtype=np.float32)

    for idx, det in enumerate(dets_nms):
        overlaps = bbox_overlaps(det[None, :4], dets_all[:, :4])[0]
        keep = overlaps >= 0.5
        if not np.any(keep):
            dets_voted[idx] = det
            continue
        weights = dets_all[keep, 4]
        weight_sum = np.maximum(np.sum(weights), 1e-6)
        dets_voted[idx, :4] = np.sum(dets_all[keep, :4] *
                                     weights[:, None], axis=0) / weight_sum
        dets_voted[idx, 4] = det[4]

    return dets_voted
