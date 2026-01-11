# src/fiberlen/metrics.py
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from .types import CompressedGraph, FiberPath, Scale


def polyline_length_px(pixels: List[Tuple[int, int]]) -> float:
    """
    Weighted 8-neighborhood length:
      orthogonal step => 1
      diagonal step   => sqrt(2)
    Fallback: use Euclidean distance for unexpected steps.
    """
    if len(pixels) < 2:
        return 0.0

    total = 0.0
    for (r0, c0), (r1, c1) in zip(pixels[:-1], pixels[1:]):
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        if dr == 1 and dc == 0:
            total += 1.0
        elif dr == 0 and dc == 1:
            total += 1.0
        elif dr == 1 and dc == 1:
            total += float(np.sqrt(2.0))
        else:
            total += float(np.hypot(r1 - r0, c1 - c0))
    return total


def compute_segment_lengths(g: CompressedGraph) -> None:
    """
    Fill Segment.length_px in-place.
    """
    for seg in g.segments.values():
        seg.length_px = polyline_length_px(seg.pixels)


def compute_fiber_lengths(
    g: CompressedGraph,
    fibers: List[FiberPath],
    *,
    scale: Scale,
) -> None:
    """
    Fill FiberPath.length_px and length_um in-place.

    Note:
      This sums segment lengths. If junction blobs are multi-pixel and segments overlap inside blobs,
      there can be a small overcount. With conservative pairing this is usually minor.
    """
    for f in fibers:
        length_px = 0.0
        for sid in f.seg_ids:
            seg = g.segments[sid]
            if seg.length_px is None:
                seg.length_px = polyline_length_px(seg.pixels)
            length_px += float(seg.length_px)
        f.length_px = length_px
        f.length_um = length_px * float(scale.um_per_px)


def filter_fibers_excluding_border(fibers: List[FiberPath]) -> List[FiberPath]:
    """
    Return fibers that do NOT touch border.
    """
    return [f for f in fibers if not f.touches_border]
