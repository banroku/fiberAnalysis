
# src/fiberlen/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np

from .io import read_skeleton_tif
from .skeleton_graph import build_compressed_graph
from .pairing import compute_pairings_conservatively
from .tracing import trace_fibers
from .metrics import compute_segment_lengths, compute_fiber_lengths, filter_fibers_excluding_border
from .types import Scale


@dataclass(frozen=True)
class RunParams:
    # IO
    foreground: str = "auto"
    pad: int = 0

    # Graph
    connectivity: int = 8
    border_margin_px: int = 5

    # Pairing
    probe_len: int = 12
    max_cost_accept: float = 0.35

    # Scale / filtering
    um_per_px: float = 1.0
    exclude_border_touching: bool = True

    # Plot params
    hist_bins: int = 30
    discrete_round_px: int = 5


def run_pipeline(image_path: Path, p: RunParams) -> Dict[str, Any]:
    sk = read_skeleton_tif(image_path, foreground=p.foreground, pad=p.pad)

    g, _, _ = build_compressed_graph(
        sk.mask,
        connectivity=p.connectivity,
        border_margin_px=p.border_margin_px,
    )

    pairings = compute_pairings_conservatively(
        g,
        probe_len=p.probe_len,
        max_cost_accept=p.max_cost_accept,
    )

    fibers = trace_fibers(g, pairings)

    compute_segment_lengths(g)
    compute_fiber_lengths(g, fibers, scale=Scale(um_per_px=p.um_per_px))

    fibers_noborder = filter_fibers_excluding_border(fibers)
    fibers_final = fibers_noborder if p.exclude_border_touching else fibers

    lengths_px = np.array(
        [f.length_px for f in fibers_final if f.length_px is not None and f.length_px > 0],
        dtype=np.float64,
    )

    return {
        "mask": sk.mask,
        "graph": g,
        "fibers": fibers_final,
        "lengths_px": lengths_px,
        "meta": {
            "fibers_total": len(fibers),
            "fibers_final": len(fibers_final),
        },
    }
