# scripts/overlay_top_fibers.py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fiberlen.config import CFG
from fiberlen.io import read_skeleton_tif
from fiberlen.skeleton_graph import build_compressed_graph
from fiberlen.pairing import compute_pairings_conservatively
from fiberlen.tracing import trace_fibers
from fiberlen.metrics import (
    compute_segment_lengths,
    compute_fiber_lengths,
    filter_fibers_excluding_border,
)
from fiberlen.types import Scale


def _plot_polyline_rc(pixels, *, linewidth: float) -> None:
    # pixels: list[(r,c)] -> plot wants x=c, y=r
    rr = [p[0] for p in pixels]
    cc = [p[1] for p in pixels]
    plt.plot(cc, rr, linewidth=linewidth)  # matplotlib default color cycle


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    out_dir = root / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff")))
    if not candidates:
        raise FileNotFoundError(f"No tif found under: {raw_dir.resolve()}")

    img_path = candidates[0]
    print("Using:", img_path)

    sk = read_skeleton_tif(img_path, foreground=CFG.io_foreground, pad=CFG.io_pad)

    g, _, _ = build_compressed_graph(
        sk.mask,
        connectivity=CFG.connectivity,
        border_margin_px=CFG.border_margin_px,
    )

    pairings = compute_pairings_conservatively(
        g,
        probe_len=CFG.pairing_probe_len,
        max_cost_accept=CFG.pairing_max_cost_accept,
    )

    fibers = trace_fibers(g, pairings)

    compute_segment_lengths(g)
    compute_fiber_lengths(g, fibers, scale=Scale(um_per_px=CFG.um_per_px))

    fibers_noborder = filter_fibers_excluding_border(fibers)
    fibers_final = fibers_noborder if CFG.exclude_border_touching else fibers

    top_n = int(CFG.overlay_top_n)
    if top_n <= 0:
        raise ValueError("CFG.overlay_top_n must be >= 1")

    fibers_sorted = sorted(
        fibers_final,
        key=lambda f: (f.length_um if f.length_um is not None else -1.0),
        reverse=True,
    )
    top = fibers_sorted[: min(top_n, len(fibers_sorted))]

    print("fibers total:", len(fibers))
    print("fibers final:", len(fibers_final))
    print("overlay_top_n:", top_n)
    if top:
        print("top length_um:", [round(f.length_um or 0.0, 2) for f in top[:5]])

    base = sk.mask.astype(np.uint8)

    fig = plt.figure()
    plt.imshow(base, cmap="gray", interpolation="nearest")

    for f in top:
        for sid in f.seg_ids:
            _plot_polyline_rc(g.segments[sid].pixels, linewidth=float(CFG.overlay_linewidth))

    plt.title(
        f"Top {len(top)} fibers "
        f"(exclude_border={CFG.exclude_border_touching}, margin={CFG.border_margin_px}px)"
    )

    out_path = out_dir / str(CFG.overlay_top_fibers_filename)
    plt.savefig(out_path, dpi=int(CFG.overlay_dpi), bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
