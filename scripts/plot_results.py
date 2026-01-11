
# scripts/plot_results.py
from __future__ import annotations

from pathlib import Path
import csv
from collections import defaultdict

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


def _run_pipeline_lengths_px() -> list[float]:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"

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

    # length_px is the main quantity for plotting
    lengths_px = [float(f.length_px) for f in fibers_final if f.length_px is not None and f.length_px > 0]
    print("fibers total:", len(fibers))
    print("fibers final:", len(fibers_final))
    print("lengths used:", len(lengths_px))
    return lengths_px


def _save_hist_table(root: Path, bin_edges: np.ndarray, counts: np.ndarray, weightsum: np.ndarray) -> None:
    out_path = root / "data" / "output" / str(CFG.plot_csv_hist_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin_left_px", "bin_right_px", "count", "sum_length_px"])
        for i in range(len(counts)):
            w.writerow([float(bin_edges[i]), float(bin_edges[i + 1]), int(counts[i]), float(weightsum[i])])

    print("Saved:", out_path)


def _save_discrete_table(root: Path, xs: list[int], counts: list[int], weightsum: list[float]) -> None:
    out_path = root / "data" / "output" / str(CFG.plot_csv_discrete_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["length_px_rounded", "count", "sum_length_px"])
        for x, c, s in zip(xs, counts, weightsum):
            w.writerow([int(x), int(c), float(s)])

    print("Saved:", out_path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    lengths = _run_pipeline_lengths_px()
    if not lengths:
        raise RuntimeError("No valid fiber lengths. Check pipeline / filtering.")

    lengths_arr = np.array(lengths, dtype=np.float64)

    # -----------------------------
    # 1) Histogram: count and weighted (sum length)
    # -----------------------------
    bins = int(CFG.plot_hist_bins)
    if bins <= 0:
        raise ValueError("CFG.plot_hist_bins must be >= 1")

    # Use [min, max] from data
    lo = float(lengths_arr.min())
    hi = float(lengths_arr.max())
    if lo == hi:
        hi = lo + 1e-6

    counts, edges = np.histogram(lengths_arr, bins=bins, range=(lo, hi))
    weightsum, _ = np.histogram(lengths_arr, bins=bins, range=(lo, hi), weights=lengths_arr)

    _save_hist_table(root, edges, counts, weightsum)

    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0]) if len(edges) > 2 else 1.0

    fig = plt.figure()
    ax1 = plt.gca()
    ax1.bar(centers, counts, width=bin_width, align="center")
    ax1.set_xlabel("fiber length (px)")
    ax1.set_ylabel("count")

    ax2 = ax1.twinx()
    ax2.plot(centers, weightsum, marker="o", linewidth=1.5)
    ax2.set_ylabel("sum of length in bin (px)")

    plt.title("Histogram: count and length-weighted sum")
    out_hist = out_dir / str(CFG.plot_hist_filename)
    plt.savefig(out_hist, dpi=int(CFG.plot_dpi), bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_hist)

    # -----------------------------
    # 2) Discrete graph: length(px) vs count and weighted (sum length)
    # -----------------------------
    step = int(CFG.plot_discrete_round_px)
    if step <= 0:
        raise ValueError("CFG.plot_discrete_round_px must be >= 1")

    # Round lengths to nearest multiple of step
    rounded = (np.round(lengths_arr / step) * step).astype(int)

    count_map: dict[int, int] = defaultdict(int)
    sum_map: dict[int, float] = defaultdict(float)

    for L_raw, L_round in zip(lengths_arr, rounded):
        count_map[int(L_round)] += 1
        sum_map[int(L_round)] += float(L_raw)

    xs = sorted(count_map.keys())
    ys_count = [count_map[x] for x in xs]
    ys_sum = [sum_map[x] for x in xs]

    _save_discrete_table(root, xs, ys_count, ys_sum)

    fig2 = plt.figure()
    ax1 = plt.gca()
    ax1.plot(xs, ys_count, marker="o", linewidth=1.5)
    ax1.set_xlabel(f"fiber length (px) rounded to {step}px")
    ax1.set_ylabel("count")

    ax2 = ax1.twinx()
    ax2.plot(xs, ys_sum, marker="o", linewidth=1.5)
    ax2.set_ylabel("sum of length at x (px)")

    plt.title("Discrete: count and length-weighted sum")
    out_disc = out_dir / str(CFG.plot_discrete_filename)
    plt.savefig(out_disc, dpi=int(CFG.plot_dpi), bbox_inches="tight")
    plt.close(fig2)
    print("Saved:", out_disc)

    print("Done.")


if __name__ == "__main__":
    main()
