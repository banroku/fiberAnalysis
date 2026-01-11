# scripts/inspect_tracing.py
from pathlib import Path

import csv

from fiberlen.io import read_skeleton_tif
from fiberlen.skeleton_graph import build_compressed_graph, summarize_graph
from fiberlen.pairing import compute_pairings_conservatively
from fiberlen.tracing import trace_fibers
from fiberlen.metrics import compute_segment_lengths, compute_fiber_lengths, filter_fibers_excluding_border
from fiberlen.types import Scale
from fiberlen.config import CFG

def main():
    raw_dir = Path("data/raw")
    out_dir = Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff")))
    if not candidates:
        raise FileNotFoundError(f"No tif found under: {raw_dir.resolve()}")
    img_path = candidates[0]
    print("Using:", img_path)

    sk = read_skeleton_tif(img_path, foreground=CFG.io_foreground, pad=CFG.io_pad)
    g, deg_img, node_labels = build_compressed_graph(sk.mask, connectivity=CFG.connectivity, border_margin_px=CFG.border_margin_px)
    print("Graph summary:", summarize_graph(g))

    pairings = compute_pairings_conservatively(g, probe_len=CFG.pairing_probe_len, max_cost_accept=CFG.pairing_max_cost_accept)
    fibers = trace_fibers(g, pairings)

    compute_segment_lengths(g)

    # スケールは一旦 1.0 µm/px としておく（後で実測値に置き換えてください）
    scale = Scale(um_per_px=CFG.um_per_px)
    compute_fiber_lengths(g, fibers, scale=scale)

    fibers_noborder = fibers
    if CFG.exclude_border_touching:
        fibers_noborder = filter_fibers_excluding_border(fibers)

    print("fibers total:", len(fibers))
    print("fibers noborder:", len(fibers_noborder))
    print("fibers excluding border:", CFG.exclude_border_touching)

    out_csv_all = out_dir / "fiber_lengths_all.csv"
    out_csv_noborder = out_dir / "fiber_lengths_excluding_border.csv"

    def write_csv(path: Path, rows):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fiber_id", "n_segments", "touches_border", "has_junction", "length_px", "length_um"])
            for r in rows:
                w.writerow([r.fiber_id, len(r.seg_ids), int(r.touches_border), int(r.has_junction), r.length_px, r.length_um])

    write_csv(out_csv_all, fibers)
    write_csv(out_csv_noborder, fibers_noborder)

    print("Saved:", out_csv_all)
    print("Saved:", out_csv_noborder)


if __name__ == "__main__":
    main()
