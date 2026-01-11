# scripts/inspect_pairing.py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fiberlen.io import read_skeleton_tif
from fiberlen.skeleton_graph import build_compressed_graph, summarize_graph
from fiberlen.pairing import compute_pairings_conservatively, segment_direction_away_from_junction
from fiberlen.types import NodeKind
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
    g, deg_img, node_labels = build_compressed_graph(sk.mask, connectivity=CFG.connectivity)

    print("Graph summary:", summarize_graph(g))

    pairings = compute_pairings_conservatively(g, probe_len=CFG.pairing_probe_len, max_cost_accept=CFG.pairing_max_cost_accept)

    # Summary stats
    junction_ids = [nid for nid, n in g.nodes.items() if n.kind == NodeKind.JUNCTION]
    n_j = len(junction_ids)
    n_with_pairs = sum(1 for p in pairings.values() if len(p.pairs) > 0)
    n_cut = n_j - n_with_pairs
    print(f"junctions total={n_j}, paired={n_with_pairs}, cut={n_cut}")

    # Make overlay image
    base = sk.mask.astype(np.uint8)
    ep = np.array([n.coord for n in g.nodes.values() if n.kind == NodeKind.ENDPOINT], dtype=int)
    jn = np.array([n.coord for n in g.nodes.values() if n.kind == NodeKind.JUNCTION], dtype=int)

    fig = plt.figure()
    plt.imshow(base, cmap="gray", interpolation="nearest")

    if len(ep) > 0:
        plt.scatter(ep[:, 1], ep[:, 0], s=10, marker="o", label="endpoints")
    if len(jn) > 0:
        plt.scatter(jn[:, 1], jn[:, 0], s=16, marker="x", label="junctions")

    # Draw accepted pairings as short green-ish lines (matplotlib default cycle will choose a color)
    # We avoid explicit color specification; use default by not passing color and letting plot cycle.
    drawn = 0
    for j_id, p in pairings.items():
        if not p.pairs:
            continue
        j_rc = g.nodes[j_id].coord
        for (sida, sidb) in p.pairs:
            ua, p0a, p1a = segment_direction_away_from_junction(g, j_id, sida, probe_len=12)
            ub, p0b, p1b = segment_direction_away_from_junction(g, j_id, sidb, probe_len=12)

            # Draw line between probe points (p1a and p1b) to indicate "straight-through" decision
            plt.plot([p1a[1], p1b[1]], [p1a[0], p1b[0]], linewidth=1.5)
            drawn += 1

    plt.title(f"pairings accepted lines={drawn}")
    plt.legend(loc="upper right")

    out_path = out_dir / "overlay_pairing.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_path)

    # Print a few example junction decisions
    shown = 0
    for j_id, p in pairings.items():
        if shown >= 8:
            break
        if p.pairs:
            print(f"Junction {j_id}: pairs={p.pairs} leftovers={p.leftovers} score={p.score}")
            shown += 1


if __name__ == "__main__":
    main()
