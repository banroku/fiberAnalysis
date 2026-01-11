# scripts/overlay_graph.py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fiberlen.io import read_skeleton_tif
from fiberlen.skeleton_graph import build_compressed_graph
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
    g, deg, node_labels = build_compressed_graph(sk.mask, connectivity=CFG.connectivity)

    # Background image: skeleton as 0/1
    base = sk.mask.astype(np.uint8)

    # Collect node coords
    ep = np.array([n.coord for n in g.nodes.values() if n.kind == NodeKind.ENDPOINT], dtype=int)
    jn = np.array([n.coord for n in g.nodes.values() if n.kind == NodeKind.JUNCTION], dtype=int)

    fig = plt.figure()
    plt.imshow(base, cmap="gray", interpolation="nearest")
    if len(ep) > 0:
        plt.scatter(ep[:, 1], ep[:, 0], s=12, marker="o", label="endpoints")
    if len(jn) > 0:
        plt.scatter(jn[:, 1], jn[:, 0], s=18, marker="x", label="junctions")
    plt.legend(loc="upper right")
    plt.title(f"nodes={len(g.nodes)} segs={len(g.segments)}")
    out_path = out_dir / "overlay_nodes.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
