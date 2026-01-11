# scripts/inspect_graph.py
from pathlib import Path

from fiberlen.io import read_skeleton_tif
from fiberlen.skeleton_graph import build_compressed_graph, summarize_graph
from fiberlen.config import CFG

def main():
    raw_dir = Path("data/raw")
    candidates = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff")))
    if not candidates:
        raise FileNotFoundError(f"No tif found under: {raw_dir.resolve()}")

    img_path = candidates[0]
    print("Using:", img_path)

    sk = read_skeleton_tif(img_path, foreground=CFG.io_foreground, pad=CFG.io_pad)
    g, deg, node_labels = build_compressed_graph(sk.mask, connectivity=CFG.connectivity)

    summary = summarize_graph(g)
    print("Graph summary:", summary)

    # A few sanity prints
    # show first 5 nodes and segments
    for i, n in list(g.nodes.items())[:5]:
        print("Node", i, n)
    for i, s in list(g.segments.items())[:5]:
        print("Seg", i, "start", s.start_node, "end", s.end_node, "n_pix", len(s.pixels), "border", s.touches_border)

if __name__ == "__main__":
    main()
