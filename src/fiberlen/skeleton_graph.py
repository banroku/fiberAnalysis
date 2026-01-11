# src/fiberlen/skeleton_graph.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .types import CompressedGraph, Node, NodeKind, Pixel, Segment


# ----------------------------
# Neighborhood utilities
# ----------------------------

_OFFSETS_8 = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]

_OFFSETS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _offsets(connectivity: int) -> List[Tuple[int, int]]:
    if connectivity == 8:
        return _OFFSETS_8
    if connectivity == 4:
        return _OFFSETS_4
    raise ValueError("connectivity must be 4 or 8")


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors(p: Pixel, h: int, w: int, connectivity: int) -> List[Pixel]:
    r, c = p
    nb = []
    for dr, dc in _offsets(connectivity):
        rr, cc = r + dr, c + dc
        if _in_bounds(rr, cc, h, w):
            nb.append((rr, cc))
    return nb


def compute_degree(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    Degree of each pixel in the skeleton pixel-graph.
    degree[r,c] = number of neighboring True pixels (excluding itself).
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)

    h, w = mask.shape
    deg = np.zeros((h, w), dtype=np.uint8)

    # Sum shifted masks
    for dr, dc in _offsets(connectivity):
        src_r0 = max(0, -dr)
        src_r1 = min(h, h - dr)
        src_c0 = max(0, -dc)
        src_c1 = min(w, w - dc)

        dst_r0 = max(0, dr)
        dst_r1 = min(h, h + dr)
        dst_c0 = max(0, dc)
        dst_c1 = min(w, w + dc)

        deg[dst_r0:dst_r1, dst_c0:dst_c1] += mask[src_r0:src_r1, src_c0:src_c1].astype(np.uint8)

    # Outside skeleton: degree is irrelevant but keep 0
    deg[~mask] = 0
    return deg


# ----------------------------
# Connected components on a boolean mask
# ----------------------------

def _connected_components(mask: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, List[List[Pixel]]]:
    """
    Label connected components in a boolean mask.
    Returns:
      labels: int32 array, -1 for background, 0..n-1 for components
      comps: list of list of pixels (row,col) per component
    """
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=np.int32)
    comps: List[List[Pixel]] = []
    off = _offsets(connectivity)

    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue

            q = deque([(r, c)])
            labels[r, c] = comp_id
            comp_pixels: List[Pixel] = [(r, c)]

            while q:
                rr, cc = q.popleft()
                for dr, dc in off:
                    r2, c2 = rr + dr, cc + dc
                    if _in_bounds(r2, c2, h, w) and mask[r2, c2] and labels[r2, c2] == -1:
                        labels[r2, c2] = comp_id
                        q.append((r2, c2))
                        comp_pixels.append((r2, c2))

            comps.append(comp_pixels)
            comp_id += 1

    return labels, comps


def _representative_pixel(pixels: List[Pixel]) -> Pixel:
    """
    Representative pixel for a component.
    Use integer-rounded centroid for stability.
    """
    rs = np.array([p[0] for p in pixels], dtype=np.float64)
    cs = np.array([p[1] for p in pixels], dtype=np.float64)
    r = int(np.round(rs.mean()))
    c = int(np.round(cs.mean()))
    # centroid might not land exactly on a pixel in the component; fallback
    if (r, c) in set(pixels):
        return (r, c)
    return pixels[0]


# ----------------------------
# Node (endpoint/junction) extraction
# ----------------------------

@dataclass(frozen=True)
class NodeComponent:
    comp_id: int
    pixels: List[Pixel]
    rep: Pixel
    kind: NodeKind
    degree: int


def extract_node_components(
    skel: np.ndarray,
    degree: np.ndarray,
    *,
    connectivity: int = 8
) -> Tuple[np.ndarray, List[NodeComponent]]:
    """
    Node pixels are skeleton pixels with degree != 2.
    We group them into connected components to handle multi-pixel junction blobs.
    Returns:
      node_comp_labels: array with -1 for non-node, else component id for node component
      node_components: list of NodeComponent
    """
    if skel.dtype != bool:
        skel = skel.astype(bool)

    node_pix = skel & (degree != 2)
    labels, comps = _connected_components(node_pix, connectivity=connectivity)

    h, w = skel.shape
    node_components: List[NodeComponent] = []

    for comp_id, pixels in enumerate(comps):
        rep = _representative_pixel(pixels)

        # Determine kind:
        # if component contains any degree==1 pixel -> endpoint (common for single-pixel endpoints)
        # else -> junction (degree 3+ or blob)
        has_deg1 = any(degree[r, c] == 1 for (r, c) in pixels)
        kind = NodeKind.ENDPOINT if has_deg1 and len(pixels) == 1 else NodeKind.JUNCTION

        # Component "degree": count unique skeleton neighbors outside the component
        pix_set = set(pixels)
        outside_neighbors: Set[Pixel] = set()
        for (r, c) in pixels:
            for nb in _neighbors((r, c), h, w, connectivity):
                if skel[nb] and nb not in pix_set:
                    outside_neighbors.add(nb)
        comp_degree = len(outside_neighbors)

        node_components.append(NodeComponent(
            comp_id=comp_id,
            pixels=pixels,
            rep=rep,
            kind=kind,
            degree=comp_degree
        ))

    return labels, node_components


# ----------------------------
# Segment tracing between nodes
# ----------------------------

def _edge_key(a: Pixel, b: Pixel) -> Tuple[Pixel, Pixel]:
    return (a, b) if a <= b else (b, a)


def build_compressed_graph(
    skel: np.ndarray,
    *,
    connectivity: int = 8,
    border_margin_px: int = 1,
) -> Tuple[CompressedGraph, np.ndarray, np.ndarray]:
    """
    Build compressed graph from skeleton mask.
    Returns:
      graph: CompressedGraph
      degree: degree image (uint8)
      node_comp_labels: int32 labels (-1 for non-node)
    """
    if skel.dtype != bool:
        skel = skel.astype(bool)
    h, w = skel.shape
    m = int(border_margin_px)
    if m < 0:
        raise ValueError("border_margin_px must be >=0")

    deg = compute_degree(skel, connectivity=connectivity)
    node_comp_labels, node_comps = extract_node_components(skel, deg, connectivity=connectivity)

    # Map node component id -> node_id, and pixel -> node_id
    g = CompressedGraph()
    comp_to_nodeid: Dict[int, int] = {}
    pixel_to_nodeid: Dict[Pixel, int] = {}

    next_node_id = 1
    for nc in node_comps:
        node_id = next_node_id
        next_node_id += 1
        comp_to_nodeid[nc.comp_id] = node_id

        node = Node(
            node_id=node_id,
            coord=nc.rep,
            kind=nc.kind,
            degree=nc.degree
        )
        g.add_node(node)

        for p in nc.pixels:
            pixel_to_nodeid[p] = node_id

    # Helper: is pixel in some node component?
    def node_id_at(p: Pixel) -> Optional[int]:
        return pixel_to_nodeid.get(p, None)

    # Trace segments starting from each node component boundary neighbor
    visited_edges: Set[Tuple[Pixel, Pixel]] = set()
    next_seg_id = 1

    for nc in node_comps:
        start_node = comp_to_nodeid[nc.comp_id]
        comp_pixels_set = set(nc.pixels)

        # Find "exits": skeleton neighbors outside the node component
        exits: List[Tuple[Pixel, Pixel]] = []  # (from_node_pixel, outside_neighbor)
        for p in nc.pixels:
            for nb in _neighbors(p, h, w, connectivity):
                if skel[nb] and nb not in comp_pixels_set:
                    exits.append((p, nb))

        for from_p, first in exits:
            ek = _edge_key(from_p, first)
            if ek in visited_edges:
                continue

            # Walk along skeleton until reaching another node component
            pixels: List[Pixel] = [from_p, first]
            prev = from_p
            cur = first

            visited_edges.add(ek)

            end_node: Optional[int] = None

            while True:
                nid = node_id_at(cur)
                if nid is not None and nid != start_node:
                    end_node = nid
                    break

                # Determine next step:
                nbs = [nb for nb in _neighbors(cur, h, w, connectivity) if skel[nb] and nb != prev]

                if len(nbs) == 0:
                    # Dead end. If current is inside the start node (unlikely), ignore.
                    # Else treat as endpoint by creating a new ENDPOINT node on this pixel.
                    # In your data this should be rare; but handle safely.
                    if node_id_at(cur) is None:
                        # create endpoint node
                        new_node_id = next_node_id
                        next_node_id += 1
                        node = Node(
                            node_id=new_node_id,
                            coord=cur,
                            kind=NodeKind.ENDPOINT,
                            degree=1
                        )
                        g.add_node(node)
                        pixel_to_nodeid[cur] = new_node_id
                        end_node = new_node_id
                    else:
                        end_node = node_id_at(cur)
                    break

                if len(nbs) > 1:
                    # This indicates we walked into a junction blob not captured as node pixels,
                    # or a topological anomaly.
                    # Choose the next that keeps going "straight" relative to prev->cur.
                    # (Minimal heuristic for robustness.)
                    pr = np.array(prev, dtype=np.float64)
                    cr = np.array(cur, dtype=np.float64)
                    v = cr - pr
                    best = None
                    best_score = -1e18
                    for nb in nbs:
                        nr = np.array(nb, dtype=np.float64)
                        u = nr - cr
                        # maximize cosine similarity
                        denom = (np.linalg.norm(v) * np.linalg.norm(u) + 1e-9)
                        score = float(np.dot(v, u) / denom)
                        if score > best_score:
                            best_score = score
                            best = nb
                    nxt = best  # type: ignore[assignment]
                else:
                    nxt = nbs[0]

                ek2 = _edge_key(cur, nxt)
                if ek2 in visited_edges:
                    # already traversed this direction; stop
                    # If next is a node, we can end there; otherwise stop at current.
                    nid2 = node_id_at(nxt)
                    if nid2 is not None:
                        end_node = nid2
                        pixels.append(nxt)
                    else:
                        end_node = node_id_at(cur)
                    break

                visited_edges.add(ek2)
                pixels.append(nxt)
                prev, cur = cur, nxt

            if end_node is None:
                continue

            # Border touch flag
            if m == 0:
                touches_border = False
            else: 
                touches_border = any(
                        (r < m) or (c < m) or (r >= h - m) or (c >= w - m)
                        for (r, c) in pixels
                )

            seg = Segment(
                seg_id=next_seg_id,
                start_node=start_node,
                end_node=end_node,
                pixels=pixels,
                touches_border=touches_border
            )
            next_seg_id += 1
            g.add_segment(seg)

    return g, deg, node_comp_labels


def summarize_graph(g: CompressedGraph) -> Dict[str, int]:
    endpoints = sum(1 for n in g.nodes.values() if n.kind == NodeKind.ENDPOINT)
    junctions = sum(1 for n in g.nodes.values() if n.kind == NodeKind.JUNCTION)
    return {
        "nodes": len(g.nodes),
        "segments": len(g.segments),
        "endpoints": endpoints,
        "junctions": junctions,
    }
