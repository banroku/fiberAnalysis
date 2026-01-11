# src/fiberlen/graph_post.py
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from .types import CompressedGraph, NodeKind


class _DSU:
    def __init__(self, items: List[int]) -> None:
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def merge_close_junctions(
    g: CompressedGraph,
    *,
    radius_px: int,
) -> Dict[str, int]:
    """
    Merge junction nodes whose representative coordinates are within radius_px.

    This is a post-process on CompressedGraph:
      - cluster junction nodes by distance between node.coord
      - pick one representative node_id per cluster (smallest id)
      - rewire segments start_node/end_node to representative
      - drop non-representative junction nodes
      - recompute degree for all nodes; keep NodeKind of non-junction nodes as-is
        (junction reps remain junction)

    Returns stats for UI/diagnostics.
    """
    r = int(radius_px)
    if r <= 0:
        return {"junctions_before": sum(1 for n in g.nodes.values() if n.kind == NodeKind.JUNCTION),
                "junctions_after": sum(1 for n in g.nodes.values() if n.kind == NodeKind.JUNCTION),
                "merged_clusters": 0,
                "merged_nodes": 0}

    # Collect junction ids and their coords
    j_ids = [nid for nid, n in g.nodes.items() if n.kind == NodeKind.JUNCTION]
    junctions_before = len(j_ids)
    if junctions_before <= 1:
        return {"junctions_before": junctions_before,
                "junctions_after": junctions_before,
                "merged_clusters": 0,
                "merged_nodes": 0}

    coords = np.array([g.nodes[nid].coord for nid in j_ids], dtype=np.int32)  # (r,c)
    # Naive O(n^2) is fine for typical junction counts (~50-500). If this grows, we can switch to spatial hashing.
    dsu = _DSU(j_ids)

    r2 = r * r
    for i in range(len(j_ids)):
        ri, ci = coords[i]
        for k in range(i + 1, len(j_ids)):
            rk, ck = coords[k]
            dr = int(rk - ri)
            dc = int(ck - ci)
            if dr * dr + dc * dc <= r2:
                dsu.union(j_ids[i], j_ids[k])

    # Build clusters
    clusters: Dict[int, List[int]] = {}
    for nid in j_ids:
        root = dsu.find(nid)
        clusters.setdefault(root, []).append(nid)

    # If every cluster is singleton, nothing to do
    merged_clusters = sum(1 for v in clusters.values() if len(v) >= 2)
    if merged_clusters == 0:
        return {"junctions_before": junctions_before,
                "junctions_after": junctions_before,
                "merged_clusters": 0,
                "merged_nodes": 0}

    # Choose representative per cluster = smallest node_id
    rep_of: Dict[int, int] = {}
    rep_nodes: Dict[int, List[int]] = {}
    for members in clusters.values():
        rep = min(members)
        rep_nodes[rep] = members
        for m in members:
            rep_of[m] = rep

    merged_nodes = sum(len(m) - 1 for m in rep_nodes.values() if len(m) >= 2)

    # Rewire segments
    for seg in g.segments.values():
        if seg.start_node in rep_of:
            seg.start_node = rep_of[seg.start_node]
        if seg.end_node in rep_of:
            seg.end_node = rep_of[seg.end_node]

    # Optionally update representative junction coord to cluster centroid (rounded)
    # This is purely for display/diagnostics; pairing uses segment geometry, not node coord.
    for rep, members in rep_nodes.items():
        if len(members) <= 1:
            continue
        pts = np.array([g.nodes[m].coord for m in members], dtype=np.float64)
        mean_r, mean_c = pts.mean(axis=0)
        new_coord = (int(round(mean_r)), int(round(mean_c)))
        n = g.nodes[rep]
        g.nodes[rep] = replace(n, coord=new_coord)  # dataclass replace; if Node isn't dataclass, adjust accordingly

    # Drop non-representative junction nodes
    keep = set(g.nodes.keys())
    for rep, members in rep_nodes.items():
        for m in members:
            if m != rep:
                keep.discard(m)
    g.nodes = {nid: n for nid, n in g.nodes.items() if nid in keep}

    # Recompute degree for all nodes based on segments
    deg: Dict[int, int] = {nid: 0 for nid in g.nodes.keys()}
    for seg in g.segments.values():
        if seg.start_node in deg:
            deg[seg.start_node] += 1
        if seg.end_node in deg:
            deg[seg.end_node] += 1

    # Apply degree and (optionally) adjust kind for non-junction nodes
    # We keep junction reps as junction; endpoints/others can be updated conservatively.
    for nid, n in list(g.nodes.items()):
        new_degree = int(deg.get(nid, 0))
        if n.kind == NodeKind.JUNCTION:
            g.nodes[nid] = replace(n, degree=new_degree, kind=NodeKind.JUNCTION)
        else:
            # Endpoint if degree==1, otherwise keep kind (or promote to junction if degree>=3)
            if new_degree == 1:
                g.nodes[nid] = replace(n, degree=new_degree, kind=NodeKind.ENDPOINT)
            elif new_degree >= 3:
                g.nodes[nid] = replace(n, degree=new_degree, kind=NodeKind.JUNCTION)
            else:
                g.nodes[nid] = replace(n, degree=new_degree)

    junctions_after = sum(1 for n in g.nodes.values() if n.kind == NodeKind.JUNCTION)

    return {
        "junctions_before": junctions_before,
        "junctions_after": junctions_after,
        "merged_clusters": merged_clusters,
        "merged_nodes": merged_nodes,
    }
