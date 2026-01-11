# src/fiberlen/pairing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import CompressedGraph, NodeKind, Pairing, Pixel


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0], dtype=np.float64)
    return v / n


def _as_vec(p: Pixel) -> np.ndarray:
    return np.array([float(p[0]), float(p[1])], dtype=np.float64)  # (row, col)


def _nearest_endpoint_index(pixels: List[Pixel], junction_rc: np.ndarray) -> int:
    """
    Return 0 or -1 (start/end) whichever endpoint is closer to the junction representative coord.
    """
    a = _as_vec(pixels[0])
    b = _as_vec(pixels[-1])
    da = float(np.linalg.norm(a - junction_rc))
    db = float(np.linalg.norm(b - junction_rc))
    return 0 if da <= db else -1


def segment_direction_away_from_junction(
    g: CompressedGraph,
    junction_node: int,
    seg_id: int,
    *,
    probe_len: int = 12,
) -> Tuple[np.ndarray, Pixel, Pixel]:
    """
    Estimate direction of a segment at a junction node, pointing outward (away from the junction).

    Returns:
      unit direction vector in (row, col) coordinates
      p0: point near junction
      p1: point probe_len pixels away along the segment
    """
    seg = g.segments[seg_id]
    j = g.nodes[junction_node]
    jrc = _as_vec(j.coord)

    px = seg.pixels
    if len(px) < 2:
        return np.array([0.0, 0.0], dtype=np.float64), px[0], px[0]

    end_idx = _nearest_endpoint_index(px, jrc)

    if end_idx == 0:
        p0 = px[0]
        p1 = px[min(probe_len, len(px) - 1)]
        v = _as_vec(p1) - _as_vec(p0)
        return _unit(v), p0, p1
    else:
        p0 = px[-1]
        p1 = px[max(0, len(px) - 1 - probe_len)]
        v = _as_vec(p1) - _as_vec(p0)  # points outward from p0 toward p1 along the list
        return _unit(v), p0, p1


def _straight_through_cost(u: np.ndarray, v: np.ndarray) -> float:
    """
    Cost for pairing two outward unit vectors at a junction.
    Straight-through means they point in opposite directions => dot ~= -1.
    We define cost in [0, 2] roughly: cost = 1 + dot.
      dot=-1 -> cost=0 (best)
      dot= 0 -> cost=1
      dot=+1 -> cost=2 (worst)
    """
    d = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return 1.0 + d


def _best_pair_for_three(cost_mat: np.ndarray) -> Tuple[Tuple[int, int], int, float]:
    """
    For 3 items, return best pair (i,j), leftover k, and best cost.
    """
    best = (0, 1)
    best_cost = float("inf")
    best_left = 2
    idx = [0, 1, 2]
    for i in range(3):
        for j in range(i + 1, 3):
            k = [t for t in idx if t not in (i, j)][0]
            c = float(cost_mat[i, j])
            if c < best_cost:
                best_cost = c
                best = (i, j)
                best_left = k
    return best, best_left, best_cost


def _best_matching_for_four(cost_mat: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
    """
    For 4 items, brute force three possible perfect matchings.
    Returns list of pairs and total cost.
    """
    matchings = [
        ([(0, 1), (2, 3)], float(cost_mat[0, 1] + cost_mat[2, 3])),
        ([(0, 2), (1, 3)], float(cost_mat[0, 2] + cost_mat[1, 3])),
        ([(0, 3), (1, 2)], float(cost_mat[0, 3] + cost_mat[1, 2])),
    ]
    matchings.sort(key=lambda x: x[1])
    return matchings[0][0], matchings[0][1]


def pair_junction_conservatively(
    g: CompressedGraph,
    junction_node: int,
    *,
    probe_len: int = 12,
    max_cost_accept: float = 0.35,
) -> Pairing:
    """
    Conservative pairing for one junction:
      - Handle only 3-way (T) and 4-way (X) junctions
      - Pair only if best pair cost is small enough (near straight-through)
      - Otherwise leave all segments as leftovers (cut).

    max_cost_accept:
      straight-through best is 0. Typical good matches: 0..0.2
      0.35 is still fairly strict.
    """
    inc = g.incident_segments(junction_node)
    deg = len(inc)

    if deg < 3:
        return Pairing(junction_node=junction_node, pairs=[], leftovers=inc, score=0.0)

    if deg > 4:
        return Pairing(junction_node=junction_node, pairs=[], leftovers=inc, score=0.0)

    dirs: List[np.ndarray] = []
    for sid in inc:
        u, _, _ = segment_direction_away_from_junction(g, junction_node, sid, probe_len=probe_len)
        dirs.append(u)

    # Build symmetric cost matrix
    cost = np.zeros((deg, deg), dtype=np.float64)
    for i in range(deg):
        for j in range(deg):
            if i == j:
                cost[i, j] = 0.0
            else:
                cost[i, j] = _straight_through_cost(dirs[i], dirs[j])

    if deg == 3:
        (i, j), k, best_cost = _best_pair_for_three(cost)
        if best_cost <= max_cost_accept:
            pairs = [(inc[i], inc[j])]
            leftovers = [inc[k]]
            score = float(max(0.0, 1.0 - (best_cost / max_cost_accept)))
            return Pairing(junction_node=junction_node, pairs=pairs, leftovers=leftovers, score=score)
        return Pairing(junction_node=junction_node, pairs=[], leftovers=inc, score=0.0)

    # deg == 4
    pairs_idx, tot = _best_matching_for_four(cost)
    best_pair_costs = [float(cost[i, j]) for (i, j) in pairs_idx]
    worst = max(best_pair_costs) if best_pair_costs else float("inf")

    if worst <= max_cost_accept:
        pairs = [(inc[i], inc[j]) for (i, j) in pairs_idx]
        leftovers: List[int] = []
        score = float(max(0.0, 1.0 - (worst / max_cost_accept)))
        return Pairing(junction_node=junction_node, pairs=pairs, leftovers=leftovers, score=score)

    return Pairing(junction_node=junction_node, pairs=[], leftovers=inc, score=0.0)


def compute_pairings_conservatively(
    g: CompressedGraph,
    *,
    probe_len: int = 12,
    max_cost_accept: float = 0.35,
) -> Dict[int, Pairing]:
    """
    Compute pairing decisions for all junction nodes in the graph.
    Returns: {junction_node_id: Pairing}
    """
    out: Dict[int, Pairing] = {}
    for nid, node in g.nodes.items():
        if node.kind != NodeKind.JUNCTION:
            continue
        out[nid] = pair_junction_conservatively(
            g,
            nid,
            probe_len=probe_len,
            max_cost_accept=max_cost_accept,
        )
    return out
