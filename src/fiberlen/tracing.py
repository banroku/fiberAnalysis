# src/fiberlen/tracing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .types import CompressedGraph, FiberPath, NodeKind, Pairing


def build_pair_map(pairings: Dict[int, Pairing]) -> Dict[Tuple[int, int], int]:
    """
    Build lookup: (junction_node_id, seg_id) -> paired_seg_id
    Only for junctions where a pairing decision exists.
    """
    pair_map: Dict[Tuple[int, int], int] = {}
    for j_id, p in pairings.items():
        for a, b in p.pairs:
            pair_map[(j_id, a)] = b
            pair_map[(j_id, b)] = a
    return pair_map


def trace_fibers(
    g: CompressedGraph,
    pairings: Dict[int, Pairing],
) -> List[FiberPath]:
    """
    Trace fibers by walking segments and applying conservative junction pairings.

    Rules:
      - At endpoint nodes: stop
      - At junction nodes: if current segment has a paired segment at that junction, continue through it
        otherwise stop (cut)
      - Each segment belongs to at most one FiberPath (visited segments are skipped)

    Returns:
      list of FiberPath with ordered seg_ids
    """
    pair_map = build_pair_map(pairings)

    print("=== DEBUG: pair_map preview ===")
    for k, v in list(pair_map.items())[:10]:
        print("pair_map", k, "->", v)
    print("pair_map size:", len(pair_map))
    print("================================")

    visited: Set[int] = set()
    fibers: List[FiberPath] = []
    next_fiber_id = 1

    def next_segment_at_junction(junction_id: int, seg_id: int) -> Optional[int]:
        return pair_map.get((junction_id, seg_id), None)

    def choose_start_node(seg_id: int) -> int:
        seg = g.segments[seg_id]
        a = g.nodes[seg.start_node]
        b = g.nodes[seg.end_node]
        if a.kind == NodeKind.ENDPOINT and b.kind != NodeKind.ENDPOINT:
            return seg.start_node
        if b.kind == NodeKind.ENDPOINT and a.kind != NodeKind.ENDPOINT:
            return seg.end_node
        return seg.start_node

    for seg_id in sorted(g.segments.keys()):
        if seg_id in visited:
            continue

        path_seg_ids: List[int] = []
        touches_border = False
        has_junction = False

        cur_seg_id = seg_id
        cur_node = choose_start_node(cur_seg_id)

        local_seen: Set[int] = set()

        while True:
            if cur_seg_id in local_seen:
                break
            local_seen.add(cur_seg_id)

            seg = g.segments[cur_seg_id]
            visited.add(cur_seg_id)
            path_seg_ids.append(cur_seg_id)
            touches_border = touches_border or bool(seg.touches_border)

            other = g.other_node(cur_seg_id, cur_node)
            other_node = g.nodes[other]

            if other_node.kind == NodeKind.ENDPOINT:
                break

            if other_node.kind == NodeKind.JUNCTION:
                has_junction = True

                key = (other, cur_seg_id)
                print(f"[TRACE] junction={other} incoming_seg={cur_seg_id} key_in_map={key in pair_map}")

                nxt = next_segment_at_junction(other, cur_seg_id)

                print(f"[TRACE] next_seg={nxt}")

                if nxt is None:
                    break
                cur_node = other
                cur_seg_id = nxt
                continue

            break

        fibers.append(
            FiberPath(
                fiber_id=next_fiber_id,
                seg_ids=path_seg_ids,
                touches_border=touches_border,
                has_junction=has_junction,
            )
        )
        next_fiber_id += 1

    return fibers
