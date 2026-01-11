# src/fiberlen/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set


Pixel = Tuple[int, int]          # (row, col)
Vector2 = Tuple[float, float]    # (dy, dx)


class NodeKind(str, Enum):
    ENDPOINT = "endpoint"
    JUNCTION = "junction"


@dataclass(frozen=True)
class Node:
    """
    Node in the compressed skeleton graph.

    - coord: representative pixel coordinate (row, col)
    - kind: ENDPOINT or JUNCTION
    - degree: degree in the original pixel graph (8-neighborhood)
    """
    node_id: int
    coord: Pixel
    kind: NodeKind
    degree: int


@dataclass
class Segment:
    """
    Segment is a polyline between two Nodes following the skeleton pixels.

    - pixels: ordered pixel coordinates from start_node -> end_node, inclusive
    - length_px: computed later (weighted 8-neighborhood length)
    - note: we keep pixels for direction estimation and visualization
    """
    seg_id: int
    start_node: int
    end_node: int
    pixels: List[Pixel]

    length_px: Optional[float] = None

    # Convenience flags for later QC
    touches_border: bool = False
    is_pruned: bool = False


@dataclass
class CompressedGraph:
    """
    A sparse graph:
      nodes: Node dictionary
      segments: Segment dictionary
      adjacency: node_id -> set of seg_id
    """
    nodes: Dict[int, Node] = field(default_factory=dict)
    segments: Dict[int, Segment] = field(default_factory=dict)
    adjacency: Dict[int, Set[int]] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self.adjacency.setdefault(node.node_id, set())

    def add_segment(self, seg: Segment) -> None:
        self.segments[seg.seg_id] = seg
        self.adjacency.setdefault(seg.start_node, set()).add(seg.seg_id)
        self.adjacency.setdefault(seg.end_node, set()).add(seg.seg_id)

    def other_node(self, seg_id: int, node_id: int) -> int:
        seg = self.segments[seg_id]
        if seg.start_node == node_id:
            return seg.end_node
        if seg.end_node == node_id:
            return seg.start_node
        raise ValueError(f"Segment {seg_id} not incident to node {node_id}.")

    def incident_segments(self, node_id: int) -> List[int]:
        return sorted(self.adjacency.get(node_id, set()))


@dataclass(frozen=True)
class Pairing:
    """
    A local decision at one junction:
      pairs: list of (seg_id_a, seg_id_b) that should be connected as one fiber through the junction.
      leftovers: segments that terminate at this junction (e.g., T-junction branch)
      score: optional confidence score (higher = better)
    """
    junction_node: int
    pairs: List[Tuple[int, int]]
    leftovers: List[int]
    score: Optional[float] = None


@dataclass
class FiberPath:
    """
    One reconstructed fiber after resolving junction pairings.

    - seg_ids: ordered list of segments that form the fiber
    - pixels: optional merged polyline for visualization / length integration
    - length_px / length_um: filled after measurement
    """
    fiber_id: int
    seg_ids: List[int]

    pixels: Optional[List[Pixel]] = None
    length_px: Optional[float] = None
    length_um: Optional[float] = None

    # QC flags / metadata
    touches_border: bool = False
    has_junction: bool = False
    notes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineParams:
    """
    Parameters for the pipeline (start with minimal set; expand as needed).
    """
    # Graph building
    connectivity: int = 8  # 4 or 8, typically 8 for skeletons

    # Spur pruning (in pixels along skeleton)
    prune_spur_max_len: int = 10

    # Junction direction estimation: how far from junction to look along a segment
    direction_probe_len: int = 12

    # Border handling
    exclude_border_touching: bool = True


@dataclass(frozen=True)
class Scale:
    """
    Pixel-to-micrometer conversion.
    """
    um_per_px: float

