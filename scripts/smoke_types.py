# scripts/smoke_types.py
from fiberlen.types import Node, NodeKind, Segment, CompressedGraph, PipelineParams, Scale

def main():
    # 1) instantiate
    n1 = Node(node_id=1, coord=(10, 10), kind=NodeKind.ENDPOINT, degree=1)
    n2 = Node(node_id=2, coord=(10, 20), kind=NodeKind.JUNCTION, degree=3)
    n3 = Node(node_id=3, coord=(10, 30), kind=NodeKind.ENDPOINT, degree=1)

    seg1 = Segment(seg_id=100, start_node=1, end_node=2, pixels=[(10,10),(10,11),(10,12),(10,20)])
    seg2 = Segment(seg_id=101, start_node=2, end_node=3, pixels=[(10,20),(10,21),(10,22),(10,30)])

    # 2) graph ops
    g = CompressedGraph()
    g.add_node(n1)
    g.add_node(n2)
    g.add_node(n3)
    g.add_segment(seg1)
    g.add_segment(seg2)

    # 3) sanity checks
    assert g.other_node(100, 1) == 2
    assert g.other_node(100, 2) == 1
    assert set(g.incident_segments(2)) == {100, 101}

    params = PipelineParams()
    scale = Scale(um_per_px=2.0)

    print("OK: types.py basic smoke test passed")
    print("incident at node 2:", g.incident_segments(2))
    print("params:", params)
    print("scale:", scale)

if __name__ == "__main__":
    main()

