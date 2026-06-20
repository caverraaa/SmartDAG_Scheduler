from src.core.compute_node import NodeType
from src.env.cluster_factory import make_cluster
from src.utils.seeding import make_rng

_TDP = {
    NodeType.CPU: (65.0, 150.0),
    NodeType.GPU: (250.0, 400.0),
    NodeType.FPGA: (30.0, 75.0),
    NodeType.TPU: (200.0, 450.0),
}


def test_node_id_equals_index_and_count() -> None:
    nodes = make_cluster(make_rng(0), n_nodes=8, beta=5.0)
    assert len(nodes) == 8
    assert all(n.node_id == i for i, n in enumerate(nodes))


def test_power_within_tdp_ranges() -> None:
    nodes = make_cluster(make_rng(1), n_nodes=12, beta=5.0)
    for n in nodes:
        lo, hi = _TDP[n.node_type]
        assert lo <= n.power_w <= hi


def test_all_four_types_present_when_enough_nodes() -> None:
    nodes = make_cluster(make_rng(2), n_nodes=4, beta=5.0)
    assert {n.node_type for n in nodes} == set(NodeType)
