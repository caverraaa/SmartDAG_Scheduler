"""Construct heterogeneous clusters of ComputeNodes (TZ §5.1)."""

import numpy as np

from src.core.compute_node import ComputeNode, NodeType
from src.env.cost_model import make_speed_table

_TDP_RANGE: dict[NodeType, tuple[float, float]] = {
    NodeType.CPU: (65.0, 150.0),
    NodeType.GPU: (250.0, 400.0),
    NodeType.FPGA: (30.0, 75.0),
    NodeType.TPU: (200.0, 450.0),
}
_TYPE_CYCLE: list[NodeType] = [NodeType.CPU, NodeType.GPU, NodeType.FPGA, NodeType.TPU]


def make_cluster(rng: np.random.Generator, n_nodes: int, beta: float) -> list[ComputeNode]:
    speed_table = make_speed_table(rng, beta)
    nodes: list[ComputeNode] = []
    for i in range(n_nodes):
        nt = _TYPE_CYCLE[i % len(_TYPE_CYCLE)]
        lo, hi = _TDP_RANGE[nt]
        nodes.append(
            ComputeNode(
                node_id=i,
                node_type=nt,
                speed_by_class=dict(speed_table[nt]),
                power_w=float(rng.uniform(lo, hi)),
                bandwidth=float(rng.uniform(5.0, 20.0)),
            )
        )
    return nodes
