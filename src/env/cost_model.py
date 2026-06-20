"""Stateless cost arithmetic + heterogeneous speed-table generation (TZ §5.1)."""

import numpy as np

from src.core.compute_node import ComputeNode, NodeType
from src.core.task import Task, TaskClass

# Per-(class, type) affinity multipliers: the "right" node is faster for each class.
_AFFINITY: dict[TaskClass, dict[NodeType, float]] = {
    TaskClass.DATA_PARALLEL: {
        NodeType.GPU: 1.0,
        NodeType.TPU: 0.9,
        NodeType.FPGA: 0.4,
        NodeType.CPU: 0.25,
    },
    TaskClass.SEQUENTIAL: {
        NodeType.CPU: 1.0,
        NodeType.FPGA: 0.6,
        NodeType.GPU: 0.35,
        NodeType.TPU: 0.3,
    },
    TaskClass.STREAMING: {
        NodeType.FPGA: 1.0,
        NodeType.TPU: 0.7,
        NodeType.GPU: 0.6,
        NodeType.CPU: 0.35,
    },
}


def exec_time(task: Task, node: ComputeNode) -> float:
    return task.base_cost / node.speed(task.task_class)


def energy(task: Task, node: ComputeNode) -> float:
    return node.power_w * exec_time(task, node)


def comm_time(data_volume: float, bandwidth: float, latency: float = 0.0) -> float:
    return data_volume / bandwidth + latency


def make_speed_table(
    rng: np.random.Generator, beta: float
) -> dict[NodeType, dict[TaskClass, float]]:
    """Map each (node_type, task_class) to a speed coefficient.

    For each task class the affinity profile is scaled so the max/min speed
    ratio across node types is approximately ``beta`` (with mild jitter).
    """
    table: dict[NodeType, dict[TaskClass, float]] = {nt: {} for nt in NodeType}
    for tc in TaskClass:
        affinity = _AFFINITY[tc]
        # Map best affinity (1.0) -> beta, worst -> 1.0, linearly.
        lo = min(affinity.values())
        hi = max(affinity.values())
        for nt in NodeType:
            frac = (affinity[nt] - lo) / (hi - lo)
            base_speed = 1.0 + frac * (beta - 1.0)
            jitter = float(rng.uniform(0.9, 1.1))
            table[nt][tc] = base_speed * jitter
    return table
