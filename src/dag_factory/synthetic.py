"""Layered-random synthetic DAG generator (TZ §7, Appendix A.3)."""

import numpy as np

from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass

_CLASSES = list(TaskClass)


def generate_synthetic(
    rng: np.random.Generator,
    n_tasks: int,
    n_layers: int,
    edge_prob: float,
    ccr: float,
) -> TaskDAG:
    """Generate a valid acyclic DAG via layer assignment.

    Edges only ever point from an earlier layer to a later layer, so the graph
    is acyclic by construction. base_cost is log-normal (heavy-tailed, A.3).
    """
    # Assign each task id to a layer; keep layers non-empty by seeding one per layer.
    layers: list[list[int]] = [[] for _ in range(n_layers)]
    for i in range(n_tasks):
        layer = i % n_layers if i < n_layers else int(rng.integers(0, n_layers))
        layers[layer].append(i)

    tasks: list[Task] = []
    for i in range(n_tasks):
        base_cost = float(rng.lognormal(mean=1.0, sigma=0.6))
        tasks.append(
            Task(
                id=i,
                base_cost=base_cost,
                mem_required=float(rng.uniform(1.0, 8.0)),
                task_class=_CLASSES[int(rng.integers(0, len(_CLASSES)))],
            )
        )

    total_compute = sum(t.base_cost for t in tasks)

    # Connect tasks to some tasks in strictly later layers.
    raw_edges: list[tuple[int, int]] = []
    for li in range(n_layers - 1):
        for src in layers[li]:
            for lj in range(li + 1, n_layers):
                for dst in layers[lj]:
                    if rng.random() < edge_prob:
                        raw_edges.append((src, dst))

    # Scale edge data volumes so total communication ≈ ccr * total compute.
    n_edges = max(1, len(raw_edges))
    per_edge_volume = (ccr * total_compute) / n_edges
    edges = [(s, d, float(per_edge_volume * rng.uniform(0.5, 1.5))) for s, d in raw_edges]

    return TaskDAG(tasks, edges)
