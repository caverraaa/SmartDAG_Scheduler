"""DAGFactory: Factory pattern over interchangeable DAG sources (TZ §3, §7)."""

import numpy as np

from src.core.dag import TaskDAG
from src.dag_factory.synthetic import generate_synthetic


class DAGFactory:
    @classmethod
    def create(cls, source: str, rng: np.random.Generator, **params: float) -> TaskDAG:
        if source == "synthetic":
            return generate_synthetic(
                rng,
                n_tasks=int(params["n_tasks"]),
                n_layers=int(params["n_layers"]),
                edge_prob=float(params["edge_prob"]),
                ccr=float(params["ccr"]),
            )
        raise ValueError(f"Unknown DAG source: {source!r}")
