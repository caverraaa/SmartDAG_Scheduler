"""Random scheduling strategy — the sanity floor baseline (TZ §8)."""

import numpy as np

from src.env.placement import ClusterState
from src.strategies.base import BaseSchedulingStrategy


class RandomStrategy(BaseSchedulingStrategy):
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        task_id = int(self._rng.choice(ready))
        alive_ids = [n.node_id for n in state.nodes if n.alive]
        node_id = int(self._rng.choice(alive_ids))
        return task_id, node_id
