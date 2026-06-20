"""Weighted-Sum Greedy: myopic argmin of the SAME objective the env rewards (TZ §8).

The scientific control isolating "RL wins by learning" from "RL wins only because
heuristics ignore energy". Uses placement.weighted_cost verbatim, so parity with
the env reward is structural. Balance is terminal-only and never chased.
"""

from src.env.placement import ClusterState, weighted_cost
from src.strategies.base import BaseSchedulingStrategy


class WeightedSumGreedyStrategy(BaseSchedulingStrategy):
    def __init__(self, w1: float, w2: float) -> None:
        self._w1 = w1
        self._w2 = w2

    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        best: tuple[float, int, int] | None = None  # (cost, task_id, node_id)
        for task_id in sorted(ready):
            task = state.dag.task(task_id)
            for node in sorted(state.nodes, key=lambda n: n.node_id):
                if not node.alive:
                    continue
                comp = weighted_cost(task, node, state)
                cost = self._w1 * comp.d_makespan_norm + self._w2 * comp.d_energy_norm
                if best is None or cost < best[0]:
                    best = (cost, task_id, node.node_id)
        assert best is not None
        return best[1], best[2]
