"""Min-Min: pick the (task, node) with the globally smallest finish time (TZ §8)."""

from src.env.placement import ClusterState, earliest_start_finish
from src.strategies.base import BaseSchedulingStrategy


class MinMinStrategy(BaseSchedulingStrategy):
    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        best: tuple[float, int, int] | None = None  # (finish, task_id, node_id)
        for task_id in ready:
            task = state.dag.task(task_id)
            for node in state.nodes:
                if not node.alive:
                    continue
                _, finish = earliest_start_finish(task, node, state)
                if best is None or finish < best[0]:
                    best = (finish, task_id, node.node_id)
        assert best is not None
        return best[1], best[2]
