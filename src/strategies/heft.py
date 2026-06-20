"""HEFT: list-scheduling by upward rank + earliest-finish-time assignment (TZ §8).

No-insertion (append-only EFT) form — stated honestly in the thesis. Ranks are
cluster-aware (averaged over processors) and cached per DAG.
"""

from weakref import WeakKeyDictionary

from src.core.dag import TaskDAG
from src.env.placement import ClusterState, earliest_start_finish
from src.strategies.base import BaseSchedulingStrategy
from src.strategies.ranking import upward_rank


class HEFTStrategy(BaseSchedulingStrategy):
    def __init__(self) -> None:
        self._rank_cache: WeakKeyDictionary[TaskDAG, dict[int, float]] = WeakKeyDictionary()

    def _ranks(self, state: ClusterState) -> dict[int, float]:
        ranks = self._rank_cache.get(state.dag)
        if ranks is None:
            ranks = upward_rank(state.dag, state.nodes)
            self._rank_cache[state.dag] = ranks
        return ranks

    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        ranks = self._ranks(state)
        task_id = max(ready, key=lambda t: (ranks[t], -t))  # highest rank, then lowest id
        task = state.dag.task(task_id)
        alive = [n for n in state.nodes if n.alive]
        node = min(alive, key=lambda n: (earliest_start_finish(task, n, state)[1], n.node_id))
        return task_id, node.node_id
