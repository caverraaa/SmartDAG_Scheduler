"""CPOP: critical-path-on-a-processor list scheduling (TZ §8).

Priority = upward_rank + downward_rank. Tasks whose priority equals the maximum
form the critical path and are bound to a single critical-path processor (the
node minimising their total exec time); other tasks use EFT. No-insertion form.
"""

from weakref import WeakKeyDictionary

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.env.cost_model import exec_time
from src.env.placement import ClusterState, earliest_start_finish
from src.strategies.base import BaseSchedulingStrategy
from src.strategies.ranking import downward_rank, upward_rank

_CP_TOL = 1e-6


def critical_path_set(upward: dict[int, float], downward: dict[int, float]) -> set[int]:
    priority = {i: upward[i] + downward[i] for i in upward}
    cp_value = max(priority.values())
    return {i for i, p in priority.items() if abs(p - cp_value) < _CP_TOL}


def critical_path_processor(dag: TaskDAG, cp: set[int], nodes: list[ComputeNode]) -> int:
    def cp_cost(node: ComputeNode) -> float:
        return sum(exec_time(dag.task(i), node) for i in cp)

    return min(nodes, key=lambda n: (cp_cost(n), n.node_id)).node_id


class CPOPStrategy(BaseSchedulingStrategy):
    def __init__(self) -> None:
        self._cache: WeakKeyDictionary[TaskDAG, tuple[dict[int, float], set[int], int]] = (
            WeakKeyDictionary()
        )

    def _structure(self, state: ClusterState) -> tuple[dict[int, float], set[int], int]:
        cached = self._cache.get(state.dag)
        if cached is None:
            ru = upward_rank(state.dag, state.nodes)
            rd = downward_rank(state.dag, state.nodes)
            priority = {i: ru[i] + rd[i] for i in ru}
            cp = critical_path_set(ru, rd)
            cp_proc = critical_path_processor(state.dag, cp, state.nodes)
            cached = (priority, cp, cp_proc)
            self._cache[state.dag] = cached
        return cached

    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        priority, cp, cp_proc = self._structure(state)
        task_id = max(ready, key=lambda t: (priority[t], -t))  # highest priority, then lowest id
        if task_id in cp and state.nodes[cp_proc].alive:
            return task_id, cp_proc
        # Non-CP task, or the critical-path processor has failed: place via EFT on a
        # surviving node (the same reactive fallback every other strategy uses).
        task = state.dag.task(task_id)
        alive = [n for n in state.nodes if n.alive]
        node = min(alive, key=lambda n: (earliest_start_finish(task, n, state)[1], n.node_id))
        return task_id, node.node_id
