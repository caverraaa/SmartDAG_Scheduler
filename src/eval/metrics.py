"""Per-run evaluation metrics (TZ §10) + a predict-timing strategy wrapper.

All functions are pure (read a finished schedule + cached refs; no re-run).
"""

import time

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.core.schedule import Schedule
from src.env.cost_model import exec_time
from src.env.placement import ClusterState
from src.strategies.base import BaseSchedulingStrategy


def utilisation(schedule: Schedule, c_max: float, alive_ids: list[int]) -> float:
    """Total busy time / (makespan x number of alive nodes)."""
    if c_max <= 0.0 or not alive_ids:
        return 0.0
    total_busy = sum(schedule.busy_time_by_node().values())
    return total_busy / (c_max * len(alive_ids))


def slr(c_max: float, m_ref: float) -> float:
    """Schedule Length Ratio: makespan / fastest-exec critical-path lower bound."""
    return c_max / m_ref if m_ref > 0.0 else 0.0


def speedup(dag: TaskDAG, nodes: list[ComputeNode], c_max: float) -> float:
    """Serial time on the single fastest node / parallel makespan (Topcuoglu)."""
    if c_max <= 0.0 or not nodes:
        return 0.0
    serial = min(sum(exec_time(dag.task(i), node) for i in range(dag.n_tasks)) for node in nodes)
    return serial / c_max


def compute_run_metrics(
    schedule: Schedule,
    info: dict,
    dag: TaskDAG,
    nodes: list[ComputeNode],
    alive_ids: list[int],
    predict_seconds: float,
) -> dict[str, float]:
    """One row of metrics for a finished episode."""
    c_max = schedule.makespan()
    return {
        "makespan": c_max,
        "energy": schedule.total_energy,
        "utilisation": utilisation(schedule, c_max, alive_ids),
        "load_balance": schedule.load_balance_index(alive_ids),
        "slr": slr(c_max, float(info["m_ref"])),
        "speedup": speedup(dag, nodes, c_max),
        "overhead_ms": predict_seconds * 1000.0,
    }


class TimingStrategy(BaseSchedulingStrategy):
    """Wrap a strategy, delegating predict while accumulating wall-clock time."""

    def __init__(self, inner: BaseSchedulingStrategy) -> None:
        self._inner = inner
        self.predict_seconds = 0.0

    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        t0 = time.perf_counter()
        action = self._inner.predict(ready, state)
        self.predict_seconds += time.perf_counter() - t0
        return action
