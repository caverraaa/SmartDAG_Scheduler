import time

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.env.cluster_env import ClusterEnv
from src.env.placement import ClusterState
from src.eval.metrics import TimingStrategy, compute_run_metrics, slr, speedup, utilisation
from src.strategies.base import BaseSchedulingStrategy
from src.utils.config import load_config


def _run_golden(golden_instance: tuple[TaskDAG, list[ComputeNode]]):
    dag, nodes = golden_instance
    env = ClusterEnv(load_config("config.yaml"))
    _, info = env.reset(dag=dag, nodes=nodes)
    for action in [(0, 1), (1, 1), (2, 1), (3, 1)]:
        _, _, done, info = env.step(action)
    alive_ids = [n.node_id for n in env.state.nodes if n.alive]
    return env.schedule, info, dag, nodes, alive_ids


def test_pure_metric_helpers() -> None:
    assert slr(6.0, 4.0) == 1.5
    assert slr(6.0, 0.0) == 0.0  # guard


def test_compute_run_metrics_golden(golden_instance) -> None:
    schedule, info, dag, nodes, alive_ids = _run_golden(golden_instance)
    m = compute_run_metrics(schedule, info, dag, nodes, alive_ids, predict_seconds=0.0)
    assert m["makespan"] == 6.0
    assert m["energy"] == 1200.0
    assert m["slr"] == 1.5
    assert m["speedup"] == 1.0
    assert m["utilisation"] == 0.5
    assert m["load_balance"] == 0.0
    assert m["overhead_ms"] == 0.0


def test_speedup_and_utilisation_helpers(golden_instance) -> None:
    schedule, info, dag, nodes, alive_ids = _run_golden(golden_instance)
    assert speedup(dag, nodes, 6.0) == 1.0
    assert utilisation(schedule, 6.0, alive_ids) == 0.5


def test_timing_strategy_accumulates_and_delegates() -> None:
    class _Slow(BaseSchedulingStrategy):
        def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
            time.sleep(0.001)
            return (ready[0], 0)

    ts = TimingStrategy(_Slow())
    action = ts.predict([3, 5], state=None)  # type: ignore[arg-type]
    assert action == (3, 0)
    assert ts.predict_seconds > 0.0
