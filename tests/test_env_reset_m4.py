import math
from dataclasses import replace

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.utils.config import load_config


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [Task(i, 2.0, 1.0, TaskClass.SEQUENTIAL) for i in range(4)]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_deterministic_knobs_zero_calendar() -> None:
    env = ClusterEnv(load_config("config.yaml"))  # noise_std=0, failure_rate=0
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    assert all(v == 0.0 for v in env.state.noise_eps.values())
    assert all(math.isinf(v) for v in env.state.failure_times.values())
    assert set(env.state.noise_eps) == {0, 1, 2, 3}
    assert set(env.state.failure_times) == {0, 1}


def test_stochastic_calendar_is_instance_seed_keyed_and_repeatable() -> None:
    cfg = replace(load_config("config.yaml"), noise_std=0.2, failure_rate=0.1)
    dag, nodes = _instance()
    a = ClusterEnv(cfg)
    a.reset(dag=dag, nodes=nodes)
    b = ClusterEnv(cfg)
    b.reset(dag=dag, nodes=nodes)
    # Same instance + same seed -> bit-identical calendar (the fairness property).
    assert a.state.noise_eps == b.state.noise_eps
    assert a.state.failure_times == b.state.failure_times
    # Stochastic knobs actually produced non-trivial values.
    assert any(v != 0.0 for v in a.state.noise_eps.values())
    assert all(v > 0.0 and v != float("inf") for v in a.state.failure_times.values())
