from dataclasses import replace

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.utils.config import load_config


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [Task(i, 3.0, 1.0, TaskClass.SEQUENTIAL) for i in range(4)]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_calendar_and_noise_are_strategy_independent() -> None:
    # Same instance + seed => bit-identical adversity regardless of who schedules.
    cfg = replace(load_config("config.yaml"), noise_std=0.2, failure_rate=0.05)
    dag, nodes = _instance()
    env1 = ClusterEnv(cfg)
    env1.reset(dag=dag, nodes=nodes)
    env2 = ClusterEnv(cfg)
    env2.reset(dag=dag, nodes=nodes)
    assert env1.state.failure_times == env2.state.failure_times
    assert env1.state.noise_eps == env2.state.noise_eps
