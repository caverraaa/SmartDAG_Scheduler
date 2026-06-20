from dataclasses import replace

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.scheduler.system_monitor import SystemMonitor
from src.utils.config import load_config


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [Task(i, 2.0, 1.0, TaskClass.SEQUENTIAL) for i in range(4)]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_monitor_reports_new_failure_once() -> None:
    env = ClusterEnv(replace(load_config("config.yaml"), failure_rate=0.1))
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    monitor = SystemMonitor()
    assert monitor.check(env.state) == []  # nothing dead yet
    env.state.failure_times[1] = 0.5
    env.step((0, 1))  # node 1 dies
    assert monitor.check(env.state) == [1]  # newly dead
    assert monitor.check(env.state) == []  # already reported


def test_monitor_notifies_subscribers() -> None:
    env = ClusterEnv(replace(load_config("config.yaml"), failure_rate=0.1))
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    seen: list[int] = []
    monitor = SystemMonitor()
    monitor.subscribe(lambda _state: seen.append(1))
    env.state.failure_times[1] = 0.5
    env.step((0, 1))
    monitor.check(env.state)
    assert seen == [1]
