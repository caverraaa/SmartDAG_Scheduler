from dataclasses import replace

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.scheduler.system_monitor import SystemMonitor
from src.scheduler.task_scheduler import run_episode
from src.strategies.heft import HEFTStrategy
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


def test_run_episode_terminates_gracefully_on_total_deadlock() -> None:
    """Verify run_episode returns without raising when all nodes die and deadlock occurs.

    This test validates that:
    1. run_episode does NOT raise AssertionError when env.schedule might be accessed
    2. The episode detects and returns the deadlock state gracefully
    3. The returned info dict contains deadlocked=True
    4. Not all tasks are scheduled (proving the deadlock occurred)
    """
    cfg = replace(load_config("config.yaml"), failure_rate=1e6)
    env = ClusterEnv(cfg)
    dag, nodes = _instance()
    schedule, info = run_episode(env, HEFTStrategy(), dag=dag, nodes=nodes)
    # run_episode should return without raising
    assert schedule is not None
    # Check that deadlock was detected
    assert info.get("deadlocked") is True
    # Not all 4 tasks should be completed due to deadlock
    assert len(schedule.assignments) < 4
