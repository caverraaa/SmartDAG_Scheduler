from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.scheduler.system_monitor import SystemMonitor
from src.scheduler.task_scheduler import run_episode
from src.strategies.random_strategy import RandomStrategy
from src.utils.config import load_config
from src.utils.seeding import make_rng


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 4.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 4.0, 1.0, TaskClass.SEQUENTIAL),
        Task(3, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_run_episode_produces_complete_valid_schedule() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    schedule, info = run_episode(env, RandomStrategy(make_rng(0)), dag=dag, nodes=nodes)
    # one assignment per task, each task exactly once
    assert len(schedule.assignments) == 4
    assert sorted(a.task_id for a in schedule.assignments) == [0, 1, 2, 3]
    # every node_id is a valid alive node index
    assert all(0 <= a.node_id < 2 for a in schedule.assignments)
    assert info["makespan"] > 0.0


def test_run_episode_respects_dependencies() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    schedule, _ = run_episode(env, RandomStrategy(make_rng(1)), dag=dag, nodes=nodes)
    finish = {a.task_id: a.finish for a in schedule.assignments}
    start = {a.task_id: a.start for a in schedule.assignments}
    # children cannot start before a parent finishes (comm >= 0)
    assert start[1] >= finish[0]
    assert start[3] >= finish[1] and start[3] >= finish[2]


def test_random_strategy_is_reproducible() -> None:
    dag, nodes = _instance()
    env_a = ClusterEnv(load_config("config.yaml"))
    env_b = ClusterEnv(load_config("config.yaml"))
    sched_a, _ = run_episode(env_a, RandomStrategy(make_rng(7)), dag=dag, nodes=nodes)
    sched_b, _ = run_episode(env_b, RandomStrategy(make_rng(7)), dag=dag, nodes=nodes)
    assert [(a.task_id, a.node_id) for a in sched_a.assignments] == [
        (a.task_id, a.node_id) for a in sched_b.assignments
    ]


def test_system_monitor_idle_in_deterministic_mode() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    assert SystemMonitor().check(env.state) == []
