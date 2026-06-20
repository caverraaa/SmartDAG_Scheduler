from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.scheduler.task_scheduler import run_episode
from src.strategies.cpop import CPOPStrategy, critical_path_processor, critical_path_set
from src.strategies.ranking import downward_rank, upward_rank
from src.utils.config import load_config


def _asym() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 6.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(3, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_critical_path_set_is_heavy_branch() -> None:
    dag, nodes = _asym()
    ru = upward_rank(dag, nodes)
    rd = downward_rank(dag, nodes)
    cp = critical_path_set(ru, rd)
    # priorities: t0=9.5, t1=9.5, t2=6.5, t3=9.5 -> CP = {0,1,3}, t2 excluded
    assert cp == {0, 1, 3}


def test_critical_path_processor_is_gpu() -> None:
    dag, nodes = _asym()
    ru = upward_rank(dag, nodes)
    rd = downward_rank(dag, nodes)
    cp = critical_path_set(ru, rd)
    # sum exec over {0,1,3}: CPU=2+6+2=10, GPU=1+3+1=5 -> GPU
    assert critical_path_processor(dag, cp, nodes) == 1


def test_cpop_full_schedule_golden() -> None:
    dag, nodes = _asym()
    env = ClusterEnv(load_config("config.yaml"))
    schedule, info = run_episode(env, CPOPStrategy(), dag=dag, nodes=nodes)
    node_of = {a.task_id: a.node_id for a in schedule.assignments}
    # CP {0,1,3} -> GPU(1); non-CP t2 -> EFT picks CPU(0)
    assert node_of == {0: 1, 1: 1, 2: 0, 3: 1}
    assert info["makespan"] == 6.0


def test_cpop_tie_break_selects_lowest_id() -> None:
    """Test that equal-priority tasks are resolved by lowest id, independent of input order."""
    # Two independent entry tasks with identical cost => equal priority
    tasks = [
        Task(0, 3.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 3.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [])  # no edges
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    env = ClusterEnv(load_config("config.yaml"))
    env.reset(dag=dag, nodes=nodes)

    # Both ready, equal priority => should pick task 0 (lowest id)
    task_id, _ = CPOPStrategy().predict([0, 1], env.state)
    assert task_id == 0

    # Reverse order in ready list => should still pick task 0 (regression guard)
    task_id, _ = CPOPStrategy().predict([1, 0], env.state)
    assert task_id == 0
