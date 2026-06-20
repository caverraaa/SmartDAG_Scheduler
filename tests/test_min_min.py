from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.scheduler.task_scheduler import run_episode
from src.strategies.min_min import MinMinStrategy
from src.utils.config import load_config


def _independent() -> tuple[TaskDAG, list[ComputeNode]]:
    # three independent tasks (no edges), two equal-speed nodes
    tasks = [
        Task(0, 1.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 3.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
    ]
    return dag, nodes


def test_min_min_first_pick_is_smallest_task() -> None:
    dag, nodes = _independent()
    env = ClusterEnv(load_config("config.yaml"))
    env.reset(dag=dag, nodes=nodes)
    task_id, node_id = MinMinStrategy().predict([0, 1, 2], env.state)
    # smallest min-completion = task0 (cost 1) on node0 (tie -> lowest index)
    assert (task_id, node_id) == (0, 0)


def test_min_min_full_schedule_golden() -> None:
    dag, nodes = _independent()
    env = ClusterEnv(load_config("config.yaml"))
    schedule, info = run_episode(env, MinMinStrategy(), dag=dag, nodes=nodes)
    placed = {(a.task_id, a.node_id, a.start, a.finish) for a in schedule.assignments}
    # Hand-computed Min-Min: t0->n0[0,1], t1->n1[0,2], t2->n0[1,4]
    assert (0, 0, 0.0, 1.0) in placed
    assert (1, 1, 0.0, 2.0) in placed
    assert (2, 0, 1.0, 4.0) in placed
    assert info["makespan"] == 4.0
