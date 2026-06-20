from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.scheduler.task_scheduler import run_episode
from src.strategies.heft import HEFTStrategy


def _asym() -> tuple[TaskDAG, list[ComputeNode]]:
    # 0->1->3, 0->2->3 ; branch via 1 is heavy (base 6) => critical
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


def test_heft_first_pick_is_highest_rank_on_fastest_node() -> None:
    dag, nodes = _asym()
    env = ClusterEnv(load_config_path())
    env.reset(dag=dag, nodes=nodes)
    task_id, node_id = HEFTStrategy().predict([0], env.state)
    assert task_id == 0  # only ready task, highest upward rank
    assert node_id == 1  # GPU finishes task0 at 1.0 vs CPU at 2.0


def test_heft_full_schedule_golden() -> None:
    dag, nodes = _asym()
    env = ClusterEnv(load_config_path())
    schedule, info = run_episode(env, HEFTStrategy(), dag=dag, nodes=nodes)
    node_of = {a.task_id: a.node_id for a in schedule.assignments}
    # HEFT (hand-computed): t0,t1,t3 -> GPU(1); t2 -> CPU(0)
    assert node_of == {0: 1, 1: 1, 2: 0, 3: 1}
    assert info["makespan"] == 6.0


def load_config_path():  # noqa: ANN201 - tiny test helper
    from src.utils.config import load_config

    return load_config("config.yaml")
