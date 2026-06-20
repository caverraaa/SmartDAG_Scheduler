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


def test_noise_changes_committed_finish_not_planning() -> None:
    cfg = replace(load_config("config.yaml"), noise_std=0.3, failure_rate=0.0)
    env = ClusterEnv(cfg)
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    eps0 = env.state.noise_eps[0]
    # task 0 on GPU(node 1): nominal exec = base/speed = 2/2 = 1.0
    env.step((0, 1))
    committed_finish = env.state.task_finish[0]
    assert abs(committed_finish - 1.0 * (1.0 + eps0)) < 1e-9  # actual = nominal*(1+eps)
    assert eps0 != 0.0  # noise was active


def test_failure_kills_node_requeues_task_episode_completes() -> None:
    cfg = replace(load_config("config.yaml"), failure_rate=0.1)
    env = ClusterEnv(cfg)
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    env.state.failure_times[1] = 0.5  # node 1 dies early; any task finishes after 0.5
    env.state.failure_times[0] = float("inf")  # node 0 is the reliable survivor
    obs, reward, done, info = env.step((0, 1))  # task 0 on node 1: finish 1.0 > 0.5 -> FAIL
    assert reward == 0.0
    assert env.state.nodes[1].alive is False
    assert 0 not in env.scheduled  # not committed -> requeued
    assert info["failed_node"] == 1
    assert 0 in env.state.dag.ready_set(env.scheduled)  # ready again
    # re-assign onto the survivor (node 0) and finish the episode greedily on node 0
    done = False
    while not done:
        ready = env.state.dag.ready_set(env.scheduled)
        _, _, done, info = env.step((ready[0], 0))
    assert sorted(env.scheduled) == [0, 1, 2, 3]
    assert info["makespan"] > 0.0


def test_deadlock_when_all_nodes_die() -> None:
    cfg = replace(load_config("config.yaml"), failure_rate=0.1)
    env = ClusterEnv(cfg)
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    env.state.failure_times[0] = 0.0
    env.state.failure_times[1] = 0.0
    env.step((0, 1))  # node 1 dies; node 0 still alive -> not done
    obs, reward, done, info = env.step((0, 0))  # node 0 dies; no alive nodes, tasks remain
    assert done is True and info["deadlocked"] is True
