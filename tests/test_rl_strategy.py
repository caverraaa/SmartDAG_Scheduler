import torch

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.rl.gnn_encoder import GNNEncoder
from src.rl.policy import TwoHeadPolicy
from src.rl.rl_strategy import RLStrategy
from src.scheduler.task_scheduler import run_episode
from src.utils.config import load_config


def _instance():
    tasks = [Task(i, 2.0, 1.0, TaskClass.SEQUENTIAL) for i in range(4)]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_rl_strategy_plugs_into_run_episode() -> None:
    torch.manual_seed(0)
    policy = TwoHeadPolicy(GNNEncoder(hidden=16, layers=2), hidden=16)
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    schedule, info = run_episode(env, RLStrategy(policy), dag=dag, nodes=nodes)
    assert sorted(a.task_id for a in schedule.assignments) == [0, 1, 2, 3]
    assert info["makespan"] > 0.0
