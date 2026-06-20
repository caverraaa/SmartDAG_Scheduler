import torch

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.rl.obs_tensors import obs_to_tensors
from src.utils.config import load_config


def _obs():
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
    env = ClusterEnv(load_config("config.yaml"))
    obs, _ = env.reset(dag=dag, nodes=nodes)
    return obs


def test_obs_to_tensors_shapes_and_dtypes() -> None:
    t = obs_to_tensors(_obs())
    assert t.task_features.shape == (4, 15) and t.task_features.dtype == torch.float32
    assert t.node_features.shape == (2, 9) and t.node_features.dtype == torch.float32
    assert t.edge_index.shape == (2, 4) and t.edge_index.dtype == torch.int64
    assert t.globals.shape == (2,)
    assert t.ready_mask.dtype == torch.bool and t.ready_mask.shape == (4,)
    assert t.alive_mask.dtype == torch.bool and t.alive_mask.shape == (2,)


def test_ready_mask_matches_observation() -> None:
    obs = _obs()
    t = obs_to_tensors(obs)
    assert t.ready_mask.tolist() == obs.ready_mask.tolist()
    assert t.alive_mask.tolist() == obs.alive_mask.tolist()
