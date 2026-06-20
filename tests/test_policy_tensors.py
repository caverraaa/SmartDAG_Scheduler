import torch

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.rl.gnn_encoder import GNNEncoder
from src.rl.obs_tensors import obs_to_tensors
from src.rl.policy import TwoHeadPolicy
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


def _policy(hidden: int = 32) -> TwoHeadPolicy:
    torch.manual_seed(0)
    return TwoHeadPolicy(GNNEncoder(hidden=hidden, layers=2), hidden=hidden)


def test_evaluate_tensors_matches_evaluate_action() -> None:
    obs = _obs()
    pol = _policy()
    lp_a, ent_a, v_a = pol.evaluate_action(obs, 0, 1)
    lp_t, ent_t, v_t = pol.evaluate_tensors(obs_to_tensors(obs), 0, 1)
    assert torch.allclose(lp_a, lp_t) and torch.allclose(ent_a, ent_t) and torch.allclose(v_a, v_t)


def test_act_from_tensors_returns_ready_and_alive() -> None:
    obs = _obs()
    pol = _policy()
    torch.manual_seed(1)
    (task_id, node_id), log_prob, value = pol.act_from_tensors(obs_to_tensors(obs))
    assert task_id == 0  # only ready task
    assert node_id in (0, 1)
    assert log_prob.shape == () and value.shape == ()


def test_act_greedy_is_deterministic_argmax() -> None:
    obs = _obs()
    pol = _policy()
    a = pol.act_greedy(obs)
    b = pol.act_greedy(obs)
    assert a == b  # deterministic
    assert a[0] == 0 and a[1] in (0, 1)
