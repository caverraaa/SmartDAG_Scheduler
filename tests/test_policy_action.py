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


def test_act_returns_ready_task_and_alive_node() -> None:
    obs = _obs()
    pol = _policy()
    torch.manual_seed(1)
    (task_id, node_id), log_prob, value = pol.act(obs)
    assert task_id == 0  # only task 0 is ready
    assert node_id in (0, 1)
    assert log_prob.shape == () and value.shape == ()


def test_joint_log_prob_is_sum_of_head_log_probs() -> None:
    obs = _obs()
    pol = _policy()
    t = obs_to_tensors(obs)
    h, n_emb, ctx = pol.encode(t)
    task_id, node_id = 0, 1
    task_lp = torch.log_softmax(pol.task_logits(h, ctx, t.ready_mask), dim=-1)[task_id]
    node_lp = torch.log_softmax(pol.node_logits(h[task_id], n_emb, ctx, t.alive_mask), dim=-1)[
        node_id
    ]
    log_prob, entropy, value = pol.evaluate_action(obs, task_id, node_id)
    assert torch.allclose(log_prob, task_lp + node_lp, atol=1e-6)
    assert entropy.item() >= 0.0


def test_gradients_flow_through_both_heads_and_critic() -> None:
    # Use a 2-ready state (after scheduling task 0) so the masked task
    # distribution is non-degenerate and head_task receives gradient through
    # both log_prob and entropy.
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
    obs0, _ = env.reset(dag=dag, nodes=nodes)
    obs1, _, _, _ = env.step((0, 0))  # schedule task 0 → tasks 1 and 2 become ready
    pol = _policy()
    log_prob, entropy, value = pol.evaluate_action(obs1, 1, 0)
    loss = -(log_prob) + (value - 1.0) ** 2 - 0.01 * entropy
    loss.backward()

    def has_grad(module) -> bool:
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert has_grad(pol.head_task)
    assert has_grad(pol.head_node)
    assert has_grad(pol.critic)
    assert has_grad(pol.encoder)
