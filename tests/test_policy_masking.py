import torch

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.env.observation import build_observation
from src.rl.gnn_encoder import GNNEncoder
from src.rl.obs_tensors import obs_to_tensors
from src.rl.policy import TwoHeadPolicy
from src.utils.config import load_config


def _obs(dead_node: int | None = None):
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
    _obs_initial, _ = env.reset(dag=dag, nodes=nodes)
    # Set alive=False after reset (reset() calls node.reset() which forces alive=True)
    if dead_node is not None:
        env.state.nodes[dead_node].alive = False
    obs = build_observation(env.state, env.scheduled, current_makespan=0.0)
    return obs


def _policy(hidden: int = 32) -> TwoHeadPolicy:
    torch.manual_seed(0)
    return TwoHeadPolicy(GNNEncoder(hidden=hidden, layers=2), hidden=hidden)


def test_task_logits_zero_prob_for_non_ready() -> None:
    obs = _obs()  # only task 0 is ready at reset
    pol = _policy()
    t = obs_to_tensors(obs)
    h, _n, ctx = pol.encode(t)
    logits = pol.task_logits(h, ctx, t.ready_mask)
    probs = torch.softmax(logits, dim=-1)
    assert probs[0] > 0.0
    assert torch.allclose(probs[[1, 2, 3]], torch.zeros(3))


def test_node_logits_zero_prob_for_dead_node() -> None:
    obs = _obs(dead_node=1)
    pol = _policy()
    t = obs_to_tensors(obs)
    h, n_emb, ctx = pol.encode(t)
    logits = pol.node_logits(h[0], n_emb, ctx, t.alive_mask)
    probs = torch.softmax(logits, dim=-1)
    assert probs[0] > 0.0 and probs[1] == 0.0


def test_handles_variable_candidate_set_sizes() -> None:
    pol = _policy()
    for n_tasks, n_nodes in [(4, 2), (3, 3)]:
        tasks = [Task(i, 2.0, 1.0, TaskClass.SEQUENTIAL) for i in range(n_tasks)]
        edges = [(0, i, 10.0) for i in range(1, n_tasks)]
        dag = TaskDAG(tasks, edges)
        nodes = [
            ComputeNode(j, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0)
            for j in range(n_nodes)
        ]
        env = ClusterEnv(load_config("config.yaml"))
        obs, _ = env.reset(dag=dag, nodes=nodes)
        t = obs_to_tensors(obs)
        h, n_emb, ctx = pol.encode(t)
        assert pol.task_logits(h, ctx, t.ready_mask).shape == (n_tasks,)
        assert pol.node_logits(h[0], n_emb, ctx, t.alive_mask).shape == (n_nodes,)


def test_value_is_scalar() -> None:
    obs = _obs()
    pol = _policy()
    t = obs_to_tensors(obs)
    _h, _n, ctx = pol.encode(t)
    v = pol.value(ctx)
    assert v.shape == ()
