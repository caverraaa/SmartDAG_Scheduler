import numpy as np
import torch

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.rl.gnn_encoder import GNNEncoder
from src.rl.policy import TwoHeadPolicy
from src.rl.ppo_trainer import PPOTrainer
from src.utils.config import load_config


def _instance():
    # Tiny instance with a real makespan<->energy trade-off (CPU cheap/slow, GPU fast/costly).
    tasks = [Task(i, 3.0, 1.0, TaskClass.SEQUENTIAL) for i in range(4)]
    dag = TaskDAG(tasks, [(0, 1, 5.0), (0, 2, 5.0), (1, 3, 5.0), (2, 3, 5.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 4.0 for tc in TaskClass}, 320.0, 10.0),
    ]
    return dag, nodes


def _mean_sampled_reward(policy, env, dag, nodes, episodes: int = 12) -> float:
    rewards = []
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset(dag=dag, nodes=nodes)
            done = False
            total = 0.0
            while not done:
                a, _lp, _v = policy.act(obs)
                obs, r, done, _ = env.step(a)
                total += r
            rewards.append(total)
    return float(np.mean(rewards))


def test_training_improves_reward_on_tiny_instance() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = load_config("config.yaml")
    policy = TwoHeadPolicy(GNNEncoder(hidden=16, layers=2), hidden=16)
    trainer = PPOTrainer(policy, cfg)
    env = ClusterEnv(cfg)
    dag, nodes = _instance()

    before = _mean_sampled_reward(policy, env, dag, nodes)
    history = trainer.train(env, n_updates=60, dag=dag, nodes=nodes)
    after = _mean_sampled_reward(policy, env, dag, nodes)

    # Learning sanity gate (§9): reward must rise. A flat/declining reward => bug.
    assert after > before
    # history records per-update stats including mean_reward
    assert len(history) == 60
    assert "mean_reward" in history[0]


def test_evaluate_vs_heft_returns_makespans() -> None:
    torch.manual_seed(0)
    cfg = load_config("config.yaml")
    trainer = PPOTrainer(TwoHeadPolicy(GNNEncoder(hidden=16, layers=2), hidden=16), cfg)
    env = ClusterEnv(cfg)
    dag, nodes = _instance()
    result = trainer.evaluate_vs_heft(env, [(dag, nodes)])
    assert "rl_makespan" in result and "heft_makespan" in result
    assert result["rl_makespan"] > 0.0 and result["heft_makespan"] > 0.0
