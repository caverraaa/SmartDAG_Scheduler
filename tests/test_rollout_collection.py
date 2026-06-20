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
    tasks = [Task(i, 2.0, 1.0, TaskClass.SEQUENTIAL) for i in range(4)]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def _trainer():
    torch.manual_seed(0)
    cfg = load_config("config.yaml")
    return PPOTrainer(TwoHeadPolicy(GNNEncoder(hidden=16, layers=2), hidden=16), cfg), cfg


def test_collect_rollouts_transition_count() -> None:
    trainer, cfg = _trainer()
    env = ClusterEnv(cfg)
    dag, nodes = _instance()
    buf = trainer.collect_rollouts(env, n_episodes=3, dag=dag, nodes=nodes)
    # each episode schedules all 4 tasks -> 4 transitions; 3 episodes -> 12
    assert len(buf) == 12
    assert buf.advantages is not None and len(buf.advantages) == 12
    # the last transition of each episode is terminal
    assert buf.transitions[3].done and buf.transitions[7].done and buf.transitions[11].done


def test_checkpoint_round_trip(tmp_path) -> None:
    trainer, cfg = _trainer()
    path = str(tmp_path / "ckpt.pth")
    trainer.save_checkpoint(path)
    # perturb params, then restore
    with torch.no_grad():
        for p in trainer.policy.parameters():
            p.add_(1.0)
    trainer.load_checkpoint(path)
    fresh = PPOTrainer(TwoHeadPolicy(GNNEncoder(hidden=16, layers=2), hidden=16), cfg)
    fresh.load_checkpoint(path)
    for a, b in zip(trainer.policy.parameters(), fresh.policy.parameters(), strict=True):
        assert torch.equal(a, b)
