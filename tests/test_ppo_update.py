import torch

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.rl.gnn_encoder import GNNEncoder
from src.rl.obs_tensors import obs_to_tensors
from src.rl.policy import TwoHeadPolicy
from src.rl.ppo_trainer import PPOTrainer
from src.rl.rollout_buffer import RolloutBuffer
from src.utils.config import Config, load_config


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
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
    return dag, nodes


def _collect_one_episode(
    policy: TwoHeadPolicy, env: ClusterEnv, dag: TaskDAG, nodes: list[ComputeNode]
) -> RolloutBuffer:
    buf = RolloutBuffer()
    obs, _ = env.reset(dag=dag, nodes=nodes)
    done = False
    with torch.no_grad():
        while not done:
            t = obs_to_tensors(obs)
            (task_id, node_id), log_prob, value = policy.act_from_tensors(t)
            obs, reward, done, _ = env.step((task_id, node_id))
            buf.add(t, task_id, node_id, log_prob.item(), value.item(), reward, done)
    buf.compute_gae(gamma=1.0, lam=0.95)
    return buf


def _setup() -> tuple[Config, TwoHeadPolicy, ClusterEnv, TaskDAG, list[ComputeNode]]:
    torch.manual_seed(0)
    cfg = load_config("config.yaml")
    policy = TwoHeadPolicy(GNNEncoder(hidden=16, layers=2), hidden=16)
    env = ClusterEnv(cfg)
    dag, nodes = _instance()
    return cfg, policy, env, dag, nodes


def test_ratio_is_one_on_first_evaluation() -> None:
    # Right after collection, params are unchanged, so re-evaluating gives the same
    # log_prob => ratio == 1.
    cfg, policy, env, dag, nodes = _setup()
    buf = _collect_one_episode(policy, env, dag, nodes)
    tr = buf.transitions[0]
    new_log_prob, _ent, _v = policy.evaluate_tensors(tr.obs, tr.task_id, tr.node_id)
    ratio = torch.exp(new_log_prob - torch.tensor(tr.log_prob))
    assert abs(ratio.item() - 1.0) < 1e-5


def test_update_changes_params_and_loss_is_finite() -> None:
    cfg, policy, env, dag, nodes = _setup()
    buf = _collect_one_episode(policy, env, dag, nodes)
    before = [p.clone() for p in policy.parameters()]
    trainer = PPOTrainer(policy, cfg)
    stats = trainer.update(buf)
    assert all(map(lambda v: v == v and abs(v) != float("inf"), stats.values()))  # finite
    after = list(policy.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_update_returns_expected_keys() -> None:
    cfg, policy, env, dag, nodes = _setup()
    buf = _collect_one_episode(policy, env, dag, nodes)
    stats = PPOTrainer(policy, cfg).update(buf)
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "total_loss", "entropy_coef"}


def test_entropy_coef_anneals_start_to_final() -> None:
    """current_entropy_coef linearly decays entropy_coef -> entropy_coef_final over the budget."""
    import dataclasses

    from src.rl.gnn_encoder import GNNEncoder
    from src.rl.policy import TwoHeadPolicy
    from src.rl.ppo_trainer import PPOTrainer
    from src.utils.config import load_config

    cfg = dataclasses.replace(
        load_config("config.yaml"), entropy_coef=0.1, entropy_coef_final=0.02, total_updates=11
    )
    tr = PPOTrainer(TwoHeadPolicy(GNNEncoder(hidden=8, layers=2), hidden=8), cfg)
    assert tr.current_entropy_coef() == 0.1  # update 0 -> start
    tr._global_update = 10  # last update (total_updates-1) -> final
    assert abs(tr.current_entropy_coef() - 0.02) < 1e-9
    tr._global_update = 5  # midpoint -> halfway
    assert abs(tr.current_entropy_coef() - 0.06) < 1e-9


def test_constant_entropy_when_no_final() -> None:
    """Absent entropy_coef_final (== entropy_coef) => no annealing (constant)."""
    from src.rl.gnn_encoder import GNNEncoder
    from src.rl.policy import TwoHeadPolicy
    from src.rl.ppo_trainer import PPOTrainer
    from src.utils.config import load_config

    cfg = load_config("config.yaml")  # entropy_coef_final defaults to entropy_coef
    tr = PPOTrainer(TwoHeadPolicy(GNNEncoder(hidden=8, layers=2), hidden=8), cfg)
    first = tr.current_entropy_coef()
    tr._global_update = cfg.total_updates - 1
    assert tr.current_entropy_coef() == first  # constant across the run
