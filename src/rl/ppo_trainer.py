"""From-scratch PPO Actor-Critic trainer (TZ §6.5, §9). gamma pinned to 1.0."""

import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.env.cluster_env import ClusterEnv
from src.rl.obs_tensors import obs_to_tensors
from src.rl.policy import TwoHeadPolicy
from src.rl.rl_strategy import RLStrategy
from src.rl.rollout_buffer import RolloutBuffer
from src.scheduler.task_scheduler import run_episode
from src.strategies.heft import HEFTStrategy
from src.utils.config import Config

GAMMA = 1.0


class PPOTrainer:
    def __init__(self, policy: TwoHeadPolicy, config: Config) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = Adam(policy.parameters(), lr=config.lr)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        if buffer.advantages is None or buffer.returns is None:
            raise ValueError("Call buffer.compute_gae() before update().")
        cfg = self.config
        n = len(buffer)
        advantages = torch.tensor(buffer.advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = torch.tensor(buffer.returns, dtype=torch.float32)
        old_log_probs = torch.tensor([tr.log_prob for tr in buffer.transitions])

        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        n_batches = 0
        for _epoch in range(cfg.ppo_epochs):
            order = torch.randperm(n)
            for start in range(0, n, cfg.minibatch_size):
                idx = order[start : start + cfg.minibatch_size]
                self.optimizer.zero_grad()
                policy_loss = torch.zeros(())
                value_loss = torch.zeros(())
                entropy_sum = torch.zeros(())
                for i in idx.tolist():
                    tr = buffer.transitions[i]
                    log_prob, entropy, value = self.policy.evaluate_tensors(
                        tr.obs, tr.task_id, tr.node_id
                    )
                    ratio = torch.exp(log_prob - old_log_probs[i])
                    adv = advantages[i]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
                    policy_loss = policy_loss - torch.min(surr1, surr2)
                    value_loss = value_loss + (value - returns[i]) ** 2
                    entropy_sum = entropy_sum + entropy
                mb = len(idx)
                policy_loss = policy_loss / mb
                value_loss = value_loss / mb
                entropy_mean = entropy_sum / mb
                total = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_mean
                total.backward()
                clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy_mean.item()
                totals["total_loss"] += total.item()
                n_batches += 1
        return {k: v / n_batches for k, v in totals.items()}

    def collect_rollouts(
        self,
        env: ClusterEnv,
        n_episodes: int,
        dag=None,
        nodes=None,
    ) -> RolloutBuffer:
        buffer = RolloutBuffer()
        with torch.no_grad():
            for _ in range(n_episodes):
                obs, _info = env.reset(dag=dag, nodes=nodes)
                done = False
                while not done:
                    t = obs_to_tensors(obs)
                    (task_id, node_id), log_prob, value = self.policy.act_from_tensors(t)
                    obs, reward, done, _info = env.step((task_id, node_id))
                    buffer.add(t, task_id, node_id, log_prob.item(), value.item(), reward, done)
        buffer.compute_gae(gamma=GAMMA, lam=self.config.gae_lambda)
        return buffer

    def train(
        self,
        env: ClusterEnv,
        n_updates: int,
        dag=None,
        nodes=None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for _ in range(n_updates):
            buffer = self.collect_rollouts(env, self.config.rollout_episodes, dag=dag, nodes=nodes)
            n_episodes = sum(1 for tr in buffer.transitions if tr.done)
            total_reward = sum(tr.reward for tr in buffer.transitions)
            stats = self.update(buffer)
            stats["mean_reward"] = total_reward / max(1, n_episodes)
            history.append(stats)
        return history

    def evaluate_vs_heft(self, env: ClusterEnv, instances: list[tuple]) -> dict[str, float]:
        rl_makespans: list[float] = []
        heft_makespans: list[float] = []
        rl_strategy = RLStrategy(self.policy)
        for dag, nodes in instances:
            _s, rl_info = run_episode(env, rl_strategy, dag=dag, nodes=nodes)
            _s, heft_info = run_episode(env, HEFTStrategy(), dag=dag, nodes=nodes)
            rl_makespans.append(rl_info["makespan"])
            heft_makespans.append(heft_info["makespan"])
        return {
            "rl_makespan": sum(rl_makespans) / len(rl_makespans),
            "heft_makespan": sum(heft_makespans) / len(heft_makespans),
        }

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, weights_only=True))
