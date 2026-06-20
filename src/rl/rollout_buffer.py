"""Rollout buffer with GAE (TZ §6.5). gamma=1 keeps the telescoping reward exact.

Stores ObsTensors snapshots at the decision point — never the live Observation
(whose .nodes mutates as the env advances).
"""

from dataclasses import dataclass

from src.rl.obs_tensors import ObsTensors


@dataclass
class Transition:
    obs: ObsTensors
    task_id: int
    node_id: int
    log_prob: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    def __init__(self) -> None:
        self.transitions: list[Transition] = []
        self.advantages: list[float] | None = None
        self.returns: list[float] | None = None

    def add(
        self,
        obs: ObsTensors,
        task_id: int,
        node_id: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.transitions.append(Transition(obs, task_id, node_id, log_prob, value, reward, done))

    def __len__(self) -> int:
        return len(self.transitions)

    def compute_gae(self, gamma: float = 1.0, lam: float = 0.95) -> None:
        n = len(self.transitions)
        advantages = [0.0] * n
        adv = 0.0
        for t in reversed(range(n)):
            tr = self.transitions[t]
            nonterminal = 0.0 if tr.done else 1.0
            next_value = 0.0 if tr.done else self.transitions[t + 1].value
            delta = tr.reward + gamma * nonterminal * next_value - tr.value
            adv = delta + gamma * lam * nonterminal * adv
            advantages[t] = adv
        self.advantages = advantages
        self.returns = [advantages[t] + self.transitions[t].value for t in range(n)]

    def clear(self) -> None:
        self.transitions = []
        self.advantages = None
        self.returns = None
