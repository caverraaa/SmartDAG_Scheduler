"""Adapter exposing a trained policy through the BaseSchedulingStrategy interface.

Lets the RL agent run inside the M2 scheduler loop (run_episode) for eval-vs-HEFT
and fair comparison. Uses greedy (argmax) action selection.
"""

from src.env.observation import build_observation
from src.env.placement import ClusterState, horizon
from src.rl.policy import TwoHeadPolicy
from src.strategies.base import BaseSchedulingStrategy


class RLStrategy(BaseSchedulingStrategy):
    def __init__(self, policy: TwoHeadPolicy) -> None:
        self._policy = policy

    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        scheduled = set(state.task_finish.keys())
        obs = build_observation(state, scheduled, current_makespan=horizon(state.nodes))
        return self._policy.act_greedy(obs)
