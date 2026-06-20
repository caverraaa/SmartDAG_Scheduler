import pytest
import torch

from src.rl.obs_tensors import ObsTensors
from src.rl.rollout_buffer import RolloutBuffer


def _stub_obs() -> ObsTensors:
    return ObsTensors(
        task_features=torch.zeros(1, 15),
        node_features=torch.zeros(1, 9),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        globals=torch.zeros(2),
        ready_mask=torch.ones(1, dtype=torch.bool),
        alive_mask=torch.ones(1, dtype=torch.bool),
    )


def test_add_and_len() -> None:
    buf = RolloutBuffer()
    buf.add(_stub_obs(), 0, 0, log_prob=-0.5, value=1.0, reward=2.0, done=False)
    buf.add(_stub_obs(), 1, 0, log_prob=-0.7, value=0.5, reward=3.0, done=True)
    assert len(buf) == 2


def test_gae_single_episode_hand_computed() -> None:
    # Two-step episode, gamma=1, lam=0.95. V=[v0,v1], r=[r0,r1], done=[F,T].
    buf = RolloutBuffer()
    v0, v1, r0, r1 = 1.0, 0.5, 2.0, 3.0
    buf.add(_stub_obs(), 0, 0, log_prob=-0.5, value=v0, reward=r0, done=False)
    buf.add(_stub_obs(), 1, 0, log_prob=-0.7, value=v1, reward=r1, done=True)
    buf.compute_gae(gamma=1.0, lam=0.95)
    # adv1 = r1 - v1 ; adv0 = (r0 + v1 - v0) + 0.95*adv1
    adv1 = r1 - v1
    adv0 = (r0 + v1 - v0) + 0.95 * adv1
    assert abs(buf.advantages[1] - adv1) < 1e-6
    assert abs(buf.advantages[0] - adv0) < 1e-6
    # returns = advantage + value
    assert abs(buf.returns[1] - (adv1 + v1)) < 1e-6
    assert abs(buf.returns[0] - (adv0 + v0)) < 1e-6


def test_gae_resets_across_episode_boundary() -> None:
    # Two 1-step episodes: each done=True, so each advantage = reward - value (no carry).
    buf = RolloutBuffer()
    buf.add(_stub_obs(), 0, 0, log_prob=-0.5, value=1.0, reward=2.0, done=True)
    buf.add(_stub_obs(), 0, 0, log_prob=-0.5, value=0.0, reward=5.0, done=True)
    buf.compute_gae(gamma=1.0, lam=0.95)
    assert abs(buf.advantages[0] - (2.0 - 1.0)) < 1e-6
    assert abs(buf.advantages[1] - (5.0 - 0.0)) < 1e-6
    # returns = advantage + value
    assert abs(buf.returns[0] - 2.0) < 1e-6  # (2.0 - 1.0) + 1.0 = 2.0
    assert abs(buf.returns[1] - 5.0) < 1e-6  # (5.0 - 0.0) + 0.0 = 5.0


def test_clear() -> None:
    buf = RolloutBuffer()
    buf.add(_stub_obs(), 0, 0, log_prob=0.0, value=0.0, reward=0.0, done=True)
    buf.clear()
    assert len(buf) == 0 and buf.advantages is None and buf.returns is None


def test_compute_gae_requires_terminal_last_transition() -> None:
    # compute_gae must not be called with a non-terminal last transition.
    buf = RolloutBuffer()
    buf.add(_stub_obs(), 0, 0, log_prob=-0.5, value=1.0, reward=2.0, done=False)
    with pytest.raises(ValueError, match="the last transition must have done=True"):
        buf.compute_gae()
