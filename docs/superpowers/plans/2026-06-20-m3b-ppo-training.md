# M3b — Custom PPO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the M3a two-head policy with a from-scratch PPO Actor-Critic — rollout buffer, GAE, clipped surrogate, value loss, entropy bonus, advantage normalization, Adam, checkpoints — and clear the learning sanity gates (reward rises, beats random, eval-vs-HEFT improves).

**Architecture:** A `RolloutBuffer` stores per-step transitions as `ObsTensors` snapshots (never the live `Observation`) and computes GAE advantages/returns with γ=1, λ≈0.95. A `PPOTrainer` collects policy-driven rollouts through `ClusterEnv`, then runs K PPO epochs minimizing the clipped surrogate + value loss − entropy bonus, accumulating gradients per-transition (graph sizes vary, so no fixed batching). An `RLStrategy` adapter exposes the trained policy through the M2 `predict` interface so eval-vs-HEFT reuses the fairness harness. Hyperparameters live in `config.yaml`.

**Tech Stack:** Python 3.10+, PyTorch (CPU), PyTorch Geometric, numpy, pytest, ruff, black.

## Global Constraints

Copied from `SmartDAG_Scheduler_TZ.md` §6.5 + §9 + Appendix A and the roadmap M3b section:

- **Python 3.10+**; full type hints; `ruff` + `black` clean; tooling via `.venv/bin/...`.
- **γ = 1.0** (pinned constant in the trainer — required so the dense per-step reward telescopes; spec §2/§6.4). λ (GAE) ≈ 0.95.
- **PPO Actor-Critic only:** rollout buffer, GAE, clipped surrogate (ε≈0.2), value loss, entropy bonus, advantage normalization, Adam. **No DQN, no PER, no target network.**
- **Pointer scoring + masking** (from M3a) — never a fixed `Discrete`. Variable candidate-set sizes ⇒ per-transition forward + gradient accumulation (no dense batching in M3b; PyG `Batch` is a later optimisation).
- **Rollout buffer tensorises per-node features at the decision point** — it stores `ObsTensors` snapshots, never the live `Observation` (whose `.nodes` mutates as the env steps).
- **Device-aware:** action-index tensors created with `device=` (dev is CPU now; do not hardcode CPU).
- **Sanity gates (§9):** beats Random fast; reward rises then plateaus; eval-vs-HEFT improves. A flat reward from the start is a bug (reward scaling / masking / advantage), not "needs more training."
- **Checkpoints** saved to `models/*.pth` (the `models/` dir is git-ignored).
- **Non-goals:** DQN/PER/target net; CLI entry point (that is M5); WfCommons; dense PyG batching; failure/noise handling (M4). Curriculum is optional and not built here.

### M1/M2/M3a interfaces consumed (already on `main`)

- `src.rl.policy.TwoHeadPolicy`: `encode(t)`, `task_logits(h,ctx,ready_mask)`, `node_logits(h_tau,n_emb,ctx,alive_mask)`, `value(ctx)`, `act(obs)->((task,node),log_prob,value)`, `evaluate_action(obs,task,node)->(log_prob,entropy,value)`. **This plan adds `act_from_tensors`, `evaluate_tensors`, `act_greedy` (Task 2).**
- `src.rl.gnn_encoder.GNNEncoder(task_in=15, node_in=9, hidden, layers)`.
- `src.rl.obs_tensors.{ObsTensors, obs_to_tensors}`; `ObsTensors` fields: `task_features`, `node_features`, `edge_index`, `globals`, `ready_mask`, `alive_mask`.
- `src.env.cluster_env.ClusterEnv`: `reset(dag=None, nodes=None)->(obs, info)`, `step((task,node))->(obs, reward, done, info)`, `.state`, `.scheduled`, `.schedule`.
- `src.env.observation.build_observation(state, scheduled, current_makespan)->Observation`.
- `src.env.placement.horizon(nodes)->float`.
- `src.core.dag.TaskDAG.ready_set(scheduled:set[int])->list[int]`.
- `src.strategies.base.BaseSchedulingStrategy` (`predict(ready, state)->(task,node)`); `src.strategies.heft.HEFTStrategy`; `src.strategies.random_strategy.RandomStrategy(rng)`; `src.scheduler.task_scheduler.run_episode(env, strategy, dag, nodes)->(schedule, info)`.
- `src.utils.config.{Config, load_config}`; `src.utils.seeding.make_rng`.

---

### Task 1: Config — PPO hyperparameters

**Files:**
- Modify: `config.yaml`
- Modify: `src/utils/config.py`
- Test: `tests/test_config_rl.py`

**Interfaces:**
- Consumes: existing `Config`/`load_config`.
- Produces: `Config` gains fields `lr: float`, `clip_eps: float`, `gae_lambda: float`, `ppo_epochs: int`, `minibatch_size: int`, `entropy_coef: float`, `value_coef: float`, `rollout_episodes: int`, `total_updates: int`, `max_grad_norm: float`, `gnn_hidden: int`, `gnn_layers: int`; `load_config` parses them.

- [ ] **Step 1: Write the failing test**

`tests/test_config_rl.py`:
```python
from src.utils.config import load_config


def test_rl_hyperparameters_loaded() -> None:
    cfg = load_config("config.yaml")
    assert cfg.lr == 0.0003
    assert cfg.clip_eps == 0.2
    assert cfg.gae_lambda == 0.95
    assert cfg.ppo_epochs == 4
    assert cfg.minibatch_size == 32
    assert cfg.entropy_coef == 0.01
    assert cfg.value_coef == 0.5
    assert cfg.rollout_episodes == 4
    assert cfg.total_updates == 50
    assert cfg.max_grad_norm == 0.5
    assert cfg.gnn_hidden == 64
    assert cfg.gnn_layers == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config_rl.py -v`
Expected: FAIL — `Config` has no attribute `lr`.

- [ ] **Step 3: Append RL keys to `config.yaml`**

```yaml
# RL / PPO (M3b). gamma is pinned to 1.0 in the trainer (telescoping reward).
lr: 0.0003
clip_eps: 0.2
gae_lambda: 0.95
ppo_epochs: 4
minibatch_size: 32
entropy_coef: 0.01
value_coef: 0.5
rollout_episodes: 4
total_updates: 50
max_grad_norm: 0.5
gnn_hidden: 64
gnn_layers: 2
```

- [ ] **Step 4: Extend `Config` and `load_config` in `src/utils/config.py`**

Add these fields to the `Config` dataclass (after `failure_rate`):
```python
    lr: float
    clip_eps: float
    gae_lambda: float
    ppo_epochs: int
    minibatch_size: int
    entropy_coef: float
    value_coef: float
    rollout_episodes: int
    total_updates: int
    max_grad_norm: float
    gnn_hidden: int
    gnn_layers: int
```
Add these to the `Config(...)` construction in `load_config` (after `failure_rate=...`):
```python
        lr=float(raw["lr"]),
        clip_eps=float(raw["clip_eps"]),
        gae_lambda=float(raw["gae_lambda"]),
        ppo_epochs=int(raw["ppo_epochs"]),
        minibatch_size=int(raw["minibatch_size"]),
        entropy_coef=float(raw["entropy_coef"]),
        value_coef=float(raw["value_coef"]),
        rollout_episodes=int(raw["rollout_episodes"]),
        total_updates=int(raw["total_updates"]),
        max_grad_norm=float(raw["max_grad_norm"]),
        gnn_hidden=int(raw["gnn_hidden"]),
        gnn_layers=int(raw["gnn_layers"]),
```

- [ ] **Step 5: Run the new test + full suite**

Run: `.venv/bin/pytest tests/test_config_rl.py -v && .venv/bin/pytest -q`
Expected: new test passes; full suite still green (existing `test_utils` still passes — it only reads the original keys, which still exist).

- [ ] **Step 6: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: PPO hyperparameters in config (M3b)"
```

---

### Task 2: Policy — tensor-level act/evaluate + greedy

**Files:**
- Modify: `src/rl/policy.py`
- Test: `tests/test_policy_tensors.py`

**Interfaces:**
- Consumes: existing `TwoHeadPolicy` internals.
- Produces, added to `TwoHeadPolicy`:
  - `act_from_tensors(self, t: ObsTensors) -> tuple[tuple[int, int], Tensor, Tensor]` — same as `act` but takes a pre-built `ObsTensors` (so the buffer can store and reuse the snapshot).
  - `evaluate_tensors(self, t: ObsTensors, task_id: int, node_id: int) -> tuple[Tensor, Tensor, Tensor]` — `(log_prob, entropy, value)`; index tensors created with `device=t.task_features.device`.
  - `act_greedy(self, obs: Observation) -> tuple[int, int]` — deterministic argmax over masked task logits then masked node logits.
  - `act` and `evaluate_action` refactored to delegate to the tensor-level methods (behaviour unchanged).

- [ ] **Step 1: Write the failing test**

`tests/test_policy_tensors.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_policy_tensors.py -v`
Expected: FAIL — `TwoHeadPolicy` has no attribute `act_from_tensors`.

- [ ] **Step 3: Refactor `src/rl/policy.py`**

Replace the `act` and `evaluate_action` methods with the following (and add the three new methods):
```python
    def act_from_tensors(self, t: ObsTensors) -> tuple[tuple[int, int], Tensor, Tensor]:
        h, n_emb, ctx = self.encode(t)
        task_dist = Categorical(logits=self.task_logits(h, ctx, t.ready_mask))
        task_id = task_dist.sample()
        node_dist = Categorical(logits=self.node_logits(h[task_id], n_emb, ctx, t.alive_mask))
        node_id = node_dist.sample()
        log_prob = task_dist.log_prob(task_id) + node_dist.log_prob(node_id)
        return (int(task_id), int(node_id)), log_prob, self.value(ctx)

    def act(self, obs: Observation) -> tuple[tuple[int, int], Tensor, Tensor]:
        return self.act_from_tensors(obs_to_tensors(obs))

    def evaluate_tensors(
        self, t: ObsTensors, task_id: int, node_id: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        h, n_emb, ctx = self.encode(t)
        device = t.task_features.device
        task_dist = Categorical(logits=self.task_logits(h, ctx, t.ready_mask))
        node_dist = Categorical(logits=self.node_logits(h[task_id], n_emb, ctx, t.alive_mask))
        log_prob = task_dist.log_prob(
            torch.tensor(task_id, device=device)
        ) + node_dist.log_prob(torch.tensor(node_id, device=device))
        entropy = task_dist.entropy() + node_dist.entropy()
        return log_prob, entropy, self.value(ctx)

    def evaluate_action(
        self, obs: Observation, task_id: int, node_id: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.evaluate_tensors(obs_to_tensors(obs), task_id, node_id)

    def act_greedy(self, obs: Observation) -> tuple[int, int]:
        t = obs_to_tensors(obs)
        h, n_emb, ctx = self.encode(t)
        task_id = int(self.task_logits(h, ctx, t.ready_mask).argmax())
        node_id = int(self.node_logits(h[task_id], n_emb, ctx, t.alive_mask).argmax())
        return task_id, node_id
```

- [ ] **Step 4: Run the new test + full suite**

Run: `.venv/bin/pytest tests/test_policy_tensors.py tests/test_policy_action.py tests/test_policy_masking.py -v && .venv/bin/pytest -q`
Expected: new tests pass; the existing M3a policy tests still pass (act/evaluate_action behaviour is unchanged); full suite green.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: tensor-level act/evaluate + greedy on the policy"
```

---

### Task 3: Rollout buffer + GAE

**Files:**
- Create: `src/rl/rollout_buffer.py`
- Test: `tests/test_rollout_buffer.py`

**Interfaces:**
- Consumes: `ObsTensors`.
- Produces:
  - `rollout_buffer.Transition` (dataclass): `obs: ObsTensors`, `task_id: int`, `node_id: int`, `log_prob: float`, `value: float`, `reward: float`, `done: bool`.
  - `rollout_buffer.RolloutBuffer`:
    - `add(self, obs: ObsTensors, task_id: int, node_id: int, log_prob: float, value: float, reward: float, done: bool) -> None`
    - `__len__`
    - `transitions: list[Transition]`
    - `advantages: list[float] | None`, `returns: list[float] | None` (filled by `compute_gae`)
    - `compute_gae(self, gamma: float = 1.0, lam: float = 0.95) -> None` — backward GAE with per-episode `done` masking.
    - `clear(self) -> None`

- [ ] **Step 1: Write the failing test**

`tests/test_rollout_buffer.py`:
```python
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


def test_clear() -> None:
    buf = RolloutBuffer()
    buf.add(_stub_obs(), 0, 0, log_prob=0.0, value=0.0, reward=0.0, done=True)
    buf.clear()
    assert len(buf) == 0 and buf.advantages is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_rollout_buffer.py -v`
Expected: FAIL — `ModuleNotFoundError` for `src.rl.rollout_buffer`.

- [ ] **Step 3: Implement `src/rl/rollout_buffer.py`**

```python
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
        self.transitions.append(
            Transition(obs, task_id, node_id, log_prob, value, reward, done)
        )

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_rollout_buffer.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: rollout buffer with GAE (gamma=1)"
```

---

### Task 4: PPO update step

**Files:**
- Create: `src/rl/ppo_trainer.py`
- Test: `tests/test_ppo_update.py`

**Interfaces:**
- Consumes: `TwoHeadPolicy.evaluate_tensors`, `RolloutBuffer`, `Config`, `torch.optim.Adam`, `torch.nn.utils.clip_grad_norm_`.
- Produces:
  - `ppo_trainer.GAMMA = 1.0` (module constant).
  - `ppo_trainer.PPOTrainer(policy: TwoHeadPolicy, config: Config)` with `self.optimizer = Adam(policy.parameters(), lr=config.lr)`.
  - `PPOTrainer.update(self, buffer: RolloutBuffer) -> dict[str, float]` — requires `buffer.compute_gae` already called; normalizes advantages; runs `config.ppo_epochs` epochs over shuffled minibatches of size `config.minibatch_size`; per transition recomputes `(log_prob, entropy, value)` via `evaluate_tensors`, computes clipped surrogate (`config.clip_eps`), value loss, entropy bonus (`config.entropy_coef`, `config.value_coef`); Adam step with grad clip (`config.max_grad_norm`). Returns mean `{"policy_loss", "value_loss", "entropy", "total_loss"}`.

- [ ] **Step 1: Write the failing test**

`tests/test_ppo_update.py`:
```python
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
from src.utils.config import load_config


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


def _collect_one_episode(policy: TwoHeadPolicy, env: ClusterEnv, dag, nodes) -> RolloutBuffer:
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


def _setup():
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
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "total_loss"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ppo_update.py -v`
Expected: FAIL — `ModuleNotFoundError` for `src.rl.ppo_trainer`.

- [ ] **Step 3: Implement `src/rl/ppo_trainer.py` (update step + ctor)**

```python
"""From-scratch PPO Actor-Critic trainer (TZ §6.5, §9). gamma pinned to 1.0."""

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.rl.policy import TwoHeadPolicy
from src.rl.rollout_buffer import RolloutBuffer
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
                totals["policy_loss"] += float(policy_loss)
                totals["value_loss"] += float(value_loss)
                totals["entropy"] += float(entropy_mean)
                totals["total_loss"] += float(total)
                n_batches += 1
        return {k: v / n_batches for k, v in totals.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ppo_update.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: PPO clipped-surrogate update step"
```

---

### Task 5: Rollout collection + RLStrategy + checkpoints

**Files:**
- Modify: `src/rl/ppo_trainer.py` (add `collect_rollouts`, `save_checkpoint`, `load_checkpoint`)
- Create: `src/rl/rl_strategy.py`
- Test: `tests/test_rollout_collection.py`
- Test: `tests/test_rl_strategy.py`

**Interfaces:**
- Consumes: `ClusterEnv`, `policy.act_from_tensors`/`act_greedy`, `RolloutBuffer`, `obs_to_tensors`, `build_observation`, `horizon`, `BaseSchedulingStrategy`, `run_episode`.
- Produces:
  - `PPOTrainer.collect_rollouts(self, env: ClusterEnv, n_episodes: int, dag=None, nodes=None) -> RolloutBuffer` — runs `n_episodes` policy-driven episodes (sampling via `act_from_tensors` under `torch.no_grad()`), stores transitions, then calls `compute_gae(GAMMA, config.gae_lambda)`.
  - `PPOTrainer.save_checkpoint(self, path: str) -> None` / `PPOTrainer.load_checkpoint(self, path: str) -> None` (policy `state_dict`; creates parent dir).
  - `rl_strategy.RLStrategy(policy: TwoHeadPolicy)` implementing `BaseSchedulingStrategy.predict(ready, state) -> (task, node)` via `act_greedy` on an observation rebuilt from the live state.

- [ ] **Step 1: Write the failing tests**

`tests/test_rollout_collection.py`:
```python
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
```

`tests/test_rl_strategy.py`:
```python
import torch

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.rl.gnn_encoder import GNNEncoder
from src.rl.policy import TwoHeadPolicy
from src.rl.rl_strategy import RLStrategy
from src.scheduler.task_scheduler import run_episode
from src.utils.config import load_config


def _instance():
    tasks = [Task(i, 2.0, 1.0, TaskClass.SEQUENTIAL) for i in range(4)]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_rl_strategy_plugs_into_run_episode() -> None:
    torch.manual_seed(0)
    policy = TwoHeadPolicy(GNNEncoder(hidden=16, layers=2), hidden=16)
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    schedule, info = run_episode(env, RLStrategy(policy), dag=dag, nodes=nodes)
    assert sorted(a.task_id for a in schedule.assignments) == [0, 1, 2, 3]
    assert info["makespan"] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_rollout_collection.py tests/test_rl_strategy.py -v`
Expected: FAIL — `PPOTrainer` has no `collect_rollouts`; `src.rl.rl_strategy` missing.

- [ ] **Step 3: Add collection + checkpoints to `src/rl/ppo_trainer.py`**

Add to the imports:
```python
import os

from src.env.cluster_env import ClusterEnv
from src.rl.obs_tensors import obs_to_tensors
```
Add these methods to `PPOTrainer`:
```python
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

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path))
```

- [ ] **Step 4: Implement `src/rl/rl_strategy.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_rollout_collection.py tests/test_rl_strategy.py -v`
Expected: collection (2 passed) + rl_strategy (1 passed).

- [ ] **Step 6: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: rollout collection, checkpoints, RLStrategy adapter"
```

---

### Task 6: Training loop + eval-vs-HEFT + learning sanity gate

**Files:**
- Modify: `src/rl/ppo_trainer.py` (add `train`, `evaluate_vs_heft`)
- Test: `tests/test_ppo_training.py`

**Interfaces:**
- Consumes: `collect_rollouts`, `update`, `RLStrategy`, `HEFTStrategy`, `run_episode`, `make_rng`.
- Produces:
  - `PPOTrainer.train(self, env: ClusterEnv, n_updates: int, dag=None, nodes=None) -> list[dict[str, float]]` — for each update: collect `config.rollout_episodes` episodes, `update`, append stats (with a `"mean_reward"` = mean episode reward over the collected rollouts). Returns the per-update history.
  - `PPOTrainer.evaluate_vs_heft(self, env: ClusterEnv, instances: list[tuple]) -> dict[str, float]` — runs the greedy `RLStrategy` and `HEFTStrategy` via `run_episode` on each `(dag, nodes)` instance; returns `{"rl_makespan", "heft_makespan"}` (means).

- [ ] **Step 1: Write the failing test**

`tests/test_ppo_training.py`:
```python
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
                (a, _lp, _v) = policy.act(obs)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ppo_training.py -v`
Expected: FAIL — `PPOTrainer` has no `train`.

- [ ] **Step 3: Add `train` + `evaluate_vs_heft` to `src/rl/ppo_trainer.py`**

Add to the imports:
```python
from src.scheduler.task_scheduler import run_episode
from src.strategies.heft import HEFTStrategy
from src.rl.rl_strategy import RLStrategy
```
Add these methods to `PPOTrainer`:
```python
    def train(
        self,
        env: ClusterEnv,
        n_updates: int,
        dag=None,
        nodes=None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for _ in range(n_updates):
            buffer = self.collect_rollouts(
                env, self.config.rollout_episodes, dag=dag, nodes=nodes
            )
            n_episodes = sum(1 for tr in buffer.transitions if tr.done)
            total_reward = sum(tr.reward for tr in buffer.transitions)
            stats = self.update(buffer)
            stats["mean_reward"] = total_reward / max(1, n_episodes)
            history.append(stats)
        return history

    def evaluate_vs_heft(
        self, env: ClusterEnv, instances: list[tuple]
    ) -> dict[str, float]:
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
```
(If `ruff` flags import ordering, move `from src.rl.rl_strategy import RLStrategy` into alphabetical position among the `src.rl` imports.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ppo_training.py -v`
Expected: 2 passed. If `test_training_improves_reward_on_tiny_instance` is flaky (PPO variance), first confirm `before`/`after` printed values trend upward; if genuinely borderline, increase `n_updates` to 80 — do **not** weaken the `after > before` assertion (a non-improving reward is the bug the gate exists to catch). Report the values seen.

- [ ] **Step 5: Run the full suite + lint/format**

Run:
```bash
.venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/black --check .
```
Expected: all tests pass (M1+M2+M3a+M3b); ruff clean; black reports no changes.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: PPO training loop + eval-vs-HEFT + learning sanity gate"
```

---

## Self-Review

**Spec coverage (roadmap M3b + TZ §6.5 + §9):**
- Rollout buffer, GAE (λ≈0.95), γ=1 telescoping — Task 3. ✅
- Clipped surrogate (ε≈0.2), value loss, entropy bonus, advantage normalization, Adam — Task 4. ✅
- Per-transition forward for variable graph sizes (no dense batching) — Task 4 (minibatch loop accumulates per-transition). ✅
- Rollout collection driving env with policy — Task 5. ✅
- Buffer stores `ObsTensors` snapshots, not the live `Observation` — Task 3 (`Transition.obs: ObsTensors`), Task 5 (collection stores `t`). ✅
- Device-aware action tensors — Task 2 (`evaluate_tensors` uses `device=`). ✅
- Greedy act + RLStrategy adapter (RL agent через `predict`) — Tasks 2 + 5. ✅
- Eval-vs-HEFT (frozen policy on a fixed set, §9) — Task 6 (`evaluate_vs_heft`). ✅
- Training budget loop — Task 6 (`train(n_updates)`). ✅
- Checkpoint save/load to `models/*.pth` — Task 5. ✅
- Config hyperparameters — Task 1. ✅
- Sanity gate (reward rises; flat ⇒ bug) — Task 6 (`test_training_improves_reward_on_tiny_instance` asserts `after > before`). ✅
- PPO-only, no DQN/PER/target net — none present. ✅

**Placeholder scan:** no TBD/TODO; every code step has complete code. Early-stop on plateau (§9) is operationalised as a fixed `total_updates` budget here (the simplest budget); plateau-detection is not required for M3b correctness and is intentionally omitted (YAGNI) — noted, not a placeholder.

**Type consistency:** `Transition.obs: ObsTensors` matches what `collect_rollouts` stores (`t = obs_to_tensors(obs)`) and what `evaluate_tensors(t, ...)` consumes. `compute_gae(gamma, lam)` signature matches its callers (buffer test, `collect_rollouts`). `PPOTrainer(policy, config)` ctor, `update(buffer)->dict`, `collect_rollouts(env, n_episodes, dag, nodes)->RolloutBuffer`, `train(env, n_updates, dag, nodes)->list[dict]`, `evaluate_vs_heft(env, instances)->dict`, `save/load_checkpoint(path)` are consistent across tasks and tests. `RLStrategy.predict(ready, state)` matches `BaseSchedulingStrategy` and `run_episode`'s call. `act_from_tensors`/`evaluate_tensors`/`act_greedy` signatures match Tasks 4–6 usage.

**Flakiness note on the learning gate:** the gate trains on a single tiny fixed instance with seeded torch+numpy for 60 updates and asserts mean sampled reward rises. This is the operational "reward rises / not flat" check from §9. It is bounded (small DAG, hidden=16) to run in seconds. The instance has a genuine makespan↔energy trade-off so there is something to learn. The instruction to bump updates (not weaken the assertion) if borderline preserves the gate's diagnostic value.

**Deferred (consistent with roadmap):** dense PyG `Batch` minibatching (perf); plateau-based early stop; curriculum; noise/failures (M4); CLI/experiment runner + WfCommons + full regime grid (M5).
