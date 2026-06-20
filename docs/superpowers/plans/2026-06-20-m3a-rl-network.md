# M3a — RL Network (Encoder + Two-Head Policy, No Training) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GraphSAGE encoder and the two-head autoregressive policy+critic network — fully unit-tested for shapes, masking, joint log-prob decomposition, and gradient flow — with **no training loop** (that is M3b).

**Architecture:** A bidirectional GraphSAGE encoder (torch_geometric) over the DAG produces per-task embeddings `h_i` and a pooled graph embedding `g`; a small MLP over per-node features produces `n_j` and a pooled `c`. The two-head autoregressive policy factorises `π(a)=π(τ|s)·π(ν|s,τ)`: Head 1 scores ready tasks (`MLP([h_i, g, c, globals])`, masked to ready), Head 2 scores alive nodes given the chosen task (`MLP([h_τ, n_j, g, c, globals])`, masked to alive). A critic scores `MLP([g, c, globals])`. Variable candidate-set sizes are handled by per-instance scoring + masking + softmax — never a fixed `Discrete`.

**Tech Stack:** Python 3.10+, PyTorch (CPU), PyTorch Geometric (PyG — **not DGL**), numpy, pytest, ruff, black. M3a introduces torch + PyG (M1/M2 had neither).

## Global Constraints

Copied from `SmartDAG_Scheduler_TZ.md` §6.1–§6.3 + Appendix A and the roadmap M3a section:

- **Python 3.10+**; full type hints; `ruff` + `black` clean; tooling via `.venv/bin/...`.
- **Encoder = GraphSAGE via torch_geometric** (NOT DGL), **bidirectional** (forward + reverse edges, direction-flagged via separate forward/reverse aggregation streams), 2–3 layers. Produces per-task `h_i` + pooled `g`; node-feature MLP → `n_j`, pooled → `c`.
- **Policy = two-head autoregressive**, factorised `π(a)=π(τ|s)·π(ν|s,τ)`. Head 1 (ordering): `score_i = MLP([h_i, globals])`, mask non-ready, softmax. Head 2 (assignment): `score_j = MLP([h_τ, n_j, globals])`, mask dead/unavailable, softmax. "globals" in the head context = `[g, c, current_makespan, fraction_done]`.
- **Critic:** `V = MLP([g, c, globals])` scalar.
- **Joint log-prob** for a `(τ, ν)` action = `logπ(τ) + logπ(ν|τ)`.
- **Pointer scoring + masking over variable candidate sets — NEVER a fixed `Discrete` action space.** Masking sets non-candidate logits to `-inf` so they receive exactly zero probability.
- **No training in M3a:** no PPO, no rollout buffer, no optimizer loop. (The rollout buffer that tensorises per-node features at the decision point is M3b — do not build it here.)
- **Resolve the M1-deferred observation normalization** now that the GNN consumes the observation: the raw continuous columns (`T_MEM`, `T_OUTDATA`, `T_OUTDEG`, `T_UNSCHED_PREDS`, `N_SPEED`, `N_POWER`) must be normalised to O(1); `N_UTIL` clamped to `[0,1]`. Already-normalised columns (costs/cp, levels/cp, one-hot flags, masks, `N_FREE_REL`) stay as-is.
- **`Observation.nodes` is a live mutable reference** — the policy/encoder must read only the numpy feature arrays + masks, never retain the `Observation` object across env steps.
- **Non-goals:** DGL; fixed `Discrete`; training loop / PPO / buffer; GPU-specific code (dev is CPU).

### M1/M2 interfaces consumed (already on `main`)

- `src.env.observation`: `Observation(task_features [N,15] float32, node_features [M,9] float32, globals [2] float32, edge_index [2,E] int64, ready_mask [N] bool, alive_mask [M] bool, nodes)`; column constants `T_*` / `N_*`; `N_TASK_FEATURES=15`, `N_NODE_FEATURES=9`; `build_observation(state, scheduled, current_makespan)`.
- `src.env.cluster_env.ClusterEnv`: `reset(dag=None, nodes=None) -> (obs, info)`, `.state`, `.scheduled`.
- Test instance helpers reuse the M1/M2 fixtures (diamond DAG + CPU/GPU nodes).

---

### Task 1: Add torch + PyG dependencies

**Files:**
- Modify: `requirements.txt`
- Test: `tests/test_torch_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `torch` and `torch_geometric` importable in `.venv`.

- [ ] **Step 1: Write the failing smoke test**

`tests/test_torch_smoke.py`:
```python
def test_torch_and_pyg_importable() -> None:
    import torch
    from torch_geometric.nn import SAGEConv

    assert torch.tensor([1.0, 2.0]).sum().item() == 3.0
    assert SAGEConv(4, 8) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_torch_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'torch'`.

- [ ] **Step 3: Install CPU torch + PyG**

Run (CPU wheels; torch first, then PyG from PyPI):
```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install torch_geometric
```
Expected: both install successfully. (This is a large download; allow several minutes.)

- [ ] **Step 4: Record deps in `requirements.txt`**

Append to `requirements.txt`:
```text
# M3a: RL network (install torch from the CPU index:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu)
torch>=2.2
torch_geometric>=2.5
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_torch_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "chore: add torch + torch_geometric (CPU) for M3a"
```

---

### Task 2: Normalize the deferred observation features

**Files:**
- Modify: `src/env/observation.py` (the `build_observation` body + module docstring)
- Test: `tests/test_observation_normalization.py`

**Interfaces:**
- Consumes: existing `build_observation`, column constants.
- Produces: a `build_observation` whose previously-raw columns are normalised to O(1). Column constants and the `Observation` shape are unchanged (so M1/M2 tests still pass).

Normalisation rules (per-instance, eps-guarded with `eps = 1e-8`):
- `T_MEM` ← `mem_required / mean(mem_required over tasks)`
- `T_OUTDATA` ← `out_data / mean(out_data over tasks)`
- `T_OUTDEG` ← `out_degree / n_tasks`
- `T_UNSCHED_PREDS` ← `#unscheduled_preds / n_tasks`
- `N_SPEED` ← `mean(speed_by_class) / mean over nodes of mean(speed_by_class)`
- `N_POWER` ← `power_w / mean(power_w over nodes)`
- `N_UTIL` ← `clamp(free_at_time / current_makespan, 0, 1)`
- Unchanged: `T_BASE_COST`, `T_COST_*`, `T_BLEVEL`, `T_TLEVEL`, all flags, masks, `N_FREE_REL`, globals.

- [ ] **Step 1: Write the failing test**

`tests/test_observation_normalization.py`:
```python
import numpy as np

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.env.observation import (
    N_POWER,
    N_SPEED,
    N_UTIL,
    T_MEM,
    T_OUTDATA,
    T_OUTDEG,
    T_UNSCHED_PREDS,
)
from src.utils.config import load_config


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [
        Task(0, 2.0, 4.0, TaskClass.SEQUENTIAL),
        Task(1, 4.0, 8.0, TaskClass.SEQUENTIAL),
        Task(2, 4.0, 2.0, TaskClass.SEQUENTIAL),
        Task(3, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 300.0, 10.0),
    ]
    return dag, nodes


def test_normalized_columns_are_order_one() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    obs, _ = env.reset(dag=dag, nodes=nodes)
    # mem mean = (4+8+2+1)/4 = 3.75 ; task0 mem 4 -> 4/3.75 ~ 1.07
    assert abs(obs.task_features[0, T_MEM] - (4.0 / 3.75)) < 1e-5
    # out-degree of task0 is 2 ; /n_tasks(4) = 0.5
    assert obs.task_features[0, T_OUTDEG] == 0.5
    # mean power = (100+300)/2 = 200 ; node1 power 300 -> 1.5
    assert abs(obs.node_features[1, N_POWER] - 1.5) < 1e-5
    # speed normalized to ~O(1): mean of N_SPEED column near 1
    assert abs(float(np.mean(obs.node_features[:, N_SPEED])) - 1.0) < 1e-5


def test_unsched_preds_normalized_and_outdata_present() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    obs, _ = env.reset(dag=dag, nodes=nodes)
    # task3 has 2 unscheduled preds at reset -> 2/4 = 0.5
    assert obs.task_features[3, T_UNSCHED_PREDS] == 0.5
    # out_data normalized: column mean ~ 1 over tasks with outgoing edges
    vals = obs.task_features[:, T_OUTDATA]
    assert vals.max() > 0.0


def test_util_clamped_unit_interval() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    obs, _ = env.reset(dag=dag, nodes=nodes)
    col = obs.node_features[:, N_UTIL]
    assert col.min() >= 0.0 and col.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_observation_normalization.py -v`
Expected: FAIL (T_OUTDEG is currently raw `2.0`, not `0.5`; N_POWER raw `300.0`, etc.).

- [ ] **Step 3: Edit `build_observation` in `src/env/observation.py`**

Update the module docstring's last sentence to state the normalisations are now applied. Then compute the instance means near the top of `build_observation` (after `mean_cost`):
```python
    mems = [dag.task(i).mem_required for i in range(n)]
    mean_mem = (sum(mems) / n) + 1e-8 if n else 1.0
    outs = [dag.out_data(i) for i in range(n)]
    mean_out = (sum(outs) / n) + 1e-8 if n else 1.0
    node_speeds = [float(np.mean(list(node.speed_by_class.values()))) for node in state.nodes]
    mean_speed = (sum(node_speeds) / m) + 1e-8 if m else 1.0
    mean_power = (sum(node.power_w for node in state.nodes) / m) + 1e-8 if m else 1.0
    denom_tasks = float(n) if n else 1.0
```
In the per-task loop replace these lines:
```python
        f[T_MEM] = task.mem_required / mean_mem
        ...
        f[T_UNSCHED_PREDS] = (
            float(sum(1 for p in dag.predecessors(tid) if p not in scheduled)) / denom_tasks
        )
        f[T_BLEVEL] = dag.b_level(tid) / cp
        f[T_TLEVEL] = dag.t_level(tid) / cp
        f[T_OUTDEG] = float(dag.out_degree(tid)) / denom_tasks
        f[T_OUTDATA] = dag.out_data(tid) / mean_out
```
In the per-node loop replace these lines:
```python
        nf[N_UTIL] = (
            min(1.0, max(0.0, node.free_at_time / current_makespan)) if current_makespan > 0 else 0.0
        )
        nf[N_POWER] = node.power_w / mean_power
        nf[N_SPEED] = float(np.mean(list(node.speed_by_class.values()))) / mean_speed
```
(Leave `T_BASE_COST`, `T_COST_*`, flags, `N_FREE_REL`, `N_TYPE_*`, `N_ALIVE`, masks, globals exactly as they are.)

- [ ] **Step 4: Run the new test + the full M1/M2 suite**

Run:
```bash
.venv/bin/pytest tests/test_observation_normalization.py -v && .venv/bin/pytest -q
```
Expected: new tests pass; full suite still green (the M1/M2 tests assert only flags/masks/shapes, which are unchanged).

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: normalize deferred observation features for the GNN (M3a)"
```

---

### Task 3: `rl` package + observation→tensor adapter

**Files:**
- Create: `src/rl/__init__.py` (empty)
- Create: `src/rl/obs_tensors.py`
- Test: `tests/test_obs_tensors.py`

**Interfaces:**
- Consumes: `src.env.observation.Observation`, `torch`.
- Produces:
  - `obs_tensors.ObsTensors` (dataclass): `task_features: Tensor [N,15] float32`, `node_features: Tensor [M,9] float32`, `edge_index: Tensor [2,E] int64`, `globals: Tensor [2] float32`, `ready_mask: Tensor [N] bool`, `alive_mask: Tensor [M] bool`.
  - `obs_tensors.obs_to_tensors(obs: Observation) -> ObsTensors` — pure conversion (normalisation already done upstream in `build_observation`).

- [ ] **Step 1: Create `src/rl/__init__.py`** (empty file)

- [ ] **Step 2: Write the failing test**

`tests/test_obs_tensors.py`:
```python
import torch

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.rl.obs_tensors import obs_to_tensors
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


def test_obs_to_tensors_shapes_and_dtypes() -> None:
    t = obs_to_tensors(_obs())
    assert t.task_features.shape == (4, 15) and t.task_features.dtype == torch.float32
    assert t.node_features.shape == (2, 9) and t.node_features.dtype == torch.float32
    assert t.edge_index.shape == (2, 4) and t.edge_index.dtype == torch.int64
    assert t.globals.shape == (2,)
    assert t.ready_mask.dtype == torch.bool and t.ready_mask.shape == (4,)
    assert t.alive_mask.dtype == torch.bool and t.alive_mask.shape == (2,)


def test_ready_mask_matches_observation() -> None:
    obs = _obs()
    t = obs_to_tensors(obs)
    assert t.ready_mask.tolist() == obs.ready_mask.tolist()
    assert t.alive_mask.tolist() == obs.alive_mask.tolist()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_obs_tensors.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.rl.obs_tensors`.

- [ ] **Step 4: Implement `src/rl/obs_tensors.py`**

```python
"""Convert a numpy Observation into torch tensors for the policy (TZ §6.2).

Feature normalisation is already applied upstream in build_observation; this is
a pure dtype/layout conversion.
"""

from dataclasses import dataclass

import torch

from src.env.observation import Observation


@dataclass
class ObsTensors:
    task_features: torch.Tensor
    node_features: torch.Tensor
    edge_index: torch.Tensor
    globals: torch.Tensor
    ready_mask: torch.Tensor
    alive_mask: torch.Tensor


def obs_to_tensors(obs: Observation) -> ObsTensors:
    return ObsTensors(
        task_features=torch.from_numpy(obs.task_features).float(),
        node_features=torch.from_numpy(obs.node_features).float(),
        edge_index=torch.from_numpy(obs.edge_index).long(),
        globals=torch.from_numpy(obs.globals).float(),
        ready_mask=torch.from_numpy(obs.ready_mask),
        alive_mask=torch.from_numpy(obs.alive_mask),
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_obs_tensors.py -v`
Expected: 2 passed.

- [ ] **Step 6: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: rl package + observation->tensor adapter"
```

---

### Task 4: GraphSAGE encoder (bidirectional)

**Files:**
- Create: `src/rl/gnn_encoder.py`
- Test: `tests/test_gnn_encoder.py`

**Interfaces:**
- Consumes: `torch`, `torch_geometric.nn.SAGEConv`.
- Produces:
  - `gnn_encoder.GNNEncoder(task_in: int = 15, node_in: int = 9, hidden: int = 64, layers: int = 2)` (`nn.Module`).
  - `GNNEncoder.forward(task_features: Tensor [N,task_in], edge_index: Tensor [2,E], node_features: Tensor [M,node_in]) -> tuple[Tensor, Tensor, Tensor, Tensor]` returning `(h [N,hidden], g [hidden], n_emb [M,hidden], c [hidden])`.
  - Bidirectional realised by per-layer separate forward and reverse `SAGEConv` streams concatenated then projected.

- [ ] **Step 1: Write the failing test**

`tests/test_gnn_encoder.py`:
```python
import torch

from src.rl.gnn_encoder import GNNEncoder


def _inputs(n: int = 4, m: int = 2, edges=((0, 1), (0, 2), (1, 3), (2, 3))):
    task_features = torch.randn(n, 15)
    node_features = torch.randn(m, 9)
    edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long) if edges else torch.zeros(
        (2, 0), dtype=torch.long
    )
    return task_features, edge_index, node_features


def test_encoder_output_shapes() -> None:
    enc = GNNEncoder(hidden=32, layers=2)
    tf, ei, nf = _inputs()
    h, g, n_emb, c = enc(tf, ei, nf)
    assert h.shape == (4, 32)
    assert g.shape == (32,)
    assert n_emb.shape == (2, 32)
    assert c.shape == (32,)


def test_encoder_handles_empty_edges() -> None:
    enc = GNNEncoder(hidden=16, layers=2)
    tf, _, nf = _inputs(n=1, m=1, edges=())
    h, g, n_emb, c = enc(tf, torch.zeros((2, 0), dtype=torch.long), nf)
    assert h.shape == (1, 16) and not torch.isnan(h).any()


def test_encoder_is_direction_sensitive() -> None:
    # reversing all edges should change the per-task embeddings (proves reverse
    # stream carries information distinct from forward)
    torch.manual_seed(0)
    enc = GNNEncoder(hidden=32, layers=2)
    tf, ei, nf = _inputs()
    h_fwd, *_ = enc(tf, ei, nf)
    h_rev, *_ = enc(tf, ei.flip(0), nf)
    assert not torch.allclose(h_fwd, h_rev, atol=1e-5)


def test_encoder_gradients_flow() -> None:
    enc = GNNEncoder(hidden=16, layers=2)
    tf, ei, nf = _inputs()
    h, g, n_emb, c = enc(tf, ei, nf)
    (g.sum() + c.sum()).backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert any(gr is not None and gr.abs().sum() > 0 for gr in grads)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_gnn_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.rl.gnn_encoder`.

- [ ] **Step 3: Implement `src/rl/gnn_encoder.py`**

```python
"""Bidirectional GraphSAGE encoder over the DAG (TZ §6.1)."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv


class _BiSAGELayer(nn.Module):
    """One GraphSAGE layer aggregating from predecessors AND successors.

    Direction is flagged by using two separate SAGEConv streams: one over the
    forward edge_index (incoming from predecessors) and one over the reversed
    edge_index (incoming from successors). Their outputs are concatenated and
    projected, so each task embedding sees both directions.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fwd = SAGEConv(in_channels, out_channels)
        self.rev = SAGEConv(in_channels, out_channels)
        self.proj = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        rev_index = edge_index.flip(0) if edge_index.numel() else edge_index
        hf = self.fwd(x, edge_index)
        hr = self.rev(x, rev_index)
        return F.relu(self.proj(torch.cat([hf, hr], dim=-1)))


class GNNEncoder(nn.Module):
    def __init__(
        self, task_in: int = 15, node_in: int = 9, hidden: int = 64, layers: int = 2
    ) -> None:
        super().__init__()
        self.input = nn.Linear(task_in, hidden)
        self.layers = nn.ModuleList(_BiSAGELayer(hidden, hidden) for _ in range(layers))
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

    def forward(
        self, task_features: Tensor, edge_index: Tensor, node_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = F.relu(self.input(task_features))
        for layer in self.layers:
            x = layer(x, edge_index)
        h = x
        g = h.mean(dim=0)
        n_emb = self.node_mlp(node_features)
        c = n_emb.mean(dim=0)
        return h, g, n_emb, c
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_gnn_encoder.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: bidirectional GraphSAGE encoder"
```

---

### Task 5: Two-head policy network — forward + masking

**Files:**
- Create: `src/rl/policy.py`
- Test: `tests/test_policy_masking.py`

**Interfaces:**
- Consumes: `GNNEncoder`, `obs_to_tensors`/`ObsTensors`, `torch`, `Observation`.
- Produces `policy.TwoHeadPolicy(encoder: GNNEncoder, hidden: int = 64, glob_in: int = 2)` (`nn.Module`) with:
  - `encode(t: ObsTensors) -> tuple[Tensor, Tensor, Tensor]` returning `(h [N,hidden], n_emb [M,hidden], ctx [2*hidden+glob_in])` where `ctx = cat([g, c, globals])`.
  - `task_logits(h, ctx, ready_mask) -> Tensor [N]` with `-inf` at non-ready positions.
  - `node_logits(h_tau, n_emb, ctx, alive_mask) -> Tensor [M]` with `-inf` at dead positions.
  - `value(ctx) -> Tensor` (scalar).

- [ ] **Step 1: Write the failing test**

`tests/test_policy_masking.py`:
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
    if dead_node is not None:
        nodes[dead_node].alive = False
    env = ClusterEnv(load_config("config.yaml"))
    obs, _ = env.reset(dag=dag, nodes=nodes)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_policy_masking.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.rl.policy`.

- [ ] **Step 3: Implement `src/rl/policy.py`**

```python
"""Two-head autoregressive policy + critic (TZ §6.3).

pi(a) = pi(task | s) * pi(node | s, task). Head 1 scores ready tasks, Head 2
scores alive nodes given the chosen task. Masking sets non-candidate logits to
-inf (exactly zero probability). Variable candidate-set sizes are handled by
per-instance scoring — never a fixed Discrete action space.
"""

import torch
from torch import Tensor, nn

from src.rl.gnn_encoder import GNNEncoder
from src.rl.obs_tensors import ObsTensors

_NEG_INF = float("-inf")


class TwoHeadPolicy(nn.Module):
    def __init__(self, encoder: GNNEncoder, hidden: int = 64, glob_in: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        ctx_dim = 2 * hidden + glob_in  # [g, c, globals]
        self.head_task = nn.Sequential(
            nn.Linear(hidden + ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.head_node = nn.Sequential(
            nn.Linear(hidden + hidden + ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def encode(self, t: ObsTensors) -> tuple[Tensor, Tensor, Tensor]:
        h, g, n_emb, c = self.encoder(t.task_features, t.edge_index, t.node_features)
        ctx = torch.cat([g, c, t.globals], dim=-1)
        return h, n_emb, ctx

    def task_logits(self, h: Tensor, ctx: Tensor, ready_mask: Tensor) -> Tensor:
        ctx_b = ctx.unsqueeze(0).expand(h.shape[0], -1)
        scores = self.head_task(torch.cat([h, ctx_b], dim=-1)).squeeze(-1)
        return scores.masked_fill(~ready_mask, _NEG_INF)

    def node_logits(
        self, h_tau: Tensor, n_emb: Tensor, ctx: Tensor, alive_mask: Tensor
    ) -> Tensor:
        cond = torch.cat([h_tau, ctx], dim=-1).unsqueeze(0).expand(n_emb.shape[0], -1)
        scores = self.head_node(torch.cat([n_emb, cond], dim=-1)).squeeze(-1)
        return scores.masked_fill(~alive_mask, _NEG_INF)

    def value(self, ctx: Tensor) -> Tensor:
        return self.critic(ctx).squeeze(-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_policy_masking.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: two-head policy forward + masking"
```

---

### Task 6: Policy `act` / `evaluate_action` — joint log-prob + gradients

**Files:**
- Modify: `src/rl/policy.py` (add methods + imports)
- Test: `tests/test_policy_action.py`

**Interfaces:**
- Consumes: Task 5's `TwoHeadPolicy` internals; `torch.distributions.Categorical`; `obs_to_tensors`; `Observation`.
- Produces, added to `TwoHeadPolicy`:
  - `act(self, obs: Observation) -> tuple[tuple[int, int], Tensor, Tensor]` returning `((task_id, node_id), log_prob, value)` — samples τ from the masked task head, then ν from the masked node head given τ; `log_prob = logπ(τ) + logπ(ν|τ)`.
  - `evaluate_action(self, obs: Observation, task_id: int, node_id: int) -> tuple[Tensor, Tensor, Tensor]` returning `(log_prob, entropy, value)` for a given action; `entropy = H(τ) + H(ν|τ)`.

- [ ] **Step 1: Write the failing test**

`tests/test_policy_action.py`:
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
    node_lp = torch.log_softmax(
        pol.node_logits(h[task_id], n_emb, ctx, t.alive_mask), dim=-1
    )[node_id]
    log_prob, entropy, value = pol.evaluate_action(obs, task_id, node_id)
    assert torch.allclose(log_prob, task_lp + node_lp, atol=1e-6)
    assert entropy.item() >= 0.0


def test_gradients_flow_through_both_heads_and_critic() -> None:
    obs = _obs()
    pol = _policy()
    log_prob, entropy, value = pol.evaluate_action(obs, 0, 1)
    loss = -(log_prob) + (value - 1.0) ** 2 - 0.01 * entropy
    loss.backward()

    def has_grad(module) -> bool:
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert has_grad(pol.head_task)
    assert has_grad(pol.head_node)
    assert has_grad(pol.critic)
    assert has_grad(pol.encoder)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_policy_action.py -v`
Expected: FAIL with `AttributeError: 'TwoHeadPolicy' object has no attribute 'act'`.

- [ ] **Step 3: Add the import and methods to `src/rl/policy.py`**

Add to the imports at the top:
```python
from torch.distributions import Categorical

from src.env.observation import Observation
from src.rl.obs_tensors import obs_to_tensors
```

Append these methods to `TwoHeadPolicy`:
```python
    def act(self, obs: Observation) -> tuple[tuple[int, int], Tensor, Tensor]:
        t = obs_to_tensors(obs)
        h, n_emb, ctx = self.encode(t)
        task_dist = Categorical(logits=self.task_logits(h, ctx, t.ready_mask))
        task_id = task_dist.sample()
        node_dist = Categorical(
            logits=self.node_logits(h[task_id], n_emb, ctx, t.alive_mask)
        )
        node_id = node_dist.sample()
        log_prob = task_dist.log_prob(task_id) + node_dist.log_prob(node_id)
        return (int(task_id), int(node_id)), log_prob, self.value(ctx)

    def evaluate_action(
        self, obs: Observation, task_id: int, node_id: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        t = obs_to_tensors(obs)
        h, n_emb, ctx = self.encode(t)
        task_dist = Categorical(logits=self.task_logits(h, ctx, t.ready_mask))
        node_dist = Categorical(
            logits=self.node_logits(h[task_id], n_emb, ctx, t.alive_mask)
        )
        log_prob = task_dist.log_prob(
            torch.tensor(task_id)
        ) + node_dist.log_prob(torch.tensor(node_id))
        entropy = task_dist.entropy() + node_dist.entropy()
        return log_prob, entropy, self.value(ctx)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_policy_action.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full suite + lint/format**

Run:
```bash
.venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/black --check .
```
Expected: all tests pass (M1+M2 + M3a); ruff clean; black reports no changes.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: policy act/evaluate with joint autoregressive log-prob"
```

---

## Self-Review

**Spec coverage (roadmap M3a + TZ §6.1–6.3):**
- GraphSAGE encoder, PyG, bidirectional (forward+reverse streams), 2–3 layers, `h_i` + pooled `g`, node MLP → `n_j` pooled → `c` — Task 4. ✅
- Two-head autoregressive policy: Head 1 `MLP([h_i, g, c, globals])` masked-ready; Head 2 `MLP([h_τ, n_j, g, c, globals])` masked-alive — Tasks 5 + 6. ✅
- Critic `MLP([g, c, globals])` scalar — Task 5. ✅
- Joint log-prob `logπ(τ)+logπ(ν|τ)` — Task 6 (tested by decomposition equality). ✅
- Pointer scoring + masking, variable candidate sets, never `Discrete` — Task 5 (`test_handles_variable_candidate_set_sizes`, masking tests). ✅
- Deferred observation normalization (N_SPEED, T_MEM, N_UTIL, T_OUTDATA, …) — Task 2. ✅
- torch + PyG deps (not DGL) — Task 1. ✅
- No training loop — none present; rollout buffer explicitly deferred to M3b. ✅
- Gradient flow through encoder + both heads + critic — Task 6 (`test_gradients_flow_through_both_heads_and_critic`). ✅

**Placeholder scan:** no TBD/TODO; every code step has complete code. ✅

**Type consistency:** `GNNEncoder.forward -> (h, g, n_emb, c)` consumed identically by `TwoHeadPolicy.encode`; `ObsTensors` fields used consistently across obs_tensors/policy; `encode -> (h, n_emb, ctx)` with `ctx = cat([g, c, globals])` dim `2*hidden+glob_in`, matching `head_task` input `hidden+ctx_dim`, `head_node` input `hidden+hidden+ctx_dim`, `critic` input `ctx_dim`; `task_logits`/`node_logits`/`value` signatures match their test call sites; `act`/`evaluate_action` signatures match Task 6 tests. ✅

**Masking correctness:** `masked_fill(~mask, -inf)` → `softmax` yields exactly 0 probability for non-ready/dead candidates (Task 5 tests assert this). At least one ready task and one alive node always exist (env invariant), so no all-`-inf` row. ✅

**Deferred to M3b (consistent with roadmap):** PPO trainer, rollout buffer (tensorising per-node features at the decision point — noted, not built), training loop, GAE/clip, config wiring of `hidden`/`layers`, checkpoints. M3a deliberately stops at a differentiable, mask-correct network with a sampling/evaluation API.
