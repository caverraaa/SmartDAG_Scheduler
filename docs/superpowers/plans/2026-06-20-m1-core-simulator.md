# M1 — Core + Deterministic Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the hand-checkable, deterministic (`noise_std=0, failure_rate=0`) DAG-scheduling simulator — core data structures, cost model, placement objective, observation builder, synthetic DAG factory, and a Gymnasium-style env where one `step` = one task→node assignment.

**Architecture:** Layered, no RL yet. `core/` = pure data (Task, ComputeNode, TaskDAG, Schedule). `env/cost_model.py` = stateless time/energy/comm arithmetic. `env/placement.py` = the single side-effect-free objective evaluator (`weighted_cost`) shared by env-reward and (later) the greedy baseline. `env/observation.py` = builds the per-task/per-node/global feature tensors + masks (no consumer yet; validated structurally). `env/cluster_env.py` = `reset`/`step`, computes & caches `M_ref`/`E_ref`, emits them in the `info` dict. `dag_factory/` = Factory pattern over a layered-random synthetic generator. `utils/` = seeding, normalization, config.

**Tech Stack:** Python 3.10+, `networkx`, `numpy`, `pyyaml`, `pytest`, `ruff`, `black`. **PyG / PyTorch are NOT installed in M1** — there is no GNN consumer until M3a; adding them now is out of scope.

## Global Constraints

Every task's requirements implicitly include all of these (copied verbatim from `SmartDAG_Scheduler_TZ.md` §4–§6 + Appendix A and the roadmap):

- **Python 3.10+**; full type hints on all public functions/classes; `ruff` + `black` clean.
- **`node_id == index`** is enforced: `nodes[i].node_id == i` for all `i` (asserted in `ClusterEnv`).
- **Append-only EFT placement.** One `step` = one assignment; a deterministic episode is exactly `N` steps (N = number of tasks). No insertion-based scheduling.
- **Single objective evaluator (A.1).** All per-step cost decomposition lives in one side-effect-free function `placement.weighted_cost(task, node, state)`, returning the *individual normalised components* `Δmakespan/M_ref` and `Δenergy/E_ref` — **not** a pre-summed scalar. `cluster_env.step` negates+weight-sums them into the reward.
- **Δmakespan (A.2).** `Δmakespan_t = max(free_at_time)_after − max(free_at_time)_before` over alive nodes (the running schedule horizon) — **not** the just-scheduled task's finish time. `0` if the task fits under the current horizon.
- **References (A.2), computed once at `reset()` and cached, emitted in the Gymnasium `info` dict, NOT in `obs`:** `M_ref` = fastest-exec critical-path lower bound (longest path using per-task min exec time across node types, comm-free); `E_ref = Σ_i min_node(energy_{i,node})`.
- **Reward (γ=1):** `r_t = −w1·(Δmakespan/M_ref) − w2·(Δenergy/E_ref)`; terminal `r_T += w3·balance_index`. Weights from `config.yaml` (defaults `w1=1.0, w2=0.3, w3=0.2`). No deadline term, no `w4`.
- **Load-balance index:** `1 − CV(per-node busy time)` over **all alive nodes** (idle nodes count as 0 busy time), clamped to `[0, 1]`.
- **Heavy-tailed `base_cost` (A.3):** synthetic generator uses a wide / heavy-tailed distribution (log-normal). Uniform task sizes are forbidden.
- **Layered-random DAGs:** acyclic by construction.
- **Isolated RNG:** use `numpy.random.Generator` via `utils.seeding.make_rng(seed)`; never global `numpy.random.seed`.
- **`Observation.nodes` is a live reference** — future (M3) buffers must tensorise at the decision point; do not rely on it being immutable across steps.
- **Non-goals (do NOT add):** task deadlines; GUI/web; DB/k8s; DQN/PER; DGL; fixed `Discrete` action space; PyG/torch in M1.

---

### Task 1: Project scaffolding & tooling

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `src/__init__.py`, `src/core/__init__.py`, `src/env/__init__.py`, `src/dag_factory/__init__.py`, `src/utils/__init__.py`
- Create: `tests/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `src` package and a working `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/black`.

- [ ] **Step 1: Create the Python package skeleton**

Create empty `src/__init__.py`, `src/core/__init__.py`, `src/env/__init__.py`, `src/dag_factory/__init__.py`, `src/utils/__init__.py`, `tests/__init__.py` (all zero-byte).

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "smartdag-scheduler"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `requirements.txt`**

```text
networkx>=3.1
numpy>=1.24
pyyaml>=6.0
pytest>=7.4
ruff>=0.4
black>=24.0
```

(PyG/torch intentionally omitted — added in M3a.)

- [ ] **Step 4: Write `config.yaml`** (defaults; consumed by `utils/config.py` in Task 2)

```yaml
# Reward weights (TZ §6.4)
w1: 1.0   # makespan
w2: 0.3   # energy
w3: 0.2   # terminal load-balance

# Instance generation
seed: 0
n_tasks: 30
n_nodes: 8
beta: 5.0        # heterogeneity factor (max/min speed ratio across node types)
ccr: 0.5         # communication-to-computation ratio
edge_prob: 0.4   # intra-layer-forward edge probability
n_layers: 6

# Stochasticity (deterministic in M1)
noise_std: 0.0
failure_rate: 0.0
```

- [ ] **Step 5: Create the virtualenv and install**

Run:
```bash
python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
```
Expected: installs complete with no errors; `smartdag-scheduler` installed in editable mode.

- [ ] **Step 6: Write the smoke test**

`tests/test_smoke.py`:
```python
def test_imports() -> None:
    import src.core  # noqa: F401
    import src.env  # noqa: F401
    import src.dag_factory  # noqa: F401
    import src.utils  # noqa: F401
```

- [ ] **Step 7: Run smoke test + lint/format**

Run:
```bash
.venv/bin/pytest tests/test_smoke.py -v && .venv/bin/ruff check . && .venv/bin/black --check .
```
Expected: 1 passed; ruff reports no errors; black reports all files would be left unchanged (or run `.venv/bin/black .` first to format).

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "chore: M1 project scaffolding, tooling, config defaults"
```

---

### Task 2: Utils — seeding, normalization, config

**Files:**
- Create: `src/utils/seeding.py`
- Create: `src/utils/normalization.py`
- Create: `src/utils/config.py`
- Test: `tests/test_utils.py`

**Interfaces:**
- Consumes: `config.yaml` (Task 1).
- Produces:
  - `seeding.make_rng(seed: int) -> numpy.random.Generator`
  - `normalization.safe_div(value: float, ref: float, eps: float = 1e-8) -> float`
  - `config.Config` (frozen dataclass: `w1,w2,w3: float`; `seed,n_tasks,n_nodes,n_layers: int`; `beta,ccr,edge_prob,noise_std,failure_rate: float`)
  - `config.load_config(path: str = "config.yaml") -> Config`

- [ ] **Step 1: Write the failing test**

`tests/test_utils.py`:
```python
import numpy as np

from src.utils.config import Config, load_config
from src.utils.normalization import safe_div
from src.utils.seeding import make_rng


def test_make_rng_is_deterministic_and_isolated() -> None:
    a = make_rng(42).random(5)
    b = make_rng(42).random(5)
    c = make_rng(7).random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert isinstance(make_rng(0), np.random.Generator)


def test_safe_div_handles_zero_ref() -> None:
    assert safe_div(10.0, 2.0) == 5.0
    assert safe_div(1.0, 0.0) < 1e7  # eps prevents div-by-zero blow-up


def test_load_config_reads_defaults() -> None:
    cfg = load_config("config.yaml")
    assert isinstance(cfg, Config)
    assert cfg.w1 == 1.0 and cfg.w2 == 0.3 and cfg.w3 == 0.2
    assert cfg.noise_std == 0.0 and cfg.failure_rate == 0.0
    assert cfg.n_tasks == 30 and cfg.n_nodes == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_utils.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.utils.config`.

- [ ] **Step 3: Implement `seeding.py`**

```python
"""Isolated RNG helpers (never touch global numpy state)."""

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return an independent generator seeded deterministically."""
    return np.random.default_rng(seed)
```

- [ ] **Step 4: Implement `normalization.py`**

```python
"""Small normalization helpers for feature/reward scaling."""


def safe_div(value: float, ref: float, eps: float = 1e-8) -> float:
    """Divide by a reference, guarding against zero references."""
    return value / (ref + eps)
```

- [ ] **Step 5: Implement `config.py`**

```python
"""Typed config loaded from config.yaml."""

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class Config:
    w1: float
    w2: float
    w3: float
    seed: int
    n_tasks: int
    n_nodes: int
    n_layers: int
    beta: float
    ccr: float
    edge_prob: float
    noise_std: float
    failure_rate: float


def load_config(path: str = "config.yaml") -> Config:
    """Parse the YAML config into a typed, frozen Config."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(
        w1=float(raw["w1"]),
        w2=float(raw["w2"]),
        w3=float(raw["w3"]),
        seed=int(raw["seed"]),
        n_tasks=int(raw["n_tasks"]),
        n_nodes=int(raw["n_nodes"]),
        n_layers=int(raw["n_layers"]),
        beta=float(raw["beta"]),
        ccr=float(raw["ccr"]),
        edge_prob=float(raw["edge_prob"]),
        noise_std=float(raw["noise_std"]),
        failure_rate=float(raw["failure_rate"]),
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_utils.py -v`
Expected: 3 passed.

- [ ] **Step 7: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: utils (seeding, normalization, typed config)"
```

---

### Task 3: Core — Task & ComputeNode

**Files:**
- Create: `src/core/task.py`
- Create: `src/core/compute_node.py`
- Test: `tests/test_core_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `task.TaskClass` (Enum: `DATA_PARALLEL`, `SEQUENTIAL`, `STREAMING`)
  - `task.Task` (frozen dataclass: `id: int`, `base_cost: float`, `mem_required: float`, `task_class: TaskClass`)
  - `compute_node.NodeType` (Enum: `CPU`, `GPU`, `FPGA`, `TPU`)
  - `compute_node.ComputeNode` (dataclass: `node_id: int`, `node_type: NodeType`, `speed_by_class: dict[TaskClass, float]`, `power_w: float`, `bandwidth: float`, `free_at_time: float = 0.0`, `alive: bool = True`; methods `speed(tc) -> float`, `reset() -> None`)

- [ ] **Step 1: Write the failing test**

`tests/test_core_types.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.task import Task, TaskClass


def _node() -> ComputeNode:
    return ComputeNode(
        node_id=0,
        node_type=NodeType.GPU,
        speed_by_class={
            TaskClass.DATA_PARALLEL: 4.0,
            TaskClass.SEQUENTIAL: 1.0,
            TaskClass.STREAMING: 2.0,
        },
        power_w=300.0,
        bandwidth=10.0,
    )


def test_task_is_immutable() -> None:
    t = Task(id=1, base_cost=5.0, mem_required=2.0, task_class=TaskClass.SEQUENTIAL)
    assert t.id == 1 and t.task_class is TaskClass.SEQUENTIAL


def test_node_speed_lookup_and_reset() -> None:
    n = _node()
    assert n.speed(TaskClass.DATA_PARALLEL) == 4.0
    assert n.free_at_time == 0.0 and n.alive is True
    n.free_at_time = 12.0
    n.alive = False
    n.reset()
    assert n.free_at_time == 0.0 and n.alive is True


def test_four_node_types_exist() -> None:
    assert {nt.name for nt in NodeType} == {"CPU", "GPU", "FPGA", "TPU"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_core_types.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.core.task`.

- [ ] **Step 3: Implement `task.py`**

```python
"""Task and task-class definitions (TZ §4)."""

from dataclasses import dataclass
from enum import Enum


class TaskClass(Enum):
    DATA_PARALLEL = "data_parallel"
    SEQUENTIAL = "sequential"
    STREAMING = "streaming"


@dataclass(frozen=True)
class Task:
    id: int
    base_cost: float
    mem_required: float
    task_class: TaskClass
```

- [ ] **Step 4: Implement `compute_node.py`**

```python
"""Compute node definitions (TZ §4)."""

from dataclasses import dataclass, field
from enum import Enum

from src.core.task import TaskClass


class NodeType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    FPGA = "fpga"
    TPU = "tpu"


@dataclass
class ComputeNode:
    node_id: int
    node_type: NodeType
    speed_by_class: dict[TaskClass, float]
    power_w: float
    bandwidth: float
    free_at_time: float = 0.0
    alive: bool = True

    def speed(self, task_class: TaskClass) -> float:
        return self.speed_by_class[task_class]

    def reset(self) -> None:
        """Return the node to its initial idle, alive state."""
        self.free_at_time = 0.0
        self.alive = True
```

(`field` import is unused — delete it if ruff flags `F401`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_core_types.py -v`
Expected: 3 passed.

- [ ] **Step 6: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: core Task and ComputeNode types"
```

---

### Task 4: Core — TaskDAG

**Files:**
- Create: `src/core/dag.py`
- Test: `tests/test_dag.py`

**Interfaces:**
- Consumes: `core.task.Task`.
- Produces `core.dag.TaskDAG`:
  - `__init__(self, tasks: list[Task], edges: list[tuple[int, int, float]])` — `edges` = `(src_id, dst_id, data_volume)`; raises `ValueError` if cyclic or if task ids are not `0..N-1`.
  - `n_tasks: int` (property)
  - `task(self, tid: int) -> Task`
  - `predecessors(self, tid: int) -> list[int]`, `successors(self, tid: int) -> list[int]`
  - `out_degree(self, tid: int) -> int`
  - `out_data(self, tid: int) -> float` (sum of outgoing edge volumes)
  - `edge_data(self, src: int, dst: int) -> float`
  - `ready_set(self, scheduled: set[int]) -> list[int]` (tasks not in `scheduled` whose predecessors ⊆ `scheduled`; sorted ascending)
  - `longest_path_length(self, node_weight: Callable[[int], float], edge_weight: Callable[[int, int], float]) -> float`
  - `b_level(self, tid: int) -> float` (longest path to a sink, `base_cost` node weight, edge weight 0, inclusive of `tid`)
  - `t_level(self, tid: int) -> float` (longest path from a source, `base_cost` node weight, exclusive of `tid`)
  - `critical_path_length(self) -> float`
  - `edge_index(self) -> list[tuple[int, int]]` (forward edges)

- [ ] **Step 1: Write the failing test**

`tests/test_dag.py`:
```python
import pytest

from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass


def _diamond() -> TaskDAG:
    # 0 -> 1 -> 3 ; 0 -> 2 -> 3   (base_costs: 2,4,4,2)
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 4.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 4.0, 1.0, TaskClass.SEQUENTIAL),
        Task(3, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    edges = [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)]
    return TaskDAG(tasks, edges)


def test_rejects_cycle() -> None:
    tasks = [Task(0, 1.0, 1.0, TaskClass.SEQUENTIAL), Task(1, 1.0, 1.0, TaskClass.SEQUENTIAL)]
    with pytest.raises(ValueError):
        TaskDAG(tasks, [(0, 1, 1.0), (1, 0, 1.0)])


def test_structure_queries() -> None:
    d = _diamond()
    assert d.n_tasks == 4
    assert d.predecessors(3) == [1, 2]
    assert d.successors(0) == [1, 2]
    assert d.out_degree(0) == 2
    assert d.out_data(0) == 20.0
    assert d.edge_data(1, 3) == 10.0


def test_ready_set_tracks_predecessors() -> None:
    d = _diamond()
    assert d.ready_set(set()) == [0]
    assert d.ready_set({0}) == [1, 2]
    assert d.ready_set({0, 1}) == [2]
    assert d.ready_set({0, 1, 2}) == [3]
    assert d.ready_set({0, 1, 2, 3}) == []


def test_levels_and_critical_path() -> None:
    d = _diamond()
    # b_level(3)=2 ; b_level(1)=4+2=6 ; b_level(0)=2+6=8
    assert d.b_level(3) == 2.0
    assert d.b_level(1) == 6.0
    assert d.b_level(0) == 8.0
    # t_level(0)=0 ; t_level(1)=2 ; t_level(3)=2+4=6
    assert d.t_level(0) == 0.0
    assert d.t_level(1) == 2.0
    assert d.t_level(3) == 6.0
    assert d.critical_path_length() == 8.0


def test_rejects_noncontiguous_ids() -> None:
    tasks = [Task(0, 1.0, 1.0, TaskClass.SEQUENTIAL), Task(5, 1.0, 1.0, TaskClass.SEQUENTIAL)]
    with pytest.raises(ValueError):
        TaskDAG(tasks, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dag.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.core.dag`.

- [ ] **Step 3: Implement `dag.py`**

```python
"""TaskDAG: a typed wrapper over networkx.DiGraph (TZ §4)."""

from collections.abc import Callable

import networkx as nx

from src.core.task import Task


class TaskDAG:
    def __init__(self, tasks: list[Task], edges: list[tuple[int, int, float]]) -> None:
        ids = [t.id for t in tasks]
        if sorted(ids) != list(range(len(tasks))):
            raise ValueError("Task ids must be exactly 0..N-1 (node_id == index).")
        self._g: nx.DiGraph = nx.DiGraph()
        for t in tasks:
            self._g.add_node(t.id, task=t)
        for src, dst, data in edges:
            self._g.add_edge(src, dst, data=float(data))
        if not nx.is_directed_acyclic_graph(self._g):
            raise ValueError("TaskDAG must be acyclic.")
        self._b_level: dict[int, float] = {}
        self._t_level: dict[int, float] = {}
        self._compute_levels()

    @property
    def n_tasks(self) -> int:
        return self._g.number_of_nodes()

    def task(self, tid: int) -> Task:
        return self._g.nodes[tid]["task"]

    def predecessors(self, tid: int) -> list[int]:
        return sorted(self._g.predecessors(tid))

    def successors(self, tid: int) -> list[int]:
        return sorted(self._g.successors(tid))

    def out_degree(self, tid: int) -> int:
        return self._g.out_degree(tid)

    def out_data(self, tid: int) -> float:
        return float(sum(self._g.edges[tid, s]["data"] for s in self._g.successors(tid)))

    def edge_data(self, src: int, dst: int) -> float:
        return float(self._g.edges[src, dst]["data"])

    def ready_set(self, scheduled: set[int]) -> list[int]:
        return [
            n
            for n in sorted(self._g.nodes)
            if n not in scheduled and all(p in scheduled for p in self._g.predecessors(n))
        ]

    def longest_path_length(
        self,
        node_weight: Callable[[int], float],
        edge_weight: Callable[[int, int], float],
    ) -> float:
        dist: dict[int, float] = {}
        for n in nx.topological_sort(self._g):
            best_pred = 0.0
            for p in self._g.predecessors(n):
                best_pred = max(best_pred, dist[p] + edge_weight(p, n))
            dist[n] = best_pred + node_weight(n)
        return max(dist.values()) if dist else 0.0

    def _compute_levels(self) -> None:
        topo = list(nx.topological_sort(self._g))

        # t_level: longest path from a source to (excluding) the node, base_cost weighted.
        for n in topo:
            preds = list(self._g.predecessors(n))
            self._t_level[n] = (
                0.0 if not preds else max(self._t_level[p] + self.task(p).base_cost for p in preds)
            )

        # b_level: longest path from the node (inclusive) to a sink, base_cost weighted.
        for n in reversed(topo):
            succ = list(self._g.successors(n))
            downstream = 0.0 if not succ else max(self._b_level[s] for s in succ)
            self._b_level[n] = self.task(n).base_cost + downstream

    def b_level(self, tid: int) -> float:
        return self._b_level[tid]

    def t_level(self, tid: int) -> float:
        return self._t_level[tid]

    def critical_path_length(self) -> float:
        return max(self._b_level.values()) if self._b_level else 0.0

    def edge_index(self) -> list[tuple[int, int]]:
        return [(int(u), int(v)) for u, v in self._g.edges]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_dag.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: TaskDAG with levels, ready-set, critical path"
```

---

### Task 5: Core — Assignment & Schedule

**Files:**
- Create: `src/core/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Consumes: nothing (operates on plain ids/floats).
- Produces:
  - `schedule.Assignment` (frozen dataclass: `task_id: int`, `node_id: int`, `start: float`, `finish: float`)
  - `schedule.Schedule`:
    - `__init__(self, n_nodes: int)`
    - `total_energy: float` (attribute, accumulated)
    - `add(self, assignment: Assignment, energy: float) -> None`
    - `assignments: list[Assignment]` (attribute)
    - `makespan(self) -> float` (max finish, 0 if empty)
    - `busy_time_by_node(self) -> dict[int, float]`
    - `load_balance_index(self, n_alive_nodes: int) -> float` (`1 − CV` over all alive nodes, idle = 0, clamped `[0,1]`)

- [ ] **Step 1: Write the failing test**

`tests/test_schedule.py`:
```python
from src.core.schedule import Assignment, Schedule


def test_makespan_and_energy_accumulate() -> None:
    s = Schedule(n_nodes=2)
    s.add(Assignment(0, 0, 0.0, 2.0), energy=200.0)
    s.add(Assignment(1, 0, 2.0, 6.0), energy=400.0)
    assert s.makespan() == 6.0
    assert s.total_energy == 600.0
    assert s.busy_time_by_node() == {0: 6.0}


def test_load_balance_perfectly_even_is_one() -> None:
    s = Schedule(n_nodes=2)
    s.add(Assignment(0, 0, 0.0, 5.0), energy=1.0)
    s.add(Assignment(1, 1, 0.0, 5.0), energy=1.0)
    assert s.load_balance_index(2) == 1.0


def test_load_balance_fully_skewed_is_zero() -> None:
    s = Schedule(n_nodes=2)
    s.add(Assignment(0, 0, 0.0, 10.0), energy=1.0)
    # node 1 idle -> busy times [10, 0] -> CV == 1 -> index 0
    assert s.load_balance_index(2) == 0.0


def test_empty_schedule_makespan_zero() -> None:
    assert Schedule(n_nodes=3).makespan() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.core.schedule`.

- [ ] **Step 3: Implement `schedule.py`**

```python
"""Assignment and Schedule result objects + integral metrics (TZ §4, §5.1)."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Assignment:
    task_id: int
    node_id: int
    start: float
    finish: float


@dataclass
class Schedule:
    n_nodes: int
    assignments: list[Assignment] = field(default_factory=list)
    total_energy: float = 0.0

    def add(self, assignment: Assignment, energy: float) -> None:
        self.assignments.append(assignment)
        self.total_energy += energy

    def makespan(self) -> float:
        return max((a.finish for a in self.assignments), default=0.0)

    def busy_time_by_node(self) -> dict[int, float]:
        busy: dict[int, float] = {}
        for a in self.assignments:
            busy[a.node_id] = busy.get(a.node_id, 0.0) + (a.finish - a.start)
        return busy

    def load_balance_index(self, n_alive_nodes: int) -> float:
        """1 - CV(busy time) over all alive nodes; idle nodes count as 0."""
        if n_alive_nodes <= 0:
            return 0.0
        busy = self.busy_time_by_node()
        times = [busy.get(i, 0.0) for i in range(n_alive_nodes)]
        mean = sum(times) / n_alive_nodes
        if mean == 0.0:
            return 1.0  # nothing scheduled yet: treat as perfectly even
        variance = sum((t - mean) ** 2 for t in times) / n_alive_nodes
        cv = (variance**0.5) / mean
        return max(0.0, min(1.0, 1.0 - cv))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_schedule.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: Schedule/Assignment with makespan, energy, load-balance index"
```

---

### Task 6: Env — cost model (time, energy, comm, speed table)

**Files:**
- Create: `src/env/cost_model.py`
- Test: `tests/test_cost_model.py`

**Interfaces:**
- Consumes: `core.task.{Task, TaskClass}`, `core.compute_node.{ComputeNode, NodeType}`, `numpy.random.Generator`.
- Produces:
  - `cost_model.exec_time(task: Task, node: ComputeNode) -> float` = `base_cost / node.speed(task_class)`
  - `cost_model.energy(task: Task, node: ComputeNode) -> float` = `node.power_w * exec_time`
  - `cost_model.comm_time(data_volume: float, bandwidth: float, latency: float = 0.0) -> float` = `data_volume / bandwidth + latency`
  - `cost_model.make_speed_table(rng, beta: float) -> dict[NodeType, dict[TaskClass, float]]` — speeds with per-class affinity (data_parallel→GPU/TPU, sequential→CPU, streaming→FPGA) and max/min ratio across node types ≈ `beta` for each task class.

- [ ] **Step 1: Write the failing test**

`tests/test_cost_model.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.task import Task, TaskClass
from src.env.cost_model import comm_time, energy, exec_time, make_speed_table
from src.utils.seeding import make_rng


def _node(speed: float, power: float = 100.0, bw: float = 10.0) -> ComputeNode:
    return ComputeNode(
        node_id=0,
        node_type=NodeType.CPU,
        speed_by_class={tc: speed for tc in TaskClass},
        power_w=power,
        bandwidth=bw,
    )


def test_exec_time_and_energy() -> None:
    t = Task(0, 8.0, 1.0, TaskClass.SEQUENTIAL)
    n = _node(speed=2.0, power=150.0)
    assert exec_time(t, n) == 4.0
    assert energy(t, n) == 600.0


def test_comm_time() -> None:
    assert comm_time(20.0, 10.0) == 2.0
    assert comm_time(20.0, 10.0, latency=0.5) == 2.5


def test_speed_table_beta_ratio() -> None:
    table = make_speed_table(make_rng(0), beta=5.0)
    assert set(table.keys()) == set(NodeType)
    for tc in TaskClass:
        speeds = [table[nt][tc] for nt in NodeType]
        ratio = max(speeds) / min(speeds)
        assert 3.0 <= ratio <= 8.0  # approximately beta=5


def test_speed_table_affinity() -> None:
    table = make_speed_table(make_rng(1), beta=5.0)
    # data_parallel fastest on GPU or TPU
    dp = {nt: table[nt][TaskClass.DATA_PARALLEL] for nt in NodeType}
    best = max(dp, key=dp.get)
    assert best in (NodeType.GPU, NodeType.TPU)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cost_model.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.env.cost_model`.

- [ ] **Step 3: Implement `cost_model.py`**

```python
"""Stateless cost arithmetic + heterogeneous speed-table generation (TZ §5.1)."""

import numpy as np

from src.core.compute_node import ComputeNode, NodeType
from src.core.task import Task, TaskClass

# Per-(class, type) affinity multipliers: the "right" node is faster for each class.
_AFFINITY: dict[TaskClass, dict[NodeType, float]] = {
    TaskClass.DATA_PARALLEL: {
        NodeType.GPU: 1.0,
        NodeType.TPU: 0.9,
        NodeType.FPGA: 0.4,
        NodeType.CPU: 0.25,
    },
    TaskClass.SEQUENTIAL: {
        NodeType.CPU: 1.0,
        NodeType.FPGA: 0.6,
        NodeType.GPU: 0.35,
        NodeType.TPU: 0.3,
    },
    TaskClass.STREAMING: {
        NodeType.FPGA: 1.0,
        NodeType.TPU: 0.7,
        NodeType.GPU: 0.6,
        NodeType.CPU: 0.35,
    },
}


def exec_time(task: Task, node: ComputeNode) -> float:
    return task.base_cost / node.speed(task.task_class)


def energy(task: Task, node: ComputeNode) -> float:
    return node.power_w * exec_time(task, node)


def comm_time(data_volume: float, bandwidth: float, latency: float = 0.0) -> float:
    return data_volume / bandwidth + latency


def make_speed_table(
    rng: np.random.Generator, beta: float
) -> dict[NodeType, dict[TaskClass, float]]:
    """Map each (node_type, task_class) to a speed coefficient.

    For each task class the affinity profile is scaled so the max/min speed
    ratio across node types is approximately ``beta`` (with mild jitter).
    """
    table: dict[NodeType, dict[TaskClass, float]] = {nt: {} for nt in NodeType}
    for tc in TaskClass:
        affinity = _AFFINITY[tc]
        # Map best affinity (1.0) -> beta, worst -> 1.0, linearly.
        lo = min(affinity.values())
        hi = max(affinity.values())
        for nt in NodeType:
            frac = (affinity[nt] - lo) / (hi - lo)
            base_speed = 1.0 + frac * (beta - 1.0)
            jitter = float(rng.uniform(0.9, 1.1))
            table[nt][tc] = base_speed * jitter
    return table
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cost_model.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: cost model (exec/energy/comm) + beta-scaled speed table"
```

---

### Task 7: Env — cluster factory

**Files:**
- Create: `src/env/cluster_factory.py`
- Test: `tests/test_cluster_factory.py`

**Interfaces:**
- Consumes: `cost_model.make_speed_table`, `core.compute_node.{ComputeNode, NodeType}`, `numpy.random.Generator`.
- Produces: `cluster_factory.make_cluster(rng: np.random.Generator, n_nodes: int, beta: float) -> list[ComputeNode]` — `nodes[i].node_id == i`; node types cycle through `[CPU, GPU, FPGA, TPU]`; `power_w` drawn from per-type TDP ranges (CPU 65–150, GPU 250–400, FPGA 30–75, TPU 200–450); `bandwidth` ~ uniform(5, 20).

- [ ] **Step 1: Write the failing test**

`tests/test_cluster_factory.py`:
```python
from src.core.compute_node import NodeType
from src.env.cluster_factory import make_cluster
from src.utils.seeding import make_rng

_TDP = {
    NodeType.CPU: (65.0, 150.0),
    NodeType.GPU: (250.0, 400.0),
    NodeType.FPGA: (30.0, 75.0),
    NodeType.TPU: (200.0, 450.0),
}


def test_node_id_equals_index_and_count() -> None:
    nodes = make_cluster(make_rng(0), n_nodes=8, beta=5.0)
    assert len(nodes) == 8
    assert all(n.node_id == i for i, n in enumerate(nodes))


def test_power_within_tdp_ranges() -> None:
    nodes = make_cluster(make_rng(1), n_nodes=12, beta=5.0)
    for n in nodes:
        lo, hi = _TDP[n.node_type]
        assert lo <= n.power_w <= hi


def test_all_four_types_present_when_enough_nodes() -> None:
    nodes = make_cluster(make_rng(2), n_nodes=4, beta=5.0)
    assert {n.node_type for n in nodes} == set(NodeType)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cluster_factory.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.env.cluster_factory`.

- [ ] **Step 3: Implement `cluster_factory.py`**

```python
"""Construct heterogeneous clusters of ComputeNodes (TZ §5.1)."""

import numpy as np

from src.core.compute_node import ComputeNode, NodeType
from src.env.cost_model import make_speed_table

_TDP_RANGE: dict[NodeType, tuple[float, float]] = {
    NodeType.CPU: (65.0, 150.0),
    NodeType.GPU: (250.0, 400.0),
    NodeType.FPGA: (30.0, 75.0),
    NodeType.TPU: (200.0, 450.0),
}
_TYPE_CYCLE: list[NodeType] = [NodeType.CPU, NodeType.GPU, NodeType.FPGA, NodeType.TPU]


def make_cluster(rng: np.random.Generator, n_nodes: int, beta: float) -> list[ComputeNode]:
    speed_table = make_speed_table(rng, beta)
    nodes: list[ComputeNode] = []
    for i in range(n_nodes):
        nt = _TYPE_CYCLE[i % len(_TYPE_CYCLE)]
        lo, hi = _TDP_RANGE[nt]
        nodes.append(
            ComputeNode(
                node_id=i,
                node_type=nt,
                speed_by_class=dict(speed_table[nt]),
                power_w=float(rng.uniform(lo, hi)),
                bandwidth=float(rng.uniform(5.0, 20.0)),
            )
        )
    return nodes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cluster_factory.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: cluster factory (node_id==index, TDP power, beta speeds)"
```

---

### Task 8: DAG factory — synthetic layered-random generator

**Files:**
- Create: `src/dag_factory/synthetic.py`
- Create: `src/dag_factory/factory.py`
- Test: `tests/test_synthetic.py`

**Interfaces:**
- Consumes: `core.dag.TaskDAG`, `core.task.{Task, TaskClass}`, `numpy.random.Generator`.
- Produces:
  - `synthetic.generate_synthetic(rng, n_tasks, n_layers, edge_prob, ccr) -> TaskDAG` — layered-random (edges only earlier→later layer ⇒ acyclic); `base_cost` log-normal (heavy-tailed); each task gets a random `TaskClass`; edge `data_volume` scaled so total comm ≈ `ccr × total compute`.
  - `factory.DAGFactory` with classmethod `create(source: str, rng, **params) -> TaskDAG`; supports `source="synthetic"`; raises `ValueError` for unknown sources. (The `"wfcommons"` source is registered in M5.)

- [ ] **Step 1: Write the failing test**

`tests/test_synthetic.py`:
```python
import numpy as np

from src.core.dag import TaskDAG
from src.dag_factory.factory import DAGFactory
from src.dag_factory.synthetic import generate_synthetic
from src.utils.seeding import make_rng


def test_generates_valid_acyclic_dag_of_requested_size() -> None:
    dag = generate_synthetic(make_rng(0), n_tasks=30, n_layers=6, edge_prob=0.4, ccr=0.5)
    assert isinstance(dag, TaskDAG)
    assert dag.n_tasks == 30  # construction would raise if cyclic


def test_base_costs_are_heavy_tailed_not_uniform() -> None:
    dag = generate_synthetic(make_rng(1), n_tasks=60, n_layers=8, edge_prob=0.4, ccr=0.5)
    costs = np.array([dag.task(i).base_cost for i in range(dag.n_tasks)])
    # heavy-tailed: coefficient of variation clearly above a uniform spread
    assert costs.std() / costs.mean() > 0.3
    assert costs.min() > 0.0


def test_factory_dispatches_synthetic_and_rejects_unknown() -> None:
    dag = DAGFactory.create(
        "synthetic", make_rng(2), n_tasks=20, n_layers=5, edge_prob=0.4, ccr=0.5
    )
    assert dag.n_tasks == 20
    try:
        DAGFactory.create("does-not-exist", make_rng(0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_reproducible_with_same_seed() -> None:
    a = generate_synthetic(make_rng(5), n_tasks=25, n_layers=5, edge_prob=0.4, ccr=0.5)
    b = generate_synthetic(make_rng(5), n_tasks=25, n_layers=5, edge_prob=0.4, ccr=0.5)
    assert a.edge_index() == b.edge_index()
    assert [a.task(i).base_cost for i in range(25)] == [b.task(i).base_cost for i in range(25)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_synthetic.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.dag_factory.synthetic`.

- [ ] **Step 3: Implement `synthetic.py`**

```python
"""Layered-random synthetic DAG generator (TZ §7, Appendix A.3)."""

import numpy as np

from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass

_CLASSES = list(TaskClass)


def generate_synthetic(
    rng: np.random.Generator,
    n_tasks: int,
    n_layers: int,
    edge_prob: float,
    ccr: float,
) -> TaskDAG:
    """Generate a valid acyclic DAG via layer assignment.

    Edges only ever point from an earlier layer to a later layer, so the graph
    is acyclic by construction. base_cost is log-normal (heavy-tailed, A.3).
    """
    # Assign each task id to a layer; keep layers non-empty by seeding one per layer.
    layers: list[list[int]] = [[] for _ in range(n_layers)]
    for i in range(n_tasks):
        layer = i % n_layers if i < n_layers else int(rng.integers(0, n_layers))
        layers[layer].append(i)

    tasks: list[Task] = []
    for i in range(n_tasks):
        base_cost = float(rng.lognormal(mean=1.0, sigma=0.6))
        tasks.append(
            Task(
                id=i,
                base_cost=base_cost,
                mem_required=float(rng.uniform(1.0, 8.0)),
                task_class=_CLASSES[int(rng.integers(0, len(_CLASSES)))],
            )
        )

    total_compute = sum(t.base_cost for t in tasks)

    # Connect tasks to some tasks in strictly later layers.
    raw_edges: list[tuple[int, int]] = []
    for li in range(n_layers - 1):
        for src in layers[li]:
            for lj in range(li + 1, n_layers):
                for dst in layers[lj]:
                    if rng.random() < edge_prob:
                        raw_edges.append((src, dst))

    # Scale edge data volumes so total communication ≈ ccr * total compute.
    n_edges = max(1, len(raw_edges))
    per_edge_volume = (ccr * total_compute) / n_edges
    edges = [(s, d, float(per_edge_volume * rng.uniform(0.5, 1.5))) for s, d in raw_edges]

    return TaskDAG(tasks, edges)
```

- [ ] **Step 4: Implement `factory.py`**

```python
"""DAGFactory: Factory pattern over interchangeable DAG sources (TZ §3, §7)."""

import numpy as np

from src.core.dag import TaskDAG
from src.dag_factory.synthetic import generate_synthetic


class DAGFactory:
    @classmethod
    def create(cls, source: str, rng: np.random.Generator, **params: float) -> TaskDAG:
        if source == "synthetic":
            return generate_synthetic(
                rng,
                n_tasks=int(params["n_tasks"]),
                n_layers=int(params["n_layers"]),
                edge_prob=float(params["edge_prob"]),
                ccr=float(params["ccr"]),
            )
        raise ValueError(f"Unknown DAG source: {source!r}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_synthetic.py -v`
Expected: 4 passed.

- [ ] **Step 6: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: synthetic layered-random DAG generator + DAGFactory"
```

---

### Task 9: Env — placement (ClusterState, EFT, weighted_cost)

**Files:**
- Create: `src/env/placement.py`
- Test: `tests/test_placement.py`

**Interfaces:**
- Consumes: `cost_model.{exec_time, energy, comm_time}`, `core.dag.TaskDAG`, `core.compute_node.ComputeNode`, `core.task.Task`.
- Produces:
  - `placement.ClusterState` (dataclass): `nodes: list[ComputeNode]`, `dag: TaskDAG`, `task_finish: dict[int, float]`, `task_node: dict[int, int]`, `m_ref: float`, `e_ref: float`, `sim_time: float = 0.0`.
  - `placement.CostComponents` (frozen dataclass): `d_makespan_norm: float`, `d_energy_norm: float`.
  - `placement.horizon(nodes: list[ComputeNode]) -> float` = `max(free_at_time)` over alive nodes (0 if none).
  - `placement.earliest_start_finish(task: Task, node: ComputeNode, state: ClusterState) -> tuple[float, float]` (append-only EFT, side-effect-free).
  - `placement.weighted_cost(task: Task, node: ComputeNode, state: ClusterState) -> CostComponents` (side-effect-free; A.1/A.2).

- [ ] **Step 1: Write the failing test**

`tests/test_placement.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.placement import (
    ClusterState,
    earliest_start_finish,
    horizon,
    weighted_cost,
)


def _setup() -> ClusterState:
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 4.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return ClusterState(
        nodes=nodes, dag=dag, task_finish={}, task_node={}, m_ref=6.0, e_ref=600.0
    )


def test_eft_first_task_starts_at_zero() -> None:
    st = _setup()
    start, finish = earliest_start_finish(st.dag.task(0), st.nodes[0], st)
    assert (start, finish) == (0.0, 2.0)


def test_eft_respects_predecessor_and_cross_node_comm() -> None:
    st = _setup()
    # task 0 finished at t=2 on node 0
    st.task_finish[0] = 2.0
    st.task_node[0] = 0
    # task 1 on node 1 (cross-node): comm = 10/10 = 1.0 -> ready at 3.0
    start, finish = earliest_start_finish(st.dag.task(1), st.nodes[1], st)
    assert start == 3.0 and finish == 7.0
    # task 1 on node 0 (same node): no comm -> ready at 2.0
    start0, finish0 = earliest_start_finish(st.dag.task(1), st.nodes[0], st)
    assert start0 == 2.0 and finish0 == 6.0


def test_weighted_cost_returns_normalised_components() -> None:
    st = _setup()
    comp = weighted_cost(st.dag.task(0), st.nodes[0], st)
    # horizon before = 0, after = 2 -> d_makespan = 2 ; /m_ref(6) = 1/3
    assert abs(comp.d_makespan_norm - (2.0 / 6.0)) < 1e-9
    # energy = 100 * 2 = 200 ; /e_ref(600) = 1/3
    assert abs(comp.d_energy_norm - (200.0 / 600.0)) < 1e-9


def test_horizon_zero_delta_when_task_fits_under_existing_horizon() -> None:
    st = _setup()
    # push node 0 horizon to 10 by pretending it is busy
    st.nodes[0].free_at_time = 10.0
    assert horizon(st.nodes) == 10.0
    # scheduling task 1 on node 1 finishes at 4 (< 10) -> d_makespan = 0
    comp = weighted_cost(st.dag.task(1), st.nodes[1], st)
    assert comp.d_makespan_norm == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_placement.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.env.placement`.

- [ ] **Step 3: Implement `placement.py`**

```python
"""The single side-effect-free objective evaluator (TZ Appendix A.1/A.2)."""

from dataclasses import dataclass

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.core.task import Task
from src.env.cost_model import comm_time, energy, exec_time


@dataclass
class ClusterState:
    nodes: list[ComputeNode]
    dag: TaskDAG
    task_finish: dict[int, float]
    task_node: dict[int, int]
    m_ref: float
    e_ref: float
    sim_time: float = 0.0


@dataclass(frozen=True)
class CostComponents:
    d_makespan_norm: float
    d_energy_norm: float


def horizon(nodes: list[ComputeNode]) -> float:
    """Running schedule horizon = max free_at_time over alive nodes."""
    alive = [n.free_at_time for n in nodes if n.alive]
    return max(alive) if alive else 0.0


def earliest_start_finish(
    task: Task, node: ComputeNode, state: ClusterState
) -> tuple[float, float]:
    """Append-only EFT: when can this task start/finish on this node, given state."""
    ready_time = 0.0
    for pred in state.dag.predecessors(task.id):
        pred_finish = state.task_finish[pred]
        if state.task_node[pred] != node.node_id:
            pred_finish += comm_time(state.dag.edge_data(pred, task.id), node.bandwidth)
        ready_time = max(ready_time, pred_finish)
    start = max(node.free_at_time, ready_time)
    return start, start + exec_time(task, node)


def weighted_cost(task: Task, node: ComputeNode, state: ClusterState) -> CostComponents:
    """Normalised (Δmakespan/M_ref, Δenergy/E_ref) for a candidate assignment.

    Δmakespan is the change in the schedule HORIZON (A.2), not the task finish.
    Side-effect-free: never mutates node/state.
    """
    before = horizon(state.nodes)
    _, finish = earliest_start_finish(task, node, state)
    after = max(before, finish)
    d_makespan = after - before
    d_energy = energy(task, node)
    return CostComponents(
        d_makespan_norm=d_makespan / state.m_ref,
        d_energy_norm=d_energy / state.e_ref,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_placement.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: placement EFT + single weighted_cost objective evaluator"
```

---

### Task 10: Env — observation builder + masks

**Files:**
- Create: `src/env/observation.py`
- Test: `tests/test_masking.py`

**Interfaces:**
- Consumes: `placement.ClusterState`, `core.dag.TaskDAG`, `cost_model.exec_time`, `core.compute_node.NodeType`, `core.task.TaskClass`, `numpy`.
- Produces:
  - Column-index constants (module-level ints) for task features (`T_BASE_COST`, `T_COST_CPU`, `T_COST_GPU`, `T_COST_FPGA`, `T_COST_TPU`, `T_MEM`, `T_DONE`, `T_READY`, `T_BLOCKED`, `T_SCHEDULED`, `T_UNSCHED_PREDS`, `T_BLEVEL`, `T_TLEVEL`, `T_OUTDEG`, `T_OUTDATA`) and `N_TASK_FEATURES`; node features (`N_TYPE_CPU`, `N_TYPE_GPU`, `N_TYPE_FPGA`, `N_TYPE_TPU`, `N_FREE_REL`, `N_UTIL`, `N_POWER`, `N_SPEED`, `N_ALIVE`) and `N_NODE_FEATURES`.
  - `observation.Observation` (dataclass): `task_features: np.ndarray [N, N_TASK_FEATURES]`, `node_features: np.ndarray [M, N_NODE_FEATURES]`, `globals: np.ndarray [2]` (`[current_makespan, fraction_done]`; graph embeddings `g,c` are added in M3a), `edge_index: np.ndarray [2, E]`, `ready_mask: np.ndarray[bool] [N]`, `alive_mask: np.ndarray[bool] [M]`, `nodes: list[ComputeNode]` (live reference).
  - `observation.build_observation(state, scheduled, current_makespan) -> Observation`.

- [ ] **Step 1: Write the failing test**

`tests/test_masking.py`:
```python
import numpy as np

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.observation import (
    N_NODE_FEATURES,
    N_TASK_FEATURES,
    N_ALIVE,
    T_READY,
    T_SCHEDULED,
    build_observation,
)
from src.env.placement import ClusterState


def _state() -> ClusterState:
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 4.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 300.0, 10.0),
    ]
    return ClusterState(nodes=nodes, dag=dag, task_finish={}, task_node={}, m_ref=6.0, e_ref=600.0)


def test_shapes() -> None:
    st = _state()
    obs = build_observation(st, scheduled=set(), current_makespan=0.0)
    assert obs.task_features.shape == (3, N_TASK_FEATURES)
    assert obs.node_features.shape == (2, N_NODE_FEATURES)
    assert obs.globals.shape == (2,)


def test_ready_mask_only_true_for_ready_unscheduled_tasks() -> None:
    st = _state()
    obs = build_observation(st, scheduled=set(), current_makespan=0.0)
    assert obs.ready_mask.tolist() == [True, False, False]  # only task 0 ready
    obs2 = build_observation(st, scheduled={0}, current_makespan=2.0)
    assert obs2.ready_mask.tolist() == [False, True, True]  # 1 and 2 now ready, 0 scheduled


def test_alive_mask_reflects_node_alive_flag() -> None:
    st = _state()
    st.nodes[1].alive = False
    obs = build_observation(st, scheduled=set(), current_makespan=0.0)
    assert obs.alive_mask.tolist() == [True, False]
    assert obs.node_features[1, N_ALIVE] == 0.0


def test_scheduled_flag_set() -> None:
    st = _state()
    obs = build_observation(st, scheduled={0}, current_makespan=2.0)
    assert obs.task_features[0, T_SCHEDULED] == 1.0
    assert obs.task_features[0, T_READY] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_masking.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.env.observation`.

- [ ] **Step 3: Implement `observation.py`**

```python
"""Observation builder: per-task / per-node / global features + masks (TZ §6.2).

No GNN consumes this in M1; it is validated structurally (shapes + masks). The
graph/node pooled embeddings g,c are prepended to globals in M3a. Two feature
normalizations are intentionally left raw here and fixed in M3a when the GNN
exists: N_SPEED (node mean speed) and T_MEM (memory scale).
"""

from dataclasses import dataclass

import numpy as np

from src.core.compute_node import ComputeNode, NodeType
from src.core.task import TaskClass
from src.env.cost_model import exec_time
from src.env.placement import ClusterState

# --- task feature columns ---
T_BASE_COST = 0
T_COST_CPU = 1
T_COST_GPU = 2
T_COST_FPGA = 3
T_COST_TPU = 4
T_MEM = 5
T_DONE = 6
T_READY = 7
T_BLOCKED = 8
T_SCHEDULED = 9
T_UNSCHED_PREDS = 10
T_BLEVEL = 11
T_TLEVEL = 12
T_OUTDEG = 13
T_OUTDATA = 14
N_TASK_FEATURES = 15

# --- node feature columns ---
N_TYPE_CPU = 0
N_TYPE_GPU = 1
N_TYPE_FPGA = 2
N_TYPE_TPU = 3
N_FREE_REL = 4
N_UTIL = 5
N_POWER = 6
N_SPEED = 7
N_ALIVE = 8
N_NODE_FEATURES = 9

_TYPE_COL = {
    NodeType.CPU: N_TYPE_CPU,
    NodeType.GPU: N_TYPE_GPU,
    NodeType.FPGA: N_TYPE_FPGA,
    NodeType.TPU: N_TYPE_TPU,
}
_COST_COL = {
    NodeType.CPU: T_COST_CPU,
    NodeType.GPU: T_COST_GPU,
    NodeType.FPGA: T_COST_FPGA,
    NodeType.TPU: T_COST_TPU,
}


@dataclass
class Observation:
    task_features: np.ndarray
    node_features: np.ndarray
    globals: np.ndarray
    edge_index: np.ndarray
    ready_mask: np.ndarray
    alive_mask: np.ndarray
    nodes: list[ComputeNode]


def _mean_cost(state: ClusterState) -> float:
    costs = [state.dag.task(i).base_cost for i in range(state.dag.n_tasks)]
    return sum(costs) / len(costs) if costs else 1.0


def build_observation(
    state: ClusterState, scheduled: set[int], current_makespan: float
) -> Observation:
    dag = state.dag
    n = dag.n_tasks
    m = len(state.nodes)
    mean_cost = _mean_cost(state)
    cp = dag.critical_path_length() or 1.0
    ready = set(dag.ready_set(scheduled))

    task_features = np.zeros((n, N_TASK_FEATURES), dtype=np.float32)
    ready_mask = np.zeros(n, dtype=bool)
    # one representative node per present type for the per-task cost vector
    type_repr: dict[NodeType, ComputeNode] = {}
    for node in state.nodes:
        type_repr.setdefault(node.node_type, node)

    for tid in range(n):
        task = dag.task(tid)
        f = task_features[tid]
        f[T_BASE_COST] = task.base_cost / mean_cost
        for nt, node in type_repr.items():
            f[_COST_COL[nt]] = exec_time(task, node) / cp
        f[T_MEM] = task.mem_required  # raw in M1; normalized in M3a
        is_scheduled = tid in scheduled
        is_ready = tid in ready
        f[T_DONE] = 1.0 if is_scheduled else 0.0
        f[T_READY] = 1.0 if (is_ready and not is_scheduled) else 0.0
        f[T_BLOCKED] = 1.0 if (not is_ready and not is_scheduled) else 0.0
        f[T_SCHEDULED] = 1.0 if is_scheduled else 0.0
        f[T_UNSCHED_PREDS] = float(sum(1 for p in dag.predecessors(tid) if p not in scheduled))
        f[T_BLEVEL] = dag.b_level(tid) / cp
        f[T_TLEVEL] = dag.t_level(tid) / cp
        f[T_OUTDEG] = float(dag.out_degree(tid))
        f[T_OUTDATA] = dag.out_data(tid)
        ready_mask[tid] = is_ready and not is_scheduled

    node_features = np.zeros((m, N_NODE_FEATURES), dtype=np.float32)
    alive_mask = np.zeros(m, dtype=bool)
    for j, node in enumerate(state.nodes):
        nf = node_features[j]
        nf[_TYPE_COL[node.node_type]] = 1.0
        nf[N_FREE_REL] = (node.free_at_time - state.sim_time) / cp
        nf[N_UTIL] = node.free_at_time / current_makespan if current_makespan > 0 else 0.0
        nf[N_POWER] = node.power_w
        nf[N_SPEED] = float(np.mean(list(node.speed_by_class.values())))  # raw in M1
        nf[N_ALIVE] = 1.0 if node.alive else 0.0
        alive_mask[j] = node.alive

    edges = dag.edge_index()
    edge_index = (
        np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    )
    globals_ = np.array(
        [current_makespan / cp, len(scheduled) / n if n else 0.0], dtype=np.float32
    )
    return Observation(
        task_features=task_features,
        node_features=node_features,
        globals=globals_,
        edge_index=edge_index,
        ready_mask=ready_mask,
        alive_mask=alive_mask,
        nodes=state.nodes,
    )
```

(`TaskClass` import is used only if needed — delete if ruff flags `F401`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_masking.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: observation builder with feature columns + ready/alive masks"
```

---

### Task 11: Env — ClusterEnv reset (instance + M_ref/E_ref)

**Files:**
- Create: `src/env/cluster_env.py`
- Test: `tests/test_env_reset.py`

**Interfaces:**
- Consumes: `utils.config.Config`, `utils.seeding.make_rng`, `dag_factory.factory.DAGFactory`, `env.cluster_factory.make_cluster`, `cost_model.{exec_time, energy}`, `placement.ClusterState`, `observation.build_observation`, `core.schedule.Schedule`, `core.dag.TaskDAG`, `core.compute_node.ComputeNode`.
- Produces `cluster_env.ClusterEnv`:
  - `__init__(self, config: Config)`
  - `reset(self, dag: TaskDAG | None = None, nodes: list[ComputeNode] | None = None) -> tuple[Observation, dict]` — samples instance from config (or uses provided `dag`/`nodes` for fixtures); enforces `nodes[i].node_id == i`; computes & caches `m_ref`, `e_ref`; returns `(obs, info)` where `info = {"m_ref": float, "e_ref": float}`.
  - static `_compute_m_ref(dag, nodes) -> float`, `_compute_e_ref(dag, nodes) -> float`.

- [ ] **Step 1: Write the failing test**

`tests/test_env_reset.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.utils.config import load_config


def _golden_instance() -> tuple[TaskDAG, list[ComputeNode]]:
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


def test_reset_emits_refs_in_info_not_obs() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    obs, info = env.reset(dag=dag, nodes=nodes)
    assert "m_ref" in info and "e_ref" in info
    # M_ref = fastest-exec critical path: min exec per task (GPU speed 2) on path 0->1->3
    # = 1 + 2 + 1 = 4
    assert info["m_ref"] == 4.0
    # E_ref = sum min energy per task; CPU=100*c, GPU=200*(c/2)=100*c -> equal -> 100*(2+4+4+2)
    assert info["e_ref"] == 1200.0


def test_reset_enforces_node_id_equals_index() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    nodes[1].node_id = 7  # break the invariant
    try:
        env.reset(dag=dag, nodes=nodes)
        raise AssertionError("expected ValueError for node_id != index")
    except ValueError:
        pass


def test_reset_samples_from_config_when_no_override() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    obs, info = env.reset()
    assert obs.task_features.shape[0] == 30  # n_tasks from config
    assert obs.node_features.shape[0] == 8  # n_nodes from config
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_env_reset.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.env.cluster_env`.

- [ ] **Step 3: Implement `cluster_env.py` (reset only; step added in Task 12)**

```python
"""Gymnasium-style decision-point environment (TZ §5.3, §6.4, Appendix A)."""

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.core.schedule import Schedule
from src.env.cluster_factory import make_cluster
from src.env.cost_model import energy, exec_time
from src.env.observation import Observation, build_observation
from src.env.placement import ClusterState
from src.dag_factory.factory import DAGFactory
from src.utils.config import Config
from src.utils.seeding import make_rng


class ClusterEnv:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._rng = make_rng(config.seed)
        self.state: ClusterState | None = None
        self.schedule: Schedule | None = None
        self.scheduled: set[int] = set()

    @staticmethod
    def _compute_m_ref(dag: TaskDAG, nodes: list[ComputeNode]) -> float:
        """Fastest-exec critical-path lower bound (min exec per task, comm-free)."""

        def node_weight(tid: int) -> float:
            task = dag.task(tid)
            return min(exec_time(task, node) for node in nodes)

        return dag.longest_path_length(node_weight=node_weight, edge_weight=lambda u, v: 0.0)

    @staticmethod
    def _compute_e_ref(dag: TaskDAG, nodes: list[ComputeNode]) -> float:
        """Absolute energy lower bound = sum of per-task minimum energy."""
        total = 0.0
        for tid in range(dag.n_tasks):
            task = dag.task(tid)
            total += min(energy(task, node) for node in nodes)
        return total

    def reset(
        self,
        dag: TaskDAG | None = None,
        nodes: list[ComputeNode] | None = None,
    ) -> tuple[Observation, dict]:
        if dag is None:
            dag = DAGFactory.create(
                "synthetic",
                self._rng,
                n_tasks=self.config.n_tasks,
                n_layers=self.config.n_layers,
                edge_prob=self.config.edge_prob,
                ccr=self.config.ccr,
            )
        if nodes is None:
            nodes = make_cluster(self._rng, self.config.n_nodes, self.config.beta)

        for i, node in enumerate(nodes):
            if node.node_id != i:
                raise ValueError(f"node_id must equal index: nodes[{i}].node_id={node.node_id}")
            node.reset()

        m_ref = self._compute_m_ref(dag, nodes)
        e_ref = self._compute_e_ref(dag, nodes)
        self.state = ClusterState(
            nodes=nodes, dag=dag, task_finish={}, task_node={}, m_ref=m_ref, e_ref=e_ref
        )
        self.schedule = Schedule(n_nodes=len(nodes))
        self.scheduled = set()
        obs = build_observation(self.state, self.scheduled, current_makespan=0.0)
        info = {"m_ref": m_ref, "e_ref": e_ref}
        return obs, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_env_reset.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: ClusterEnv.reset with cached M_ref/E_ref in info dict"
```

---

### Task 12: Env — ClusterEnv step (assignment, reward, terminal balance)

**Files:**
- Modify: `src/env/cluster_env.py`
- Test: `tests/test_env_step.py`

**Interfaces:**
- Consumes: everything from Task 11 plus `placement.{earliest_start_finish, weighted_cost}`, `core.schedule.Assignment`.
- Produces `ClusterEnv.step(self, action: tuple[int, int]) -> tuple[Observation, float, bool, dict]`:
  - `action = (task_id, node_id)`; raises `ValueError` if the task is not ready, already scheduled, or the node is dead / `node_id != index`.
  - reward = `−(w1·d_makespan_norm + w2·d_energy_norm)`; on the terminal step add `+w3·balance_index`.
  - mutates state (append-only EFT): sets `node.free_at_time = finish`, records `task_finish`/`task_node`, adds the `Assignment` (+ its energy) to the schedule, marks the task scheduled.
  - `done = (len(scheduled) == n_tasks)`; `info = {"m_ref","e_ref","makespan","energy","balance"(terminal only)}`.

- [ ] **Step 1: Write the failing test**

`tests/test_env_step.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.env.observation import T_DONE, T_UNSCHED_PREDS
from src.utils.config import load_config


def _golden_instance() -> tuple[TaskDAG, list[ComputeNode]]:
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


def test_deterministic_episode_is_n_steps() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    env.reset(dag=dag, nodes=nodes)
    order = [(0, 0), (1, 0), (2, 0), (3, 0)]
    done = False
    steps = 0
    for action in order:
        _, _, done, _ = env.step(action)
        steps += 1
    assert steps == 4 and done is True


def test_step_transitions_done_and_unsched_preds() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    env.reset(dag=dag, nodes=nodes)
    obs, _, _, _ = env.step((0, 0))
    # task 0 now done
    assert obs.task_features[0, T_DONE] == 1.0
    # children 1 and 2 had 1 unscheduled pred (task 0) -> now 0
    assert obs.task_features[1, T_UNSCHED_PREDS] == 0.0
    assert obs.task_features[2, T_UNSCHED_PREDS] == 0.0


def test_step_rejects_unready_task() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    env.reset(dag=dag, nodes=nodes)
    try:
        env.step((3, 0))  # task 3 not ready (preds 1,2 unscheduled)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_all_on_cpu_matches_hand_computed_schedule() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    env.reset(dag=dag, nodes=nodes)
    info = {}
    for action in [(0, 0), (1, 0), (2, 0), (3, 0)]:
        _, _, done, info = env.step(action)
    # CPU speed 1 -> exec == base_cost; intra-node comm = 0
    # finishes: t0=2, t1=6, t2=10, t3=12 -> makespan 12 ; energy=100*12=1200
    assert info["makespan"] == 12.0
    assert info["energy"] == 1200.0
    assert info["balance"] == 0.0  # node1 idle -> fully skewed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_env_step.py -v`
Expected: FAIL with `AttributeError: 'ClusterEnv' object has no attribute 'step'`.

- [ ] **Step 3: Add imports and the `step` method to `cluster_env.py`**

Add to the imports at the top of `src/env/cluster_env.py`:
```python
from src.core.schedule import Assignment
from src.env.placement import earliest_start_finish, weighted_cost
```

Append this method to the `ClusterEnv` class:
```python
    def step(self, action: tuple[int, int]) -> tuple[Observation, float, bool, dict]:
        if self.state is None or self.schedule is None:
            raise RuntimeError("Call reset() before step().")
        task_id, node_id = action
        state = self.state
        if not (0 <= node_id < len(state.nodes)) or state.nodes[node_id].node_id != node_id:
            raise ValueError(f"Invalid node_id {node_id}")
        node = state.nodes[node_id]
        if not node.alive:
            raise ValueError(f"Node {node_id} is dead")
        if task_id in self.scheduled:
            raise ValueError(f"Task {task_id} already scheduled")
        if task_id not in set(state.dag.ready_set(self.scheduled)):
            raise ValueError(f"Task {task_id} is not ready")

        task = state.dag.task(task_id)
        components = weighted_cost(task, node, state)
        reward = -(
            self.config.w1 * components.d_makespan_norm
            + self.config.w2 * components.d_energy_norm
        )

        start, finish = earliest_start_finish(task, node, state)
        step_energy = energy(task, node)
        node.free_at_time = finish
        state.task_finish[task_id] = finish
        state.task_node[task_id] = node_id
        state.sim_time = max(state.sim_time, finish)
        self.schedule.add(Assignment(task_id, node_id, start, finish), energy=step_energy)
        self.scheduled.add(task_id)

        done = len(self.scheduled) == state.dag.n_tasks
        makespan = self.schedule.makespan()
        info: dict = {
            "m_ref": state.m_ref,
            "e_ref": state.e_ref,
            "makespan": makespan,
            "energy": self.schedule.total_energy,
        }
        if done:
            n_alive = sum(1 for n in state.nodes if n.alive)
            balance = self.schedule.load_balance_index(n_alive)
            reward += self.config.w3 * balance
            info["balance"] = balance

        obs = build_observation(state, self.scheduled, current_makespan=makespan)
        return obs, reward, done, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_env_step.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: ClusterEnv.step (append-only EFT, telescoping reward, terminal balance)"
```

---

### Task 13: Reward telescoping invariant

**Files:**
- Test: `tests/test_reward_telescoping.py`

**Interfaces:**
- Consumes: `ClusterEnv`, `placement.weighted_cost`.
- Produces: a regression test proving `Σ_t (d_makespan_norm·M_ref) == final makespan` and `Σ_t (d_energy_norm·E_ref) == total energy` under γ=1 (the dense per-step signal equals the totals, A.1/A.2).

- [ ] **Step 1: Write the test**

`tests/test_reward_telescoping.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.env.placement import weighted_cost
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


def test_makespan_and_energy_telescope_to_totals() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    obs, info = env.reset(dag=dag, nodes=nodes)
    m_ref, e_ref = info["m_ref"], info["e_ref"]

    sum_makespan = 0.0
    sum_energy = 0.0
    actions = [(0, 1), (1, 0), (2, 1), (3, 0)]  # mixed nodes
    final_info: dict = {}
    for action in actions:
        task = env.state.dag.task(action[0])
        node = env.state.nodes[action[1]]
        comp = weighted_cost(task, node, env.state)  # measure BEFORE applying
        sum_makespan += comp.d_makespan_norm * m_ref
        sum_energy += comp.d_energy_norm * e_ref
        _, _, _, final_info = env.step(action)

    assert abs(sum_makespan - final_info["makespan"]) < 1e-6
    assert abs(sum_energy - final_info["energy"]) < 1e-6
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_reward_telescoping.py -v`
Expected: 1 passed. (If it fails, the bug is in `horizon`/`weighted_cost`/`step` ordering — fix there, not in the test.)

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "test: reward components telescope to total makespan/energy (gamma=1)"
```

---

### Task 14: Golden-schedule regression fixture

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_golden_schedule.py`

**Interfaces:**
- Consumes: `ClusterEnv`, core types.
- Produces: a shared `golden_instance` pytest fixture (tiny 3–4 task DAG with a hand-verified deterministic schedule) and a regression test pinning the known-optimal outcome. This fixture is the permanent anchor for M2–M5 (A.4).

- [ ] **Step 1: Create the shared fixture**

`tests/conftest.py`:
```python
import pytest

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass


@pytest.fixture
def golden_instance() -> tuple[TaskDAG, list[ComputeNode]]:
    """Diamond DAG (0->{1,2}->3) on 1 CPU + 1 GPU; hand-verified schedule.

    base_costs: t0=2, t1=4, t2=4, t3=2. CPU speed 1 (power 100),
    GPU speed 2 (power 200). All edges carry data=10, bandwidth=10 (comm=1
    cross-node, 0 intra-node).
    """
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
```

- [ ] **Step 2: Write the regression test**

`tests/test_golden_schedule.py`:
```python
from src.core.dag import TaskDAG
from src.core.compute_node import ComputeNode
from src.env.cluster_env import ClusterEnv
from src.utils.config import load_config


def test_golden_all_gpu_schedule(
    golden_instance: tuple[TaskDAG, list[ComputeNode]],
) -> None:
    dag, nodes = golden_instance
    env = ClusterEnv(load_config("config.yaml"))
    env.reset(dag=dag, nodes=nodes)
    info: dict = {}
    # All tasks on GPU (node 1, speed 2): exec = base/2 -> t0=1,t1=2,t2=2,t3=1
    # intra-node comm 0; serialized on one node:
    # t0:0-1, t1:1-3, t2:3-5, t3:5-6 -> makespan 6 ; energy = 200*(1+2+2+1)=1200
    for action in [(0, 1), (1, 1), (2, 1), (3, 1)]:
        _, _, done, info = env.step(action)
    assert done is True
    assert info["makespan"] == 6.0
    assert info["energy"] == 1200.0
    assert info["balance"] == 0.0  # only GPU busy, CPU idle


def test_golden_refs_are_stable(
    golden_instance: tuple[TaskDAG, list[ComputeNode]],
) -> None:
    dag, nodes = golden_instance
    env = ClusterEnv(load_config("config.yaml"))
    _, info = env.reset(dag=dag, nodes=nodes)
    assert info["m_ref"] == 4.0
    assert info["e_ref"] == 1200.0
```

- [ ] **Step 3: Run the golden tests**

Run: `.venv/bin/pytest tests/test_golden_schedule.py -v`
Expected: 2 passed.

- [ ] **Step 4: Run the full suite + lint/format**

Run:
```bash
.venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/black --check .
```
Expected: all tests pass; ruff clean; black reports no changes.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test: golden-schedule regression fixture (M2-M5 anchor)"
```

---

## Self-Review

**Spec coverage (roadmap M1 section + TZ §4–6 + Appendix A):**
- Core data structures (§4): Task/ComputeNode (Task 3), TaskDAG (Task 4), Schedule/Assignment (Task 5). ✅
- Cost model (§5.1): exec/energy/comm + β speed table (Task 6); cluster construction (Task 7). ✅
- Placement single evaluator (A.1), Δmakespan horizon (A.2): Task 9. ✅
- M_ref/E_ref definitions + cached + in `info` (A.2/A.3): Task 11. ✅
- Observation features + masks (§6.2): Task 10; deferred N_SPEED/T_MEM normalization explicitly flagged for M3a. ✅
- Reward telescoping (§6.4, γ=1): Tasks 12 + 13. ✅
- Gymnasium reset/step (§5.3): Tasks 11–12. ✅
- Synthetic layered-random + heavy-tailed base_cost + Factory (§7, A.3): Task 8. ✅
- node_id==index (A.2): enforced Task 11; tested. ✅
- Named tests: test_dag (T4), test_cost_model (T6), test_env_step (T12), test_masking (T10), test_reward_telescoping (T13), golden fixture (T14). ✅
- Determinism (noise=0, failure=0): config defaults 0; no stochastic code paths in M1. T_DONE/T_SCHEDULED intentionally coincide (both set from `scheduled`) — divergence deferred to M4 as designed. ✅

**Placeholder scan:** no TBD/TODO; every code step contains complete code. ✅

**Type consistency:** `weighted_cost`→`CostComponents(d_makespan_norm,d_energy_norm)` consistent across placement/env/telescoping; `ClusterState` fields consistent across placement/observation/env; observation column constants imported by name in tests match definitions; `Schedule.add(assignment, energy)` signature consistent (Tasks 5, 12). ✅

**Note on M_ref edge weight:** M1 defines `M_ref` as the comm-free fastest-exec critical path (edge weight 0) — a valid makespan lower bound and exactly what the golden fixture's hand-computed `4.0` assumes. Documented in Task 11.
