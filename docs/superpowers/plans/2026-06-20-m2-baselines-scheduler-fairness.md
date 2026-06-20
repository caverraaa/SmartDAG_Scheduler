# M2 — Baselines + Scheduler Loop + Fairness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the classical scheduling strategies (HEFT, CPOP, Min-Min, Weighted-Sum Greedy, Random) behind one `predict(ready, state)` interface, plus a decision-point scheduler loop that drives them all through the identical `ClusterEnv.step`, with tests that guard faithfulness and the fairness invariant.

**Architecture:** Every strategy implements `BaseSchedulingStrategy.predict(ready, state) -> (task_id, node_id)`. A single `run_episode(env, strategy, ...)` loop computes the ready set, asks the strategy for an action, and applies it via `ClusterEnv.step` — so all strategies share the exact same env mechanics, EFT, cost model, and (later) Observer trigger. The Weighted-Sum Greedy minimises the *same* `placement.weighted_cost` the env reward uses (fairness parity is structural). HEFT/CPOP precompute cluster-aware ranks cached per-DAG.

**Tech Stack:** Python 3.10+, `networkx`, `numpy`, `pytest`, `ruff`, `black`. No torch/PyG (still no GNN consumer until M3a).

## Global Constraints

Every task's requirements implicitly include all of these (copied from `SmartDAG_Scheduler_TZ.md` §8 + Appendix A and the roadmap M2 section):

- **Python 3.10+**; full type hints on all public functions/classes; `ruff` + `black` clean; tooling via `.venv/bin/...`.
- **One strategy interface:** `predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]` returning `(task_id, node_id)`. Identical for every strategy (RL agent will implement the same in M3).
- **Heuristics run faithfully as themselves** — no "upgrading" a heuristic to match the agent's capabilities. HEFT/CPOP/Min-Min are time-only (ignore energy/balance by design).
- **Append-only EFT** placement, reused from M1 `placement.earliest_start_finish`. Baselines are the **no-insertion** HEFT/CPOP forms (this is stated honestly in the thesis; do not add insertion-based slotting).
- **Fairness invariant (sacred):** all strategies run on the same DAG instances, same cluster, same seed, same Observer trigger. The scheduler loop drives every strategy through the **same `ClusterEnv.step`** — no strategy gets a private execution path. Guard this in tests.
- **Weighted-Sum Greedy parity:** the greedy minimises `w1·(Δmakespan/M_ref) + w2·(Δenergy/E_ref)` over `(ready task, alive node)` pairs using the **same** `placement.weighted_cost` the env negates into the reward (§8, A.1). Balance is terminal-only — the greedy never chases it.
- **HEFT/CPOP structural caches** keyed by DAG via `weakref.WeakKeyDictionary` so ranks don't leak/recompute across sampled DAGs.
- **`node_id == index`** holds (enforced by `ClusterEnv`); strategies return a node's `node_id` (== its list index).
- **Isolated RNG:** `RandomStrategy` takes a `numpy.random.Generator` (via `utils.seeding.make_rng`); never global numpy state.
- **Deterministic mode (M2):** `noise_std=0, failure_rate=0`. The `SystemMonitor` Observer is a scaffold — wired into the loop but emits no events until M4. Do not build failure-handling machinery now.
- **Non-goals:** DQN/PER; insertion-based scheduling; energy/balance terms inside HEFT/CPOP/Min-Min; any private per-strategy env path; torch/PyG.

### M1 interfaces these tasks consume (already on `main`)

- `src.env.placement`: `ClusterState(nodes, dag, task_finish, task_node, m_ref, e_ref, sim_time)`; `earliest_start_finish(task, node, state) -> (start, finish)`; `weighted_cost(task, node, state) -> CostComponents(d_makespan_norm, d_energy_norm)`; `horizon(nodes) -> float`.
- `src.env.cluster_env.ClusterEnv`: `reset(dag=None, nodes=None) -> (obs, info)`; `step((task_id, node_id)) -> (obs, reward, done, info)`; attributes `.state: ClusterState`, `.schedule: Schedule`, `.scheduled: set[int]`.
- `src.core.dag.TaskDAG`: `n_tasks`, `task(tid) -> Task`, `predecessors(tid)`, `successors(tid)`, `ready_set(scheduled: set[int]) -> list[int]` (sorted ascending), `edge_data(src, dst)`, `b_level`/`t_level`, `critical_path_length`, `longest_path_length`.
- `src.core.compute_node.ComputeNode`: `node_id`, `node_type`, `speed_by_class`, `power_w`, `bandwidth`, `free_at_time`, `alive`, `speed(task_class)`.
- `src.core.task.Task` (`id`, `base_cost`, `mem_required`, `task_class`), `TaskClass`.
- `src.core.schedule.Schedule`: `.assignments: list[Assignment(task_id, node_id, start, finish)]`, `makespan()`, `total_energy`, `load_balance_index(n_alive)`.
- `src.env.cost_model`: `exec_time(task, node)`, `energy(task, node)`, `comm_time(data, bandwidth, latency=0)`.
- `src.utils.seeding.make_rng(seed) -> np.random.Generator`; `src.utils.config.load_config()`.

---

### Task 1: Strategy base + Random + scheduler loop + SystemMonitor scaffold

This is the harness: the abstract interface, the simplest concrete strategy, the decision-point loop, and the (idle) Observer seam. None is independently testable without the others, so they form one task whose deliverable is "a strategy can drive a full episode through the env."

**Files:**
- Create: `src/strategies/__init__.py` (empty)
- Create: `src/strategies/base.py`
- Create: `src/strategies/random_strategy.py`
- Create: `src/scheduler/__init__.py` (empty)
- Create: `src/scheduler/system_monitor.py`
- Create: `src/scheduler/task_scheduler.py`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: `ClusterState`, `ClusterEnv`, `TaskDAG.ready_set`, `make_rng`.
- Produces:
  - `base.BaseSchedulingStrategy` (ABC) with abstract `predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]`.
  - `random_strategy.RandomStrategy(rng: np.random.Generator)` implementing `predict`.
  - `system_monitor.SystemMonitor` with `subscribe(self, callback)` and `check(self, state: ClusterState) -> list` (returns `[]` in deterministic mode).
  - `task_scheduler.run_episode(env: ClusterEnv, strategy: BaseSchedulingStrategy, dag: TaskDAG | None = None, nodes: list[ComputeNode] | None = None, monitor: SystemMonitor | None = None) -> tuple[Schedule, dict]` returning `(env.schedule, final_info)`.

- [ ] **Step 1: Create the package `__init__.py` files**

Create empty `src/strategies/__init__.py` and `src/scheduler/__init__.py`.

- [ ] **Step 2: Write the failing test**

`tests/test_scheduler.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.scheduler.system_monitor import SystemMonitor
from src.scheduler.task_scheduler import run_episode
from src.strategies.random_strategy import RandomStrategy
from src.utils.config import load_config
from src.utils.seeding import make_rng


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


def test_run_episode_produces_complete_valid_schedule() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    schedule, info = run_episode(env, RandomStrategy(make_rng(0)), dag=dag, nodes=nodes)
    # one assignment per task, each task exactly once
    assert len(schedule.assignments) == 4
    assert sorted(a.task_id for a in schedule.assignments) == [0, 1, 2, 3]
    # every node_id is a valid alive node index
    assert all(0 <= a.node_id < 2 for a in schedule.assignments)
    assert info["makespan"] > 0.0


def test_run_episode_respects_dependencies() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    schedule, _ = run_episode(env, RandomStrategy(make_rng(1)), dag=dag, nodes=nodes)
    finish = {a.task_id: a.finish for a in schedule.assignments}
    start = {a.task_id: a.start for a in schedule.assignments}
    # children cannot start before a parent finishes (comm >= 0)
    assert start[1] >= finish[0]
    assert start[3] >= finish[1] and start[3] >= finish[2]


def test_random_strategy_is_reproducible() -> None:
    dag, nodes = _instance()
    env_a = ClusterEnv(load_config("config.yaml"))
    env_b = ClusterEnv(load_config("config.yaml"))
    sched_a, _ = run_episode(env_a, RandomStrategy(make_rng(7)), dag=dag, nodes=nodes)
    sched_b, _ = run_episode(env_b, RandomStrategy(make_rng(7)), dag=dag, nodes=nodes)
    assert [(a.task_id, a.node_id) for a in sched_a.assignments] == [
        (a.task_id, a.node_id) for a in sched_b.assignments
    ]


def test_system_monitor_idle_in_deterministic_mode() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    assert SystemMonitor().check(env.state) == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_scheduler.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.scheduler.task_scheduler`.

- [ ] **Step 4: Implement `src/strategies/base.py`**

```python
"""Strategy interface shared by every scheduler (heuristic + RL) (TZ §3, §8)."""

from abc import ABC, abstractmethod

from src.env.placement import ClusterState


class BaseSchedulingStrategy(ABC):
    """A scheduling policy: choose one (task, node) at a decision point."""

    @abstractmethod
    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        """Return (task_id, node_id) for the next assignment.

        `ready` is the list of ready (unscheduled, all-predecessors-done) task
        ids; `state` is the live ClusterState. The returned node_id must be the
        index of an alive node.
        """
        raise NotImplementedError
```

- [ ] **Step 5: Implement `src/strategies/random_strategy.py`**

```python
"""Random scheduling strategy — the sanity floor baseline (TZ §8)."""

import numpy as np

from src.env.placement import ClusterState
from src.strategies.base import BaseSchedulingStrategy


class RandomStrategy(BaseSchedulingStrategy):
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        task_id = int(self._rng.choice(ready))
        alive_ids = [n.node_id for n in state.nodes if n.alive]
        node_id = int(self._rng.choice(alive_ids))
        return task_id, node_id
```

- [ ] **Step 6: Implement `src/scheduler/system_monitor.py`**

```python
"""SystemMonitor: the Observer that detects failure/overload events (TZ §3, §8).

In deterministic M2 mode it is idle (no events). M4 will make `check` emit
failure events here, so that all strategies react through the SAME trigger
(the fairness invariant). It is wired into the scheduler loop now to establish
that single trigger point.
"""

from collections.abc import Callable

from src.env.placement import ClusterState


class SystemMonitor:
    def __init__(self) -> None:
        self._subscribers: list[Callable[[ClusterState], None]] = []

    def subscribe(self, callback: Callable[[ClusterState], None]) -> None:
        self._subscribers.append(callback)

    def check(self, state: ClusterState) -> list:
        """Return the events fired at this decision point. Empty in M2."""
        return []
```

- [ ] **Step 7: Implement `src/scheduler/task_scheduler.py`**

```python
"""Decision-point scheduler loop driving any strategy through ClusterEnv (TZ §3, §8)."""

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.core.schedule import Schedule
from src.env.cluster_env import ClusterEnv
from src.scheduler.system_monitor import SystemMonitor
from src.strategies.base import BaseSchedulingStrategy


def run_episode(
    env: ClusterEnv,
    strategy: BaseSchedulingStrategy,
    dag: TaskDAG | None = None,
    nodes: list[ComputeNode] | None = None,
    monitor: SystemMonitor | None = None,
) -> tuple[Schedule, dict]:
    """Run one full episode: reset, then assign tasks one decision point at a time.

    Every strategy is driven through the identical ClusterEnv.step, which is the
    fairness invariant (§8). `monitor.check` is invoked at each decision point as
    the uniform Observer trigger (no-op in deterministic M2).
    """
    env.reset(dag=dag, nodes=nodes)
    done = False
    info: dict = {}
    while not done:
        if monitor is not None:
            monitor.check(env.state)
        ready = env.state.dag.ready_set(env.scheduled)
        action = strategy.predict(ready, env.state)
        _, _, done, info = env.step(action)
    assert env.schedule is not None
    return env.schedule, info
```

- [ ] **Step 8: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_scheduler.py -v`
Expected: 4 passed.

- [ ] **Step 9: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: strategy base, random strategy, scheduler loop, SystemMonitor scaffold"
```

---

### Task 2: Ranking utilities (cluster-aware HEFT/CPOP ranks)

**Files:**
- Create: `src/strategies/ranking.py`
- Test: `tests/test_ranking.py`

**Interfaces:**
- Consumes: `TaskDAG` (`n_tasks`, `task`, `successors`, `predecessors`, `edge_data`), `ComputeNode`, `cost_model.exec_time`.
- Produces:
  - `ranking.mean_exec(dag: TaskDAG, tid: int, nodes: list[ComputeNode]) -> float` — mean `exec_time` over all nodes.
  - `ranking.mean_comm(dag: TaskDAG, src: int, dst: int, nodes: list[ComputeNode]) -> float` — `edge_data(src,dst)` divided by the mean node bandwidth.
  - `ranking.upward_rank(dag: TaskDAG, nodes: list[ComputeNode]) -> dict[int, float]` — HEFT upward rank: `rank_u(i) = mean_exec(i) + max_{j in succ}(mean_comm(i,j) + rank_u(j))`; sinks = `mean_exec(i)`.
  - `ranking.downward_rank(dag: TaskDAG, nodes: list[ComputeNode]) -> dict[int, float]` — `rank_d(i) = max_{p in pred}(rank_d(p) + mean_exec(p) + mean_comm(p,i))`; sources = `0.0`.

- [ ] **Step 1: Write the failing test**

`tests/test_ranking.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.strategies.ranking import downward_rank, mean_comm, mean_exec, upward_rank


def _hetero_diamond() -> tuple[TaskDAG, list[ComputeNode]]:
    # 0->1->3, 0->2->3 ; base costs 2,4,4,2 ; edges all data=10
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


def test_mean_exec_and_comm() -> None:
    dag, nodes = _hetero_diamond()
    # task0 base 2: (2/1 + 2/2)/2 = 1.5
    assert mean_exec(dag, 0, nodes) == 1.5
    assert mean_exec(dag, 1, nodes) == 3.0
    # edge data 10 / mean bandwidth 10 = 1.0
    assert mean_comm(dag, 0, 1, nodes) == 1.0


def test_upward_rank_golden() -> None:
    dag, nodes = _hetero_diamond()
    ru = upward_rank(dag, nodes)
    assert ru[3] == 1.5
    assert ru[1] == 5.5
    assert ru[2] == 5.5
    assert ru[0] == 8.0


def test_downward_rank_golden() -> None:
    dag, nodes = _hetero_diamond()
    rd = downward_rank(dag, nodes)
    assert rd[0] == 0.0
    assert rd[1] == 2.5
    assert rd[2] == 2.5
    assert rd[3] == 6.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ranking.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.strategies.ranking`.

- [ ] **Step 3: Implement `src/strategies/ranking.py`**

```python
"""Cluster-aware HEFT/CPOP rank functions (averaged over processors) (TZ §8).

These are the canonical list-scheduling priorities: averaged exec/comm costs,
distinct from TaskDAG's structural base_cost-weighted b_level/t_level.
"""

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.env.cost_model import exec_time


def _mean_bandwidth(nodes: list[ComputeNode]) -> float:
    return sum(n.bandwidth for n in nodes) / len(nodes)


def mean_exec(dag: TaskDAG, tid: int, nodes: list[ComputeNode]) -> float:
    task = dag.task(tid)
    return sum(exec_time(task, n) for n in nodes) / len(nodes)


def mean_comm(dag: TaskDAG, src: int, dst: int, nodes: list[ComputeNode]) -> float:
    return dag.edge_data(src, dst) / _mean_bandwidth(nodes)


def upward_rank(dag: TaskDAG, nodes: list[ComputeNode]) -> dict[int, float]:
    memo: dict[int, float] = {}

    def ru(i: int) -> float:
        if i in memo:
            return memo[i]
        succ = dag.successors(i)
        if not succ:
            value = mean_exec(dag, i, nodes)
        else:
            value = mean_exec(dag, i, nodes) + max(
                mean_comm(dag, i, j, nodes) + ru(j) for j in succ
            )
        memo[i] = value
        return value

    for i in range(dag.n_tasks):
        ru(i)
    return memo


def downward_rank(dag: TaskDAG, nodes: list[ComputeNode]) -> dict[int, float]:
    memo: dict[int, float] = {}

    def rd(i: int) -> float:
        if i in memo:
            return memo[i]
        pred = dag.predecessors(i)
        if not pred:
            value = 0.0
        else:
            value = max(rd(p) + mean_exec(dag, p, nodes) + mean_comm(dag, p, i, nodes) for p in pred)
        memo[i] = value
        return value

    for i in range(dag.n_tasks):
        rd(i)
    return memo
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ranking.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: cluster-aware HEFT/CPOP rank utilities"
```

---

### Task 3: HEFT strategy

**Files:**
- Create: `src/strategies/heft.py`
- Test: `tests/test_heft.py`

**Interfaces:**
- Consumes: `BaseSchedulingStrategy`, `ranking.upward_rank`, `placement.earliest_start_finish`, `ClusterState`.
- Produces: `heft.HEFTStrategy()` — `predict` selects the ready task with the highest `upward_rank` (ties → lowest id), then the alive node minimising EFT finish time (ties → lowest index). Ranks cached per-DAG via `WeakKeyDictionary`.

- [ ] **Step 1: Write the failing test**

`tests/test_heft.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.scheduler.task_scheduler import run_episode
from src.strategies.heft import HEFTStrategy


def _asym() -> tuple[TaskDAG, list[ComputeNode]]:
    # 0->1->3, 0->2->3 ; branch via 1 is heavy (base 6) => critical
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 6.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(3, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_heft_first_pick_is_highest_rank_on_fastest_node() -> None:
    dag, nodes = _asym()
    env = ClusterEnv(load_config_path())
    env.reset(dag=dag, nodes=nodes)
    task_id, node_id = HEFTStrategy().predict([0], env.state)
    assert task_id == 0  # only ready task, highest upward rank
    assert node_id == 1  # GPU finishes task0 at 1.0 vs CPU at 2.0


def test_heft_full_schedule_golden() -> None:
    dag, nodes = _asym()
    env = ClusterEnv(load_config_path())
    schedule, info = run_episode(env, HEFTStrategy(), dag=dag, nodes=nodes)
    node_of = {a.task_id: a.node_id for a in schedule.assignments}
    # HEFT (hand-computed): t0,t1,t3 -> GPU(1); t2 -> CPU(0)
    assert node_of == {0: 1, 1: 1, 2: 0, 3: 1}
    assert info["makespan"] == 6.0


def load_config_path():  # noqa: ANN201 - tiny test helper
    from src.utils.config import load_config

    return load_config("config.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_heft.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.strategies.heft`.

- [ ] **Step 3: Implement `src/strategies/heft.py`**

```python
"""HEFT: list-scheduling by upward rank + earliest-finish-time assignment (TZ §8).

No-insertion (append-only EFT) form — stated honestly in the thesis. Ranks are
cluster-aware (averaged over processors) and cached per DAG.
"""

from weakref import WeakKeyDictionary

from src.core.dag import TaskDAG
from src.env.placement import ClusterState, earliest_start_finish
from src.strategies.base import BaseSchedulingStrategy
from src.strategies.ranking import upward_rank


class HEFTStrategy(BaseSchedulingStrategy):
    def __init__(self) -> None:
        self._rank_cache: WeakKeyDictionary[TaskDAG, dict[int, float]] = WeakKeyDictionary()

    def _ranks(self, state: ClusterState) -> dict[int, float]:
        ranks = self._rank_cache.get(state.dag)
        if ranks is None:
            ranks = upward_rank(state.dag, state.nodes)
            self._rank_cache[state.dag] = ranks
        return ranks

    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        ranks = self._ranks(state)
        task_id = max(ready, key=lambda t: ranks[t])  # ties -> lowest id (ready is sorted)
        task = state.dag.task(task_id)
        alive = [n for n in state.nodes if n.alive]
        node = min(alive, key=lambda n: earliest_start_finish(task, n, state)[1])
        return task_id, node.node_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_heft.py -v`
Expected: 2 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: faithful HEFT strategy (upward-rank ordering + EFT)"
```

---

### Task 4: CPOP strategy

**Files:**
- Create: `src/strategies/cpop.py`
- Test: `tests/test_cpop.py`

**Interfaces:**
- Consumes: `BaseSchedulingStrategy`, `ranking.upward_rank`, `ranking.downward_rank`, `cost_model.exec_time`, `placement.earliest_start_finish`, `ClusterState`.
- Produces: `cpop.CPOPStrategy()` — priority `= rank_u + rank_d`; the critical-path set is the tasks whose priority equals the max priority (within `1e-6`); the critical-path processor is the node minimising the summed exec time of the CP tasks. `predict` selects the highest-priority ready task (ties → lowest id); CP tasks go to the CP processor, others to the min-EFT alive node. Structure cached per-DAG via `WeakKeyDictionary`.

- [ ] **Step 1: Write the failing test**

`tests/test_cpop.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.scheduler.task_scheduler import run_episode
from src.strategies.cpop import CPOPStrategy, critical_path_processor, critical_path_set
from src.strategies.ranking import downward_rank, upward_rank
from src.utils.config import load_config


def _asym() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 6.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(3, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_critical_path_set_is_heavy_branch() -> None:
    dag, nodes = _asym()
    ru = upward_rank(dag, nodes)
    rd = downward_rank(dag, nodes)
    cp = critical_path_set(ru, rd)
    # priorities: t0=9.5, t1=9.5, t2=6.5, t3=9.5 -> CP = {0,1,3}, t2 excluded
    assert cp == {0, 1, 3}


def test_critical_path_processor_is_gpu() -> None:
    dag, nodes = _asym()
    ru = upward_rank(dag, nodes)
    rd = downward_rank(dag, nodes)
    cp = critical_path_set(ru, rd)
    # sum exec over {0,1,3}: CPU=2+6+2=10, GPU=1+3+1=5 -> GPU
    assert critical_path_processor(dag, cp, nodes) == 1


def test_cpop_full_schedule_golden() -> None:
    dag, nodes = _asym()
    env = ClusterEnv(load_config("config.yaml"))
    schedule, info = run_episode(env, CPOPStrategy(), dag=dag, nodes=nodes)
    node_of = {a.task_id: a.node_id for a in schedule.assignments}
    # CP {0,1,3} -> GPU(1); non-CP t2 -> EFT picks CPU(0)
    assert node_of == {0: 1, 1: 1, 2: 0, 3: 1}
    assert info["makespan"] == 6.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cpop.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.strategies.cpop`.

- [ ] **Step 3: Implement `src/strategies/cpop.py`**

```python
"""CPOP: critical-path-on-a-processor list scheduling (TZ §8).

Priority = upward_rank + downward_rank. Tasks whose priority equals the maximum
form the critical path and are bound to a single critical-path processor (the
node minimising their total exec time); other tasks use EFT. No-insertion form.
"""

from weakref import WeakKeyDictionary

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.env.cost_model import exec_time
from src.env.placement import ClusterState, earliest_start_finish
from src.strategies.base import BaseSchedulingStrategy
from src.strategies.ranking import downward_rank, upward_rank

_CP_TOL = 1e-6


def critical_path_set(
    upward: dict[int, float], downward: dict[int, float]
) -> set[int]:
    priority = {i: upward[i] + downward[i] for i in upward}
    cp_value = max(priority.values())
    return {i for i, p in priority.items() if abs(p - cp_value) < _CP_TOL}


def critical_path_processor(
    dag: TaskDAG, cp: set[int], nodes: list[ComputeNode]
) -> int:
    def cp_cost(node: ComputeNode) -> float:
        return sum(exec_time(dag.task(i), node) for i in cp)

    return min(nodes, key=cp_cost).node_id


class CPOPStrategy(BaseSchedulingStrategy):
    def __init__(self) -> None:
        self._cache: WeakKeyDictionary[TaskDAG, tuple[dict[int, float], set[int], int]] = (
            WeakKeyDictionary()
        )

    def _structure(self, state: ClusterState) -> tuple[dict[int, float], set[int], int]:
        cached = self._cache.get(state.dag)
        if cached is None:
            ru = upward_rank(state.dag, state.nodes)
            rd = downward_rank(state.dag, state.nodes)
            priority = {i: ru[i] + rd[i] for i in ru}
            cp = critical_path_set(ru, rd)
            cp_proc = critical_path_processor(state.dag, cp, state.nodes)
            cached = (priority, cp, cp_proc)
            self._cache[state.dag] = cached
        return cached

    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        priority, cp, cp_proc = self._structure(state)
        task_id = max(ready, key=lambda t: priority[t])  # ties -> lowest id
        if task_id in cp:
            return task_id, cp_proc
        task = state.dag.task(task_id)
        alive = [n for n in state.nodes if n.alive]
        node = min(alive, key=lambda n: earliest_start_finish(task, n, state)[1])
        return task_id, node.node_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_cpop.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: faithful CPOP strategy (critical-path processor)"
```

---

### Task 5: Min-Min strategy

**Files:**
- Create: `src/strategies/min_min.py`
- Test: `tests/test_min_min.py`

**Interfaces:**
- Consumes: `BaseSchedulingStrategy`, `placement.earliest_start_finish`, `ClusterState`.
- Produces: `min_min.MinMinStrategy()` — over all `(ready task, alive node)` pairs, selects the pair with the globally minimum earliest finish time (ties → lowest task id, then lowest node id).

- [ ] **Step 1: Write the failing test**

`tests/test_min_min.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.scheduler.task_scheduler import run_episode
from src.strategies.min_min import MinMinStrategy
from src.utils.config import load_config


def _independent() -> tuple[TaskDAG, list[ComputeNode]]:
    # three independent tasks (no edges), two equal-speed nodes
    tasks = [
        Task(0, 1.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 3.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
    ]
    return dag, nodes


def test_min_min_first_pick_is_smallest_task() -> None:
    dag, nodes = _independent()
    env = ClusterEnv(load_config("config.yaml"))
    env.reset(dag=dag, nodes=nodes)
    task_id, node_id = MinMinStrategy().predict([0, 1, 2], env.state)
    # smallest min-completion = task0 (cost 1) on node0 (tie -> lowest index)
    assert (task_id, node_id) == (0, 0)


def test_min_min_full_schedule_golden() -> None:
    dag, nodes = _independent()
    env = ClusterEnv(load_config("config.yaml"))
    schedule, info = run_episode(env, MinMinStrategy(), dag=dag, nodes=nodes)
    placed = {(a.task_id, a.node_id, a.start, a.finish) for a in schedule.assignments}
    # Hand-computed Min-Min: t0->n0[0,1], t1->n1[0,2], t2->n0[1,4]
    assert (0, 0, 0.0, 1.0) in placed
    assert (1, 1, 0.0, 2.0) in placed
    assert (2, 0, 1.0, 4.0) in placed
    assert info["makespan"] == 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_min_min.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.strategies.min_min`.

- [ ] **Step 3: Implement `src/strategies/min_min.py`**

```python
"""Min-Min: pick the (task, node) with the globally smallest finish time (TZ §8)."""

from src.env.placement import ClusterState, earliest_start_finish
from src.strategies.base import BaseSchedulingStrategy


class MinMinStrategy(BaseSchedulingStrategy):
    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        best: tuple[float, int, int] | None = None  # (finish, task_id, node_id)
        for task_id in ready:
            task = state.dag.task(task_id)
            for node in state.nodes:
                if not node.alive:
                    continue
                _, finish = earliest_start_finish(task, node, state)
                if best is None or finish < best[0]:
                    best = (finish, task_id, node.node_id)
        assert best is not None
        return best[1], best[2]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_min_min.py -v`
Expected: 2 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: faithful Min-Min strategy"
```

---

### Task 6: Weighted-Sum Greedy strategy

**Files:**
- Create: `src/strategies/weighted_sum_greedy.py`
- Test: `tests/test_weighted_sum_greedy.py`

**Interfaces:**
- Consumes: `BaseSchedulingStrategy`, `placement.weighted_cost`, `ClusterState`.
- Produces: `weighted_sum_greedy.WeightedSumGreedyStrategy(w1: float, w2: float)` — over all `(ready task, alive node)` pairs, selects the argmin of `w1·d_makespan_norm + w2·d_energy_norm` from `placement.weighted_cost` (ties → lowest task id, then lowest node id). Same objective the env reward negates (fairness parity).

- [ ] **Step 1: Write the failing test**

`tests/test_weighted_sum_greedy.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.env.placement import weighted_cost
from src.scheduler.task_scheduler import run_episode
from src.strategies.weighted_sum_greedy import WeightedSumGreedyStrategy
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


def test_greedy_picks_argmin_of_weighted_cost() -> None:
    dag, nodes = _instance()
    env = ClusterEnv(load_config("config.yaml"))
    env.reset(dag=dag, nodes=nodes)
    w1, w2 = 1.0, 0.3
    chosen = WeightedSumGreedyStrategy(w1, w2).predict([0], env.state)
    # brute-force the same objective over (task 0, each node)
    costs = {}
    for node in env.state.nodes:
        c = weighted_cost(env.state.dag.task(0), node, env.state)
        costs[node.node_id] = w1 * c.d_makespan_norm + w2 * c.d_energy_norm
    best_node = min(costs, key=costs.get)
    assert chosen == (0, best_node)


def test_greedy_completes_episode() -> None:
    dag, nodes = _instance()
    env = ClusterEnv(load_config("config.yaml"))
    schedule, info = run_episode(env, WeightedSumGreedyStrategy(1.0, 0.3), dag=dag, nodes=nodes)
    assert len(schedule.assignments) == 4
    assert info["makespan"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_weighted_sum_greedy.py -v`
Expected: FAIL with `ModuleNotFoundError` for `src.strategies.weighted_sum_greedy`.

- [ ] **Step 3: Implement `src/strategies/weighted_sum_greedy.py`**

```python
"""Weighted-Sum Greedy: myopic argmin of the SAME objective the env rewards (TZ §8).

The scientific control isolating "RL wins by learning" from "RL wins only because
heuristics ignore energy". Uses placement.weighted_cost verbatim, so parity with
the env reward is structural. Balance is terminal-only and never chased.
"""

from src.env.placement import ClusterState, weighted_cost
from src.strategies.base import BaseSchedulingStrategy


class WeightedSumGreedyStrategy(BaseSchedulingStrategy):
    def __init__(self, w1: float, w2: float) -> None:
        self._w1 = w1
        self._w2 = w2

    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        best: tuple[float, int, int] | None = None  # (cost, task_id, node_id)
        for task_id in ready:
            task = state.dag.task(task_id)
            for node in state.nodes:
                if not node.alive:
                    continue
                comp = weighted_cost(task, node, state)
                cost = self._w1 * comp.d_makespan_norm + self._w2 * comp.d_energy_norm
                if best is None or cost < best[0]:
                    best = (cost, task_id, node.node_id)
        assert best is not None
        return best[1], best[2]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_weighted_sum_greedy.py -v`
Expected: 2 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: Weighted-Sum Greedy control (shares env objective)"
```

---

### Task 7: Fairness invariant + greedy↔reward parity tests

**Files:**
- Test: `tests/test_fairness.py`

**Interfaces:**
- Consumes: all strategies, `run_episode`, `ClusterEnv`, `placement.weighted_cost`, `load_config`, `make_rng`.
- Produces: integration tests proving (a) every strategy run on the *same* instance through the *same* env produces a complete, valid schedule covering each task exactly once, and (b) the greedy's chosen action minimises exactly the quantity the env reward negates (parity), with no terminal-balance contamination on a non-terminal step.

- [ ] **Step 1: Write the test**

`tests/test_fairness.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.env.placement import weighted_cost
from src.scheduler.task_scheduler import run_episode
from src.strategies.cpop import CPOPStrategy
from src.strategies.heft import HEFTStrategy
from src.strategies.min_min import MinMinStrategy
from src.strategies.random_strategy import RandomStrategy
from src.strategies.weighted_sum_greedy import WeightedSumGreedyStrategy
from src.utils.config import load_config
from src.utils.seeding import make_rng


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 6.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(3, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def _all_strategies() -> list:
    cfg = load_config("config.yaml")
    return [
        HEFTStrategy(),
        CPOPStrategy(),
        MinMinStrategy(),
        WeightedSumGreedyStrategy(cfg.w1, cfg.w2),
        RandomStrategy(make_rng(0)),
    ]


def test_all_strategies_produce_valid_complete_schedules_on_same_instance() -> None:
    dag, nodes = _instance()
    for strategy in _all_strategies():
        env = ClusterEnv(load_config("config.yaml"))
        schedule, info = run_episode(env, strategy, dag=dag, nodes=nodes)
        assert sorted(a.task_id for a in schedule.assignments) == [0, 1, 2, 3], strategy
        assert info["makespan"] > 0.0


def test_greedy_choice_matches_env_reward_objective() -> None:
    # The action the greedy picks must be the global argmin of the SAME objective
    # the env reward negates; on a non-terminal step reward == -(w1*dmk + w2*den).
    dag, nodes = _instance()
    cfg = load_config("config.yaml")
    env = ClusterEnv(load_config("config.yaml"))
    env.reset(dag=dag, nodes=nodes)

    greedy = WeightedSumGreedyStrategy(cfg.w1, cfg.w2)
    ready = env.state.dag.ready_set(env.scheduled)
    task_id, node_id = greedy.predict(ready, env.state)

    # objective the greedy minimised for its choice
    comp = weighted_cost(env.state.dag.task(task_id), env.state.nodes[node_id], env.state)
    greedy_obj = cfg.w1 * comp.d_makespan_norm + cfg.w2 * comp.d_energy_norm

    # it must be the global minimum over all (ready, alive node) pairs
    all_objs = []
    for t in ready:
        for n in env.state.nodes:
            c = weighted_cost(env.state.dag.task(t), n, env.state)
            all_objs.append(cfg.w1 * c.d_makespan_norm + cfg.w2 * c.d_energy_norm)
    assert greedy_obj == min(all_objs)

    # stepping that action yields reward == -greedy_obj (first step is non-terminal)
    _, reward, done, _ = env.step((task_id, node_id))
    assert done is False
    assert abs(reward - (-greedy_obj)) < 1e-9


def test_same_instance_gives_identical_inputs_across_strategies() -> None:
    # Fairness: the env builds the identical first decision-point state regardless
    # of strategy, because dag+nodes are the same objects and reset is deterministic.
    dag, nodes = _instance()
    env_a = ClusterEnv(load_config("config.yaml"))
    env_b = ClusterEnv(load_config("config.yaml"))
    _, info_a = env_a.reset(dag=dag, nodes=nodes)
    _, info_b = env_b.reset(dag=dag, nodes=nodes)
    assert info_a["m_ref"] == info_b["m_ref"]
    assert info_a["e_ref"] == info_b["e_ref"]
    assert env_a.state.dag.ready_set(env_a.scheduled) == env_b.state.dag.ready_set(env_b.scheduled)
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/pytest tests/test_fairness.py -v`
Expected: 3 passed. (If `test_greedy_choice_matches_env_reward_objective` fails, the bug is a divergence between the greedy objective and the env reward — fix the strategy/env to share `weighted_cost`, not the test.)

- [ ] **Step 3: Run the full suite + lint/format**

Run:
```bash
.venv/bin/pytest -q && .venv/bin/ruff check . && .venv/bin/black --check .
```
Expected: all tests pass (M1's 46 + M2's new tests); ruff clean; black reports no changes.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "test: fairness invariant + greedy/reward parity"
```

---

## Self-Review

**Spec coverage (roadmap M2 + TZ §8):**
- `BaseSchedulingStrategy.predict(ready, state)` interface — Task 1. ✅
- Decision-point loop driving every strategy through the same `ClusterEnv.step` — Task 1 (`run_episode`). ✅
- SystemMonitor Observer scaffold (idle in deterministic mode, wired into the loop) — Task 1. ✅
- HEFT (upward-rank ordering + EFT, faithful, golden) — Tasks 2 + 3. ✅
- CPOP (rank_u+rank_d priority, critical-path processor, faithful, golden) — Tasks 2 + 4. ✅
- Min-Min (global-min-completion rule, faithful, golden) — Task 5. ✅
- Weighted-Sum Greedy reusing `placement.weighted_cost` for parity — Task 6. ✅
- Random floor — Task 1. ✅
- `ranking.py` (mean_exec/mean_comm/upward_rank/downward_rank) — Task 2. ✅
- WeakKeyDictionary per-DAG caches for HEFT/CPOP — Tasks 3, 4. ✅
- Fairness invariant guarded in tests (`test_fairness`); greedy↔reward parity — Task 7. ✅
- Heuristics faithful (no energy/balance terms; no-insertion EFT) — Tasks 3–5 (time-only EFT). ✅
- node_id==index, isolated rng — Task 1 (RandomStrategy rng), all strategies return node_id. ✅

**Placeholder scan:** no TBD/TODO; every code step has complete code. The only test helper named loosely is `load_config_path()` in `tests/test_heft.py`, which has a real one-line body. ✅

**Type consistency:** `predict(ready: list[int], state: ClusterState) -> tuple[int, int]` is identical across base/random/heft/cpop/min_min/weighted_sum_greedy. `run_episode(...) -> tuple[Schedule, dict]` consistent with its callers in tests. `upward_rank`/`downward_rank` return `dict[int, float]`, consumed as such by heft/cpop. `critical_path_set(upward, downward) -> set[int]` and `critical_path_processor(dag, cp, nodes) -> int` match their test imports. ✅

**Golden values verified by hand** (documented in the test comments): hetero-diamond upward/downward ranks (Task 2); asymmetric-DAG HEFT schedule `{0:1,1:1,2:0,3:1}`, makespan 6.0 (Task 3); CPOP CP set `{0,1,3}`, CP processor GPU(1), same schedule (Task 4); independent-tasks Min-Min schedule makespan 4.0 (Task 5). These were computed from the algorithm definitions, not the implementation.

**Deferred (consistent with roadmap):** failure reactivity (strategies re-assigning onto survivors) and the CPOP dead-CP-processor fallback are M4 — M2 strategies assume all nodes alive (deterministic mode), which the `if not node.alive: continue` filters already accommodate structurally without speculative failure code.
