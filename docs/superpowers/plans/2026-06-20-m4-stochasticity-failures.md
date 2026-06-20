# M4 — Stochasticity & Failures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add execution-time noise and exogenous node failures to the existing append-only `ClusterEnv` (config knobs `noise_std`, `failure_rate`), activate the `SystemMonitor` Observer, and make every strategy react to the live survivor state — while `(0,0)` reproduces the M1 golden schedule bit-for-bit.

**Architecture:** At `reset()`, two **isolated RNG streams** (keyed by `config.seed` + a per-instance signature) produce a per-task noise map `ε[task_id]` and a per-node **failure calendar** `t_f[node_id] ~ Exp(failure_rate)`, cached on `ClusterState`. In `step()`, the actual (noisy) duration is revealed only at commit (planning/reward stay nominal); a placement on node X **fails iff its actual finish exceeds `t_f[X]`**, in which case X dies, the task is requeued (nothing committed, reward 0), and `SystemMonitor` reports it. Strategies re-assign onto survivors through their existing `predict`.

**Tech Stack:** Python 3.10+, numpy, networkx, torch/PyG (only via existing RL code), pytest, ruff, black.

## Global Constraints

Copied from `docs/superpowers/specs/2026-06-20-m4-stochasticity-failures-design.md` (the approved design) + `SmartDAG_Scheduler_TZ.md` §5.2/§8/§14:

- **Python 3.10+**; full type hints; `ruff` + `black` clean; tooling via `.venv/bin/...`.
- **γ = 1.0** (unchanged). **No event-queue / running clock** — failures are an exogenous calendar checked at placement (append-only model). **No partial-work charging** — a failed attempt commits nothing. **One failure per node per episode.**
- **Failure calendar:** per-node `t_f ~ Exponential(failure_rate)` drawn at `reset()` from an **isolated failure RNG**; `failure_rate==0 ⇒ t_f=+inf`. Strategy-independent, cached on `ClusterState`. Keyed by `(config.seed, instance signature)` so the same instance+seed yields the same calendar (fairness) and different instances yield different calendars.
- **Placement-failure rule:** a placement of task T on node X with window `[start, actual_finish]` succeeds iff `actual_finish <= t_f[X]`; otherwise X dies (`alive=False`), T is **not** committed (stays ready / requeued), `SystemMonitor` reports it, **reward = 0** (no Δmakespan computed). Lazy detection.
- **Noise:** per-task `ε[task_id]` drawn at `reset()` from an **isolated noise RNG** (keyed by task_id, not draw order); `actual_exec = max(0, nominal_exec·(1+ε))`; revealed only at commit. Planning/observation/`weighted_cost`/reward use **nominal**; only the committed `Assignment.finish`, `node.free_at_time`, and committed energy use **actual**. `noise_std==0 ⇒ ε=0`.
- **Reward:** successful step unchanged (`−(w1·Δmk_nominal/M_ref + w2·Δen_nominal/E_ref)` from `weighted_cost`; terminal `+w3·balance`). Failed step = `0.0`. `weighted_cost` stays nominal (greedy parity preserved).
- **Horizon** (for reward + makespan): `max(free_at_time)` over **all** nodes with committed work (dead nodes' completed tasks still elapsed). **Placement candidates remain alive-only** (strategies/env filter).
- **`load_balance_index`** computes `1 − CV` over the **actual alive node ids** (not `range(n_alive)`).
- **Deadlock:** unscheduled tasks remain AND zero alive nodes ⇒ `done=True`, `info["deadlocked"]=True`, makespan = partial.
- **RNG isolation is load-bearing:** noise/failure draw from streams independent of the DAG/cluster-generation stream, so `(0,0)` reproduces the M1 golden schedule **bit-for-bit**.
- **Non-goals:** event-queue simulator; mid-execution concurrent failures; partial-work charging; cascading failures; retraining under regimes (M5); the eval grid / Wilcoxon / CLI / WfCommons (M5).

### Interfaces consumed (current, on `main`)

- `src.env.placement.ClusterState(nodes, dag, task_finish, task_node, m_ref, e_ref, sim_time=0.0)`; `earliest_start_finish(task,node,state)->(start,finish)`; `weighted_cost(...)->CostComponents`; `horizon(nodes)->float`.
- `src.env.cluster_env.ClusterEnv` (`reset`, `step` 4-tuple, `.state`, `.schedule`, `.scheduled`).
- `src.core.schedule.Schedule` (`add(assignment, energy)`, `makespan()`, `busy_time_by_node()`, `load_balance_index(...)`); `Assignment(task_id,node_id,start,finish)`.
- `src.env.cost_model.{exec_time, energy}`; `src.env.observation.build_observation(state, scheduled, current_makespan)`.
- `src.scheduler.system_monitor.SystemMonitor`; `src.scheduler.task_scheduler.run_episode(env, strategy, dag, nodes, monitor)`.
- `src.utils.seeding.make_rng`; `src.utils.config.Config` (has `noise_std`, `failure_rate`, `seed`).

---

### Task 1: Isolated per-concern RNG helper

**Files:**
- Modify: `src/utils/seeding.py`
- Test: `tests/test_seeding.py`

**Interfaces:**
- Produces: `seeding.derive_rng(seed: int, salt: str) -> numpy.random.Generator` — an independent sub-stream keyed by `(seed, salt)` that never consumes from `make_rng(seed)` or any other salt.

- [ ] **Step 1: Write the failing test**

`tests/test_seeding.py`:
```python
import numpy as np

from src.utils.seeding import derive_rng, make_rng


def test_derive_rng_is_reproducible() -> None:
    a = derive_rng(0, "noise").random(5)
    b = derive_rng(0, "noise").random(5)
    assert np.array_equal(a, b)


def test_derive_rng_streams_are_independent_per_salt() -> None:
    noise = derive_rng(0, "noise").random(5)
    failure = derive_rng(0, "failure").random(5)
    assert not np.array_equal(noise, failure)


def test_derive_rng_does_not_consume_base_stream() -> None:
    # Drawing from a derived stream must not perturb make_rng(seed)'s sequence.
    base_first = make_rng(0).random(3)
    _ = derive_rng(0, "noise").random(3)
    base_again = make_rng(0).random(3)
    assert np.array_equal(base_first, base_again)


def test_derive_rng_returns_generator() -> None:
    assert isinstance(derive_rng(1, "x"), np.random.Generator)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_seeding.py -v`
Expected: FAIL — `ImportError: cannot import name 'derive_rng'`.

- [ ] **Step 3: Implement `derive_rng` in `src/utils/seeding.py`**

```python
"""Isolated RNG helpers (never touch global numpy state)."""

import zlib

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return an independent generator seeded deterministically."""
    return np.random.default_rng(seed)


def derive_rng(seed: int, salt: str) -> np.random.Generator:
    """Independent sub-stream keyed by (seed, salt) for per-concern RNG isolation.

    Drawing from a derived stream never consumes from make_rng(seed) or any other
    salt's stream, so toggling one concern (e.g. noise) cannot perturb another's
    draws or the base DAG/cluster-generation stream.
    """
    sub = zlib.crc32(salt.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence([seed, sub]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_seeding.py -v`
Expected: 4 passed.

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: derive_rng for per-concern RNG isolation (M4)"
```

---

### Task 2: `load_balance_index` over actual alive node ids

**Files:**
- Modify: `src/core/schedule.py`
- Modify: `tests/test_schedule.py` (update the two existing calls to the new signature)
- Test: `tests/test_load_balance_alive_ids.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Schedule.load_balance_index(self, alive_node_ids: list[int]) -> float` — `1 − CV(busy time)` over exactly the given node ids (idle ones count as 0), clamped `[0,1]`; `[]` ⇒ `0.0`. **Signature changed** from `n_alive_nodes: int` to `alive_node_ids: list[int]`.

- [ ] **Step 1: Write the failing test**

`tests/test_load_balance_alive_ids.py`:
```python
from src.core.schedule import Assignment, Schedule


def test_balance_over_noncontiguous_alive_ids() -> None:
    # nodes 0 and 2 each busy 5.0; node 1 is dead and absent.
    s = Schedule(n_nodes=3)
    s.add(Assignment(0, 0, 0.0, 5.0), energy=1.0)
    s.add(Assignment(1, 2, 0.0, 5.0), energy=1.0)
    # Over the actual alive ids {0,2} the two are perfectly balanced -> 1.0
    assert s.load_balance_index([0, 2]) == 1.0


def test_dead_midrange_node_excluded() -> None:
    # node 0 busy 10, node 2 idle; alive ids {0,2}. busy=[10,0] -> CV 1 -> 0.0
    s = Schedule(n_nodes=3)
    s.add(Assignment(0, 0, 0.0, 10.0), energy=1.0)
    assert s.load_balance_index([0, 2]) == 0.0


def test_empty_alive_ids_returns_zero() -> None:
    assert Schedule(n_nodes=2).load_balance_index([]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_load_balance_alive_ids.py -v`
Expected: FAIL — `load_balance_index` still takes an int (`TypeError`/wrong result).

- [ ] **Step 3: Update `load_balance_index` in `src/core/schedule.py`**

Replace the existing method with:
```python
    def load_balance_index(self, alive_node_ids: list[int]) -> float:
        """1 - CV(busy time) over the given alive node ids; idle nodes count as 0."""
        if not alive_node_ids:
            return 0.0
        busy = self.busy_time_by_node()
        times = [busy.get(i, 0.0) for i in alive_node_ids]
        k = len(alive_node_ids)
        mean = sum(times) / k
        if mean == 0.0:
            return 1.0  # nothing scheduled yet: treat as perfectly even
        variance = sum((t - mean) ** 2 for t in times) / k
        cv = (variance**0.5) / mean
        return max(0.0, min(1.0, 1.0 - cv))
```

- [ ] **Step 4: Update the two existing calls in `tests/test_schedule.py`**

Change `s.load_balance_index(2)` → `s.load_balance_index([0, 1])` in both `test_load_balance_perfectly_even_is_one` and `test_load_balance_fully_skewed_is_zero` (the dense `range(2)` is exactly `[0, 1]`, so the asserted results are unchanged).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_load_balance_alive_ids.py tests/test_schedule.py -v`
Expected: all passed.

- [ ] **Step 6: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: load_balance_index over actual alive node ids (M4 carry-forward)"
```

---

### Task 3: ClusterState calendar fields + horizon over all committed nodes

**Files:**
- Modify: `src/env/placement.py`
- Test: `tests/test_placement_m4.py`

**Interfaces:**
- Produces:
  - `ClusterState` gains `failure_times: dict[int, float] = {}` and `noise_eps: dict[int, float] = {}` (both `field(default_factory=dict)`, so existing constructions are unaffected).
  - `horizon(nodes)` now returns `max(free_at_time)` over **all** nodes (not alive-filtered).

- [ ] **Step 1: Write the failing test**

`tests/test_placement_m4.py`:
```python
from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.placement import ClusterState, horizon


def _nodes() -> list[ComputeNode]:
    return [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]


def test_cluster_state_calendar_fields_default_empty() -> None:
    dag = TaskDAG([Task(0, 1.0, 1.0, TaskClass.SEQUENTIAL)], [])
    st = ClusterState(nodes=_nodes(), dag=dag, task_finish={}, task_node={}, m_ref=1.0, e_ref=1.0)
    assert st.failure_times == {} and st.noise_eps == {}


def test_horizon_includes_dead_node_committed_work() -> None:
    nodes = _nodes()
    nodes[1].free_at_time = 9.0
    nodes[1].alive = False  # dead, but its committed work elapsed to t=9
    nodes[0].free_at_time = 4.0
    assert horizon(nodes) == 9.0  # dead node still counts toward the horizon
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_placement_m4.py -v`
Expected: FAIL — `ClusterState` has no `failure_times`; `horizon` excludes the dead node (returns 4.0).

- [ ] **Step 3: Edit `src/env/placement.py`**

Change the import line `from dataclasses import dataclass` to:
```python
from dataclasses import dataclass, field
```
Add the two fields at the end of the `ClusterState` dataclass (after `sim_time`):
```python
    failure_times: dict[int, float] = field(default_factory=dict)
    noise_eps: dict[int, float] = field(default_factory=dict)
```
Replace `horizon`:
```python
def horizon(nodes: list[ComputeNode]) -> float:
    """Running schedule horizon = max free_at_time over ALL nodes with committed work.

    Dead nodes are included: their completed tasks still elapsed and count toward the
    makespan. Placement candidates are filtered to alive nodes by the strategies/env.
    """
    return max((n.free_at_time for n in nodes), default=0.0)
```

- [ ] **Step 4: Run the new test + the placement/M1 suite**

Run: `.venv/bin/pytest tests/test_placement_m4.py tests/test_placement.py -v`
Expected: new tests pass; existing placement tests still pass (all-alive ⇒ horizon unchanged).

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: ClusterState failure/noise calendars + horizon over all committed nodes"
```

---

### Task 4: `reset()` builds the noise map + failure calendar

**Files:**
- Modify: `src/env/cluster_env.py`
- Test: `tests/test_env_reset_m4.py`

**Interfaces:**
- Consumes: `derive_rng`, `ClusterState.failure_times`/`noise_eps`, `config.noise_std`/`failure_rate`/`seed`.
- Produces:
  - `ClusterEnv._instance_signature(dag, nodes) -> int` (static) — deterministic crc32 over the instance's task costs/classes, edges, and node specs.
  - `reset()` populates `state.noise_eps` (`ε[task_id]`) and `state.failure_times` (`t_f[node_id]`) from isolated streams keyed by `(config.seed, instance signature)`. `noise_std==0 ⇒ all ε=0`; `failure_rate==0 ⇒ all t_f=+inf`.

- [ ] **Step 1: Write the failing test**

`tests/test_env_reset_m4.py`:
```python
import math
from dataclasses import replace

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.utils.config import load_config


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [Task(i, 2.0, 1.0, TaskClass.SEQUENTIAL) for i in range(4)]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_deterministic_knobs_zero_calendar() -> None:
    env = ClusterEnv(load_config("config.yaml"))  # noise_std=0, failure_rate=0
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    assert all(v == 0.0 for v in env.state.noise_eps.values())
    assert all(math.isinf(v) for v in env.state.failure_times.values())
    assert set(env.state.noise_eps) == {0, 1, 2, 3}
    assert set(env.state.failure_times) == {0, 1}


def test_stochastic_calendar_is_instance_seed_keyed_and_repeatable() -> None:
    cfg = replace(load_config("config.yaml"), noise_std=0.2, failure_rate=0.1)
    dag, nodes = _instance()
    a = ClusterEnv(cfg)
    a.reset(dag=dag, nodes=nodes)
    b = ClusterEnv(cfg)
    b.reset(dag=dag, nodes=nodes)
    # Same instance + same seed -> bit-identical calendar (the fairness property).
    assert a.state.noise_eps == b.state.noise_eps
    assert a.state.failure_times == b.state.failure_times
    # Stochastic knobs actually produced non-trivial values.
    assert any(v != 0.0 for v in a.state.noise_eps.values())
    assert all(v > 0.0 and v != float("inf") for v in a.state.failure_times.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_env_reset_m4.py -v`
Expected: FAIL — `state.noise_eps`/`failure_times` are empty (reset doesn't populate them yet).

- [ ] **Step 3: Edit `src/env/cluster_env.py`**

Add `zlib` import at the top and extend the seeding import:
```python
import zlib
```
```python
from src.utils.seeding import derive_rng, make_rng
```
Add this static method to `ClusterEnv` (e.g. after `_compute_e_ref`):
```python
    @staticmethod
    def _instance_signature(dag: TaskDAG, nodes: list[ComputeNode]) -> int:
        """Deterministic crc32 over the instance, so the calendar is keyed to it."""
        parts: list[str] = []
        for tid in range(dag.n_tasks):
            t = dag.task(tid)
            parts.append(f"t{tid}:{t.base_cost!r}:{t.mem_required!r}:{t.task_class.value}")
        for u, v in dag.edge_index():
            parts.append(f"e{u}-{v}:{dag.edge_data(u, v)!r}")
        for n in nodes:
            speeds = ",".join(
                f"{c.value}={n.speed_by_class[c]!r}"
                for c in sorted(n.speed_by_class, key=lambda c: c.value)
            )
            parts.append(f"n{n.node_id}:{n.node_type.value}:{n.power_w!r}:{n.bandwidth!r}:{speeds}")
        return zlib.crc32("|".join(parts).encode("utf-8"))
```
In `reset`, replace the `ClusterState(...)` construction block with calendar construction first:
```python
        sig = self._instance_signature(dag, nodes)
        noise_rng = derive_rng(self.config.seed, f"noise|{sig}")
        fail_rng = derive_rng(self.config.seed, f"failure|{sig}")
        if self.config.noise_std > 0.0:
            noise_eps = {
                tid: float(noise_rng.normal(0.0, self.config.noise_std))
                for tid in range(dag.n_tasks)
            }
        else:
            noise_eps = {tid: 0.0 for tid in range(dag.n_tasks)}
        if self.config.failure_rate > 0.0:
            scale = 1.0 / self.config.failure_rate
            failure_times = {n.node_id: float(fail_rng.exponential(scale)) for n in nodes}
        else:
            failure_times = {n.node_id: float("inf") for n in nodes}

        self.state = ClusterState(
            nodes=nodes,
            dag=dag,
            task_finish={},
            task_node={},
            m_ref=m_ref,
            e_ref=e_ref,
            failure_times=failure_times,
            noise_eps=noise_eps,
        )
```

- [ ] **Step 4: Run the new test + the full suite**

Run: `.venv/bin/pytest tests/test_env_reset_m4.py -v && .venv/bin/pytest -q`
Expected: new tests pass; full suite still green (existing reset/step/golden tests run at `(0,0)`, where `ε=0`/`t_f=inf`, and the calendar draws come from isolated streams so the deterministic schedule is unperturbed).

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: reset builds instance-keyed noise map + failure calendar (isolated RNG)"
```

---

### Task 5: `step()` — noise, failure, deadlock

**Files:**
- Modify: `src/env/cluster_env.py`
- Test: `tests/test_env_step_m4.py`

**Interfaces:**
- Consumes: `state.noise_eps`, `state.failure_times`, `weighted_cost` (nominal), `earliest_start_finish`, `energy`, `Schedule.load_balance_index(alive_ids)`.
- Produces a rewritten `ClusterEnv.step(action) -> (obs, reward, done, info)`:
  - nominal `weighted_cost` → reward (unchanged shape); nominal `(start, nominal_finish)` from `earliest_start_finish`.
  - `actual_exec = max(0, (nominal_finish-start)·(1+ε[task_id]))`, `actual_finish = start + actual_exec`.
  - **Failure** if `actual_finish > t_f[node_id]`: `node.alive=False`; nothing committed; `reward=0.0`; `info["failed_node"]=node_id`; deadlock check; return.
  - **Success** otherwise: commit `Assignment(task_id, node_id, start, actual_finish)` with `energy = power_w·actual_exec`; set `free_at_time/task_finish/task_node/sim_time` to actual; `scheduled.add`.
  - Terminal balance over **alive ids**. `info` always has `m_ref/e_ref/makespan/energy`; failure adds `failed_node`; terminal/deadlock adds `balance`/`deadlocked`.

- [ ] **Step 1: Write the failing test**

`tests/test_env_step_m4.py`:
```python
from dataclasses import replace

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.utils.config import load_config


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [Task(i, 2.0, 1.0, TaskClass.SEQUENTIAL) for i in range(4)]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_noise_changes_committed_finish_not_planning() -> None:
    cfg = replace(load_config("config.yaml"), noise_std=0.3, failure_rate=0.0)
    env = ClusterEnv(cfg)
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    eps0 = env.state.noise_eps[0]
    # task 0 on GPU(node 1): nominal exec = base/speed = 2/2 = 1.0
    env.step((0, 1))
    committed_finish = env.state.task_finish[0]
    assert abs(committed_finish - 1.0 * (1.0 + eps0)) < 1e-9  # actual = nominal*(1+eps)
    assert eps0 != 0.0  # noise was active


def test_failure_kills_node_requeues_task_episode_completes() -> None:
    cfg = replace(load_config("config.yaml"), failure_rate=0.1)
    env = ClusterEnv(cfg)
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    env.state.failure_times[1] = 0.5  # node 1 dies early; any task finishes after 0.5
    obs, reward, done, info = env.step((0, 1))  # task 0 on node 1: finish 1.0 > 0.5 -> FAIL
    assert reward == 0.0
    assert env.state.nodes[1].alive is False
    assert 0 not in env.scheduled  # not committed -> requeued
    assert info["failed_node"] == 1
    assert 0 in env.state.dag.ready_set(env.scheduled)  # ready again
    # re-assign onto the survivor (node 0) and finish the episode greedily on node 0
    done = False
    while not done:
        ready = env.state.dag.ready_set(env.scheduled)
        _, _, done, info = env.step((ready[0], 0))
    assert sorted(env.scheduled) == [0, 1, 2, 3]
    assert info["makespan"] > 0.0


def test_deadlock_when_all_nodes_die() -> None:
    cfg = replace(load_config("config.yaml"), failure_rate=0.1)
    env = ClusterEnv(cfg)
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    env.state.failure_times[0] = 0.0
    env.state.failure_times[1] = 0.0
    env.step((0, 1))  # node 1 dies; node 0 still alive -> not done
    obs, reward, done, info = env.step((0, 0))  # node 0 dies; no alive nodes, tasks remain
    assert done is True and info["deadlocked"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_env_step_m4.py -v`
Expected: FAIL — current `step` ignores noise/failures (commits nominal, never fails).

- [ ] **Step 3: Replace `step` in `src/env/cluster_env.py`**

```python
    def step(self, action: tuple[int, int]) -> tuple[Observation, float, bool, dict]:
        """One assignment. Returns the 4-tuple (obs, reward, done, info) per spec §5.3.

        Planning (reward, weighted_cost, observation) is on NOMINAL costs; the actual
        noisy duration is revealed only at commit. A placement fails iff its actual
        finish exceeds the node's exogenous failure time t_f (the node then dies and the
        task is requeued, reward 0). done=True at full completion or deadlock.
        """
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
        components = weighted_cost(task, node, state)  # NOMINAL planning
        start, nominal_finish = earliest_start_finish(task, node, state)
        eps = state.noise_eps.get(task_id, 0.0)
        actual_exec = max(0.0, (nominal_finish - start) * (1.0 + eps))
        actual_finish = start + actual_exec

        t_f = state.failure_times.get(node_id, float("inf"))
        if actual_finish > t_f:
            # Node dies before this task could finish: task lost (requeued), nothing committed.
            node.alive = False
            remaining = state.dag.n_tasks - len(self.scheduled)
            deadlocked = remaining > 0 and not any(n.alive for n in state.nodes)
            makespan = self.schedule.makespan()
            info: dict = {
                "m_ref": state.m_ref,
                "e_ref": state.e_ref,
                "makespan": makespan,
                "energy": self.schedule.total_energy,
                "failed_node": node_id,
                "deadlocked": deadlocked,
            }
            if deadlocked:
                alive_ids = [n.node_id for n in state.nodes if n.alive]
                info["balance"] = self.schedule.load_balance_index(alive_ids)
            obs = build_observation(state, self.scheduled, current_makespan=makespan)
            return obs, 0.0, deadlocked, info

        # Success: reward is nominal; commit uses the actual (noisy) finish/energy.
        reward = -(
            self.config.w1 * components.d_makespan_norm + self.config.w2 * components.d_energy_norm
        )
        actual_energy = node.power_w * actual_exec
        node.free_at_time = actual_finish
        state.task_finish[task_id] = actual_finish
        state.task_node[task_id] = node_id
        state.sim_time = max(state.sim_time, actual_finish)
        self.schedule.add(Assignment(task_id, node_id, start, actual_finish), energy=actual_energy)
        self.scheduled.add(task_id)

        done = len(self.scheduled) == state.dag.n_tasks
        makespan = self.schedule.makespan()
        info = {
            "m_ref": state.m_ref,
            "e_ref": state.e_ref,
            "makespan": makespan,
            "energy": self.schedule.total_energy,
        }
        if done:
            alive_ids = [n.node_id for n in state.nodes if n.alive]
            balance = self.schedule.load_balance_index(alive_ids)
            reward += self.config.w3 * balance
            info["balance"] = balance

        obs = build_observation(state, self.scheduled, current_makespan=makespan)
        return obs, reward, done, info
```

- [ ] **Step 4: Run the new test + the full suite (incl. golden regression)**

Run: `.venv/bin/pytest tests/test_env_step_m4.py -v && .venv/bin/pytest -q`
Expected: new M4 step tests pass; the full suite stays green — in particular `tests/test_golden_schedule.py`, `tests/test_env_step.py`, and `tests/test_reward_telescoping.py` are bit-for-bit unchanged at `(0,0)` (`ε=0` ⇒ actual==nominal; `t_f=inf` ⇒ no failure; `load_balance_index([0,1])` == old `load_balance_index(2)`).

- [ ] **Step 5: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: ClusterEnv.step with noise, exogenous failures, deadlock"
```

---

### Task 6: SystemMonitor activation + scheduler deadlock guard + cross-strategy fairness

**Files:**
- Modify: `src/scheduler/system_monitor.py`
- Modify: `src/scheduler/task_scheduler.py`
- Test: `tests/test_system_monitor.py`
- Test: `tests/test_stochastic_fairness.py`

**Interfaces:**
- Produces:
  - `SystemMonitor.check(state) -> list[int]` — the node ids that **newly** died since the last `check` (tracked via an internal `_seen_dead` set); notifies subscribers for each.
  - `run_episode` gains a deadlock guard: it never calls `predict` when no node is alive (ends the episode instead).

- [ ] **Step 1: Write the failing tests**

`tests/test_system_monitor.py`:
```python
from dataclasses import replace

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.scheduler.system_monitor import SystemMonitor
from src.utils.config import load_config


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [Task(i, 2.0, 1.0, TaskClass.SEQUENTIAL) for i in range(4)]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_monitor_reports_new_failure_once() -> None:
    env = ClusterEnv(replace(load_config("config.yaml"), failure_rate=0.1))
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    monitor = SystemMonitor()
    assert monitor.check(env.state) == []  # nothing dead yet
    env.state.failure_times[1] = 0.5
    env.step((0, 1))  # node 1 dies
    assert monitor.check(env.state) == [1]  # newly dead
    assert monitor.check(env.state) == []  # already reported


def test_monitor_notifies_subscribers() -> None:
    env = ClusterEnv(replace(load_config("config.yaml"), failure_rate=0.1))
    dag, nodes = _instance()
    env.reset(dag=dag, nodes=nodes)
    seen: list[int] = []
    monitor = SystemMonitor()
    monitor.subscribe(lambda _state: seen.append(1))
    env.state.failure_times[1] = 0.5
    env.step((0, 1))
    monitor.check(env.state)
    assert seen == [1]
```

`tests/test_stochastic_fairness.py`:
```python
from dataclasses import replace

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.utils.config import load_config


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [Task(i, 3.0, 1.0, TaskClass.SEQUENTIAL) for i in range(4)]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def test_calendar_and_noise_are_strategy_independent() -> None:
    # Same instance + seed => bit-identical adversity regardless of who schedules.
    cfg = replace(load_config("config.yaml"), noise_std=0.2, failure_rate=0.05)
    dag, nodes = _instance()
    env1 = ClusterEnv(cfg)
    env1.reset(dag=dag, nodes=nodes)
    env2 = ClusterEnv(cfg)
    env2.reset(dag=dag, nodes=nodes)
    assert env1.state.failure_times == env2.state.failure_times
    assert env1.state.noise_eps == env2.state.noise_eps
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_system_monitor.py tests/test_stochastic_fairness.py -v`
Expected: `test_system_monitor` fails (`check` returns `[]` always); `test_stochastic_fairness` should already pass from Task 4 (acceptable — it pins the cross-strategy property at the suite level).

- [ ] **Step 3: Update `src/scheduler/system_monitor.py`**

```python
"""SystemMonitor: the Observer that detects failure events (TZ §3, §8).

check() reports the nodes that have newly died since the previous call, so every
strategy reacts through the same uniform trigger (the fairness invariant).
"""

from collections.abc import Callable

from src.env.placement import ClusterState


class SystemMonitor:
    def __init__(self) -> None:
        self._subscribers: list[Callable[[ClusterState], None]] = []
        self._seen_dead: set[int] = set()

    def subscribe(self, callback: Callable[[ClusterState], None]) -> None:
        self._subscribers.append(callback)

    def check(self, state: ClusterState) -> list[int]:
        """Return the node ids that newly died since the last check; notify subscribers."""
        dead = {n.node_id for n in state.nodes if not n.alive}
        new_failures = sorted(dead - self._seen_dead)
        self._seen_dead = dead
        for _nid in new_failures:
            for callback in self._subscribers:
                callback(state)
        return new_failures
```

- [ ] **Step 4: Add the deadlock guard to `run_episode` in `src/scheduler/task_scheduler.py`**

Replace the `while not done:` loop body with:
```python
    while not done:
        if monitor is not None:
            monitor.check(env.state)
        if not any(n.alive for n in env.state.nodes):
            break  # deadlock: no surviving node can run the remaining tasks
        ready = env.state.dag.ready_set(env.scheduled)
        action = strategy.predict(ready, env.state)
        _, _, done, info = env.step(action)
```

- [ ] **Step 5: Run the new tests + full suite**

Run: `.venv/bin/pytest tests/test_system_monitor.py tests/test_stochastic_fairness.py -v && .venv/bin/pytest -q`
Expected: all pass; full suite green (the M2 `test_fairness` still passes — `run_episode` behaviour at `(0,0)` is unchanged since all nodes stay alive).

- [ ] **Step 6: Lint, format, commit**

```bash
.venv/bin/ruff check . && .venv/bin/black . && git add -A && git commit -m "feat: SystemMonitor failure reporting + scheduler deadlock guard"
```

---

## Self-Review

**Spec coverage (design doc §1–§7 + exit criteria):**
- Failure calendar (isolated RNG, per-node Exp, instance+seed keyed) — Tasks 1, 4. ✅
- Placement-failure rule `actual_finish > t_f ⇒ die+requeue+reward 0` — Task 5. ✅
- Noise reveal-at-completion (nominal planning, actual commit) — Tasks 4, 5. ✅
- RNG isolation ⇒ `(0,0)` golden bit-for-bit — Tasks 1, 4, 5 (golden test in suite). ✅
- Reward 0 on failure; no Δmakespan from death; horizon over all committed nodes — Tasks 3, 5. ✅
- Energy telescoping (Σ committed = total); no partial charge — Task 5. ✅
- `load_balance_index` over alive ids — Task 2. ✅
- SystemMonitor reports realized failures; uniform trigger — Task 6. ✅
- Deadlock terminal + scheduler guard — Tasks 5, 6. ✅
- T_DONE/T_SCHEDULED kept distinct, not deduped (observation columns unchanged; `Assignment` keeps start≠finish) — no code change needed (M1 columns retained; design notes they stay observationally coincident in this model). ✅
- Tests: noise, failures, deadlock, calendar fairness (bit-identical across strategies/envs), load-balance alive-ids, golden regression — Tasks 2,4,5,6 + existing golden. ✅

**Placeholder scan:** no TBD/TODO; complete code in every step. The `_instance_signature` uses `repr()` of floats (round-trip-stable in CPython) + crc32 — deterministic, not a placeholder.

**Type consistency:** `derive_rng(seed:int, salt:str)->Generator` consistent (Tasks 1,4). `ClusterState.failure_times/noise_eps: dict[int,float]` consistent (Tasks 3,4,5). `load_balance_index(alive_node_ids: list[int])` consistent across schedule + both call sites (Tasks 2,5). `step` 4-tuple unchanged; `info` keys (`failed_node`, `deadlocked`, `balance`) documented. `SystemMonitor.check(state)->list[int]` (Task 6).

**`(0,0)` regression argument (the load-bearing claim):** at `noise_std=0` every `ε=0` ⇒ `actual_exec == nominal_finish-start == exec_time(task,node)` ⇒ `actual_energy == energy(task,node)` and `actual_finish == nominal_finish` (bit-identical to M1's commit); at `failure_rate=0` every `t_f=+inf` ⇒ `actual_finish > inf` is always False ⇒ success path only; `horizon` over all nodes == over alive nodes (all alive); `load_balance_index([0,1])` == old `load_balance_index(2)`. The noise/failure draws come from `derive_rng` streams independent of the DAG/cluster stream, so they never shift the deterministic sequence. Therefore `tests/test_golden_schedule.py` and all M1–M3b tests remain green unchanged.

**Deferred (M5):** per-episode noise-seed override for the "5–10 noise seeds per DAG" eval grid (M4 keys by instance+config.seed, which suffices for M4's fairness tests and gives training diversity across instances); the regime grid / robustness metrics / Wilcoxon / CLI / WfCommons.
