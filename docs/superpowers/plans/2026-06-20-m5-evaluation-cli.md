# M5 — Evaluation + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the metrics, the regime-grid evaluation runner with paired significance testing, and the `drl_scheduler.py` train/eval CLI — turning the trained agent + baselines into reproducible thesis evidence.

**Architecture:** A new `src/eval/` layer above `scheduler`/`rl`. Pure `metrics.py` and `significance.py` are unit-tested in isolation; `evaluate.py` orchestrates the grid by running every strategy through the existing `run_episode` (fairness is structural — same env seed + same instance ⇒ identical noise/failure calendar). The CLI is a thin argparse shell driven entirely by `config.yaml`.

**Tech Stack:** Python 3.10+, pandas (CSV/aggregation), scipy (Wilcoxon), NumPy, PyTorch (checkpoint load), pytest.

## Global Constraints

- Python >= 3.10; full type hints on all public functions.
- Run all tools via the `.venv/bin/` prefix (bare `python`/`pytest`/`ruff`/`black` are NOT on PATH; PEP 668). Working dir: `/home/caverraaa/workspace/github.com/caverraaa/rl-scheduler`.
- ruff + black clean (line-length 100); ruff lint select `E,F,I,UP,B`.
- TDD: write the failing test first, run it to confirm it fails, then implement.
- No magic numbers in code — all experiment settings live in `config.yaml` (the new `eval:` block).
- Respect layering: `eval` may import `core/env/dag_factory/strategies/scheduler/rl/utils`; nothing imports `eval`. No back-dependencies.
- DAGs only via `DAGFactory` (synthetic + `load_from_wfcommons`).
- **Fairness invariant (thesis-critical):** every strategy runs through the identical `run_episode` + `env.reset` on the same `(dag, nodes)` and same env `seed` per `(regime, instance, noise_seed)`. No per-strategy special-casing. Guarded by `test_evaluate`.
- The eval runner loads **committed** `dag_benchmarks/*.json`; it must NOT `import wfcommons`.
- `Config` is a frozen dataclass — build per-regime envs with `dataclasses.replace(base_cfg, seed=…, noise_std=…, failure_rate=…)`.

## Key existing signatures (consume these; do not change them)

- `run_episode(env, strategy, dag=None, nodes=None, monitor=None) -> tuple[Schedule, dict]`. `info` has keys `makespan, energy, balance, m_ref, e_ref`.
- `Schedule.makespan() -> float`, `Schedule.busy_time_by_node() -> dict[int,float]`, `Schedule.load_balance_index(alive_ids: list[int]) -> float`, `Schedule.total_energy: float`.
- `cost_model.exec_time(task, node) -> float`.
- `ClusterEnv(config: Config)`; `env.reset(dag=, nodes=) -> (obs, info)`; `env.state.nodes` (list of `ComputeNode` with `.node_id`, `.alive`); `env.step(action) -> (obs, reward, done, info)`.
- `make_cluster(rng, n_nodes, beta) -> list[ComputeNode]`.
- `DAGFactory.create("synthetic", rng, n_tasks=, n_layers=, edge_prob=, ccr=)`; `DAGFactory.load_from_wfcommons(path, rng, recipe=None)`.
- Strategies: `HEFTStrategy()`, `CPOPStrategy()`, `MinMinStrategy()`, `WeightedSumGreedyStrategy(w1, w2)`, `RandomStrategy(rng)`, `RLStrategy(policy)`.
- `TwoHeadPolicy(GNNEncoder(hidden=, layers=), hidden=)`; `policy.load_state_dict(torch.load(path, weights_only=True))`.
- `PPOTrainer(policy, config)`; `.train(env, n_updates, dag=None, nodes=None) -> list[dict]`; `.save_checkpoint(path)`.
- `make_rng(seed) -> Generator`; `derive_rng(seed, salt) -> Generator`.
- `load_config(path="config.yaml") -> Config` (frozen; fields incl. `seed, n_layers, edge_prob, ccr, w1, w2, gnn_hidden, gnn_layers, total_updates, noise_std, failure_rate`).

## File Structure

| File | Responsibility |
|------|----------------|
| `src/eval/__init__.py` (create) | Package marker. |
| `src/eval/eval_config.py` (create) | `EvalConfig` dataclass + `load_eval_config` (reads `eval:` block). |
| `config.yaml` (modify) | Add the `eval:` block. |
| `requirements.txt` (modify) | Add `scipy>=1.11`. |
| `src/eval/metrics.py` (create) | Pure metric fns + `compute_run_metrics` + `TimingStrategy`. |
| `src/eval/significance.py` (create) | `paired_wilcoxon` (scipy wrapper + guard). |
| `src/eval/evaluate.py` (create) | Instance/strategy building, `run_grid` (raw rows), then `summarize` + `compare_significance` + `write_results` + `print_tables`. |
| `drl_scheduler.py` (create) | CLI: `train` / `eval` subcommands. |
| tests: `test_eval_config.py`, `test_metrics.py`, `test_significance.py`, `test_evaluate.py`, `test_cli.py` (create) | Per-task tests. |

---

## Task 1: EvalConfig + config block + scipy dep

**Files:**
- Create: `src/eval/__init__.py`, `src/eval/eval_config.py`
- Modify: `config.yaml`, `requirements.txt`
- Test: `tests/test_eval_config.py`

**Interfaces:**
- Produces: `EvalConfig` frozen dataclass with fields
  `noise_std: list[float]`, `beta: list[float]`, `failure_rate: float`, `failures: list[bool]`,
  `n_dags: int`, `dag_sizes: list[int]`, `n_nodes: int`, `noise_seeds: list[int]`,
  `dag_seed_base: int`, `benchmark_dir: str`, `checkpoint_glob: str`, `results_dir: str`.
  `load_eval_config(path: str = "config.yaml") -> EvalConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_config.py
from src.eval.eval_config import EvalConfig, load_eval_config


def test_load_eval_config_from_config() -> None:
    e = load_eval_config("config.yaml")
    assert isinstance(e, EvalConfig)
    assert len(e.noise_std) >= 1 and len(e.beta) >= 1
    assert e.failures == [False, True]
    assert e.n_dags >= 1 and e.n_nodes >= 1
    assert len(e.noise_seeds) >= 1
    assert e.dag_seed_base >= 1
    assert e.benchmark_dir and e.checkpoint_glob and e.results_dir


def test_eval_config_is_frozen() -> None:
    import dataclasses

    import pytest

    e = load_eval_config("config.yaml")
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.n_dags = 99  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_eval_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval'`

- [ ] **Step 3: Add the package marker, config block, and scipy dep**

Create `src/eval/__init__.py`:

```python
"""Evaluation layer: metrics, regime-grid runner, significance (TZ §7, §10)."""
```

Append to `config.yaml`:

```yaml

# M5 evaluation grid. Smoke-sized defaults; full thesis grid values in comments.
eval:
  noise_std: [0.0, 0.1, 0.2]
  beta: [2.0, 5.0, 10.0]
  failure_rate: 0.0          # the "on" value when failures enabled; 0.0 => failures-on is a no-op
  failures: [false, true]
  n_dags: 5                  # full: 30-50
  dag_sizes: [20, 40, 60]    # sampled (round-robin) for held-out synthetic DAGs
  n_nodes: 8
  noise_seeds: [0, 1, 2]     # full: 5-10 distinct seeds
  dag_seed_base: 100000      # held-out stream offset (disjoint from training seeds)
  benchmark_dir: dag_benchmarks
  checkpoint_glob: models/rl_seed*.pth
  results_dir: results
```

Append to `requirements.txt`:

```
scipy>=1.11
```

- [ ] **Step 4: Write the implementation**

```python
# src/eval/eval_config.py
"""Typed config for the M5 evaluation grid, loaded from config.yaml `eval:` block."""

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class EvalConfig:
    noise_std: list[float]
    beta: list[float]
    failure_rate: float
    failures: list[bool]
    n_dags: int
    dag_sizes: list[int]
    n_nodes: int
    noise_seeds: list[int]
    dag_seed_base: int
    benchmark_dir: str
    checkpoint_glob: str
    results_dir: str


def load_eval_config(path: str = "config.yaml") -> EvalConfig:
    """Parse the ``eval:`` block of config.yaml into a typed, frozen EvalConfig."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    e = raw["eval"]
    return EvalConfig(
        noise_std=[float(x) for x in e["noise_std"]],
        beta=[float(x) for x in e["beta"]],
        failure_rate=float(e["failure_rate"]),
        failures=[bool(x) for x in e["failures"]],
        n_dags=int(e["n_dags"]),
        dag_sizes=[int(x) for x in e["dag_sizes"]],
        n_nodes=int(e["n_nodes"]),
        noise_seeds=[int(x) for x in e["noise_seeds"]],
        dag_seed_base=int(e["dag_seed_base"]),
        benchmark_dir=str(e["benchmark_dir"]),
        checkpoint_glob=str(e["checkpoint_glob"]),
        results_dir=str(e["results_dir"]),
    )
```

- [ ] **Step 5: Run tests + verify scipy importable + lint**

Run: `.venv/bin/pip install 'scipy>=1.11' >/dev/null 2>&1; .venv/bin/python -c "import scipy.stats; print('scipy ok')" && .venv/bin/pytest tests/test_eval_config.py -v && .venv/bin/ruff check src/eval tests/test_eval_config.py && .venv/bin/black --check src/eval tests/test_eval_config.py`
Expected: `scipy ok`, tests PASS, ruff clean, black clean. (scipy is already present transitively from wfcommons; the install is a no-op safety net.)

- [ ] **Step 6: Commit**

```bash
git add src/eval/__init__.py src/eval/eval_config.py config.yaml requirements.txt tests/test_eval_config.py
git commit -m "feat: EvalConfig loader + eval config block + scipy dependency"
```

---

## Task 2: Metrics + TimingStrategy

**Files:**
- Create: `src/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Schedule`, `TaskDAG`, `ComputeNode`, `cost_model.exec_time`, `BaseSchedulingStrategy`, `ClusterState`.
- Produces:
  - `utilisation(schedule, c_max, alive_ids) -> float`
  - `slr(c_max, m_ref) -> float`
  - `speedup(dag, nodes, c_max) -> float`
  - `compute_run_metrics(schedule, info, dag, nodes, alive_ids, predict_seconds) -> dict[str, float]`
    keys: `makespan, energy, utilisation, load_balance, slr, speedup, overhead_ms`.
  - `class TimingStrategy(BaseSchedulingStrategy)` wrapping an inner strategy, exposing
    `predict_seconds: float`.

**Hand-checked golden values** (the existing diamond golden instance, all tasks on GPU node 1):
`m_ref=4.0`, `makespan=6.0`, `energy=1200.0`, `balance=0.0`. Derived:
`slr = 6/4 = 1.5`; serial-on-fastest = sum(base_cost)/GPU_speed = (2+4+4+2)/2 = 6 → `speedup = 6/6 = 1.0`;
busy on GPU = exec(1+2+2+1)=6, alive nodes=2 → `utilisation = 6/(6×2) = 0.5`; `load_balance = 0.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import time

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.env.cluster_env import ClusterEnv
from src.env.placement import ClusterState
from src.eval.metrics import TimingStrategy, compute_run_metrics, slr, speedup, utilisation
from src.strategies.base import BaseSchedulingStrategy
from src.utils.config import load_config


def _run_golden(golden_instance: tuple[TaskDAG, list[ComputeNode]]):
    dag, nodes = golden_instance
    env = ClusterEnv(load_config("config.yaml"))
    _, info = env.reset(dag=dag, nodes=nodes)
    for action in [(0, 1), (1, 1), (2, 1), (3, 1)]:
        _, _, done, info = env.step(action)
    alive_ids = [n.node_id for n in env.state.nodes if n.alive]
    return env.schedule, info, dag, nodes, alive_ids


def test_pure_metric_helpers() -> None:
    assert slr(6.0, 4.0) == 1.5
    assert slr(6.0, 0.0) == 0.0  # guard


def test_compute_run_metrics_golden(golden_instance) -> None:
    schedule, info, dag, nodes, alive_ids = _run_golden(golden_instance)
    m = compute_run_metrics(schedule, info, dag, nodes, alive_ids, predict_seconds=0.0)
    assert m["makespan"] == 6.0
    assert m["energy"] == 1200.0
    assert m["slr"] == 1.5
    assert m["speedup"] == 1.0
    assert m["utilisation"] == 0.5
    assert m["load_balance"] == 0.0
    assert m["overhead_ms"] == 0.0


def test_speedup_and_utilisation_helpers(golden_instance) -> None:
    schedule, info, dag, nodes, alive_ids = _run_golden(golden_instance)
    assert speedup(dag, nodes, 6.0) == 1.0
    assert utilisation(schedule, 6.0, alive_ids) == 0.5


def test_timing_strategy_accumulates_and_delegates() -> None:
    class _Slow(BaseSchedulingStrategy):
        def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
            time.sleep(0.001)
            return (ready[0], 0)

    ts = TimingStrategy(_Slow())
    action = ts.predict([3, 5], state=None)  # type: ignore[arg-type]
    assert action == (3, 0)
    assert ts.predict_seconds > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.metrics'`

- [ ] **Step 3: Write the implementation**

```python
# src/eval/metrics.py
"""Per-run evaluation metrics (TZ §10) + a predict-timing strategy wrapper.

All functions are pure (read a finished schedule + cached refs; no re-run).
"""

import time

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.core.schedule import Schedule
from src.env.cost_model import exec_time
from src.env.placement import ClusterState
from src.strategies.base import BaseSchedulingStrategy


def utilisation(schedule: Schedule, c_max: float, alive_ids: list[int]) -> float:
    """Total busy time / (makespan x number of alive nodes)."""
    if c_max <= 0.0 or not alive_ids:
        return 0.0
    total_busy = sum(schedule.busy_time_by_node().values())
    return total_busy / (c_max * len(alive_ids))


def slr(c_max: float, m_ref: float) -> float:
    """Schedule Length Ratio: makespan / fastest-exec critical-path lower bound."""
    return c_max / m_ref if m_ref > 0.0 else 0.0


def speedup(dag: TaskDAG, nodes: list[ComputeNode], c_max: float) -> float:
    """Serial time on the single fastest node / parallel makespan (Topcuoglu)."""
    if c_max <= 0.0 or not nodes:
        return 0.0
    serial = min(
        sum(exec_time(dag.task(i), node) for i in range(dag.n_tasks)) for node in nodes
    )
    return serial / c_max


def compute_run_metrics(
    schedule: Schedule,
    info: dict,
    dag: TaskDAG,
    nodes: list[ComputeNode],
    alive_ids: list[int],
    predict_seconds: float,
) -> dict[str, float]:
    """One row of metrics for a finished episode."""
    c_max = schedule.makespan()
    return {
        "makespan": c_max,
        "energy": schedule.total_energy,
        "utilisation": utilisation(schedule, c_max, alive_ids),
        "load_balance": schedule.load_balance_index(alive_ids),
        "slr": slr(c_max, float(info["m_ref"])),
        "speedup": speedup(dag, nodes, c_max),
        "overhead_ms": predict_seconds * 1000.0,
    }


class TimingStrategy(BaseSchedulingStrategy):
    """Wrap a strategy, delegating predict while accumulating wall-clock time."""

    def __init__(self, inner: BaseSchedulingStrategy) -> None:
        self._inner = inner
        self.predict_seconds = 0.0

    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        t0 = time.perf_counter()
        action = self._inner.predict(ready, state)
        self.predict_seconds += time.perf_counter() - t0
        return action
```

- [ ] **Step 4: Run tests + lint**

Run: `.venv/bin/pytest tests/test_metrics.py -v && .venv/bin/ruff check src/eval/metrics.py tests/test_metrics.py && .venv/bin/black --check src/eval/metrics.py tests/test_metrics.py`
Expected: PASS (4 tests), ruff clean, black clean.

- [ ] **Step 5: Commit**

```bash
git add src/eval/metrics.py tests/test_metrics.py
git commit -m "feat: eval metrics (slr/speedup/utilisation/overhead) + TimingStrategy"
```

---

## Task 3: Paired Wilcoxon significance

**Files:**
- Create: `src/eval/significance.py`
- Test: `tests/test_significance.py`

**Interfaces:**
- Produces: `paired_wilcoxon(a: list[float], b: list[float]) -> tuple[float, float]` returning
  `(statistic, p_value)`. Guards the all-equal / zero-difference case by returning `(0.0, 1.0)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_significance.py
import pytest

from src.eval.significance import paired_wilcoxon


def test_all_equal_returns_p_one() -> None:
    stat, p = paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert stat == 0.0
    assert p == 1.0


def test_clear_difference_is_significant() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b = [x + 5.0 for x in a]  # b strictly larger everywhere
    stat, p = paired_wilcoxon(a, b)
    assert p < 0.05


def test_matches_scipy_reference() -> None:
    from scipy.stats import wilcoxon

    a = [5.0, 3.0, 8.0, 2.0, 7.0, 6.0, 9.0]
    b = [4.0, 4.0, 6.0, 3.0, 5.0, 7.0, 6.0]
    stat, p = paired_wilcoxon(a, b)
    ref_stat, ref_p = wilcoxon(a, b)
    assert stat == pytest.approx(float(ref_stat))
    assert p == pytest.approx(float(ref_p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_significance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.significance'`

- [ ] **Step 3: Write the implementation**

```python
# src/eval/significance.py
"""Paired significance test for RL-vs-baseline comparisons (TZ §10)."""

from scipy.stats import wilcoxon


def paired_wilcoxon(a: list[float], b: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank over paired samples; (statistic, p_value).

    Returns (0.0, 1.0) when every paired difference is zero (scipy raises in
    that degenerate case), so callers never need to special-case identical runs.
    """
    if len(a) != len(b):
        raise ValueError("paired_wilcoxon requires equal-length samples.")
    if all(x == y for x, y in zip(a, b)):
        return (0.0, 1.0)
    stat, p = wilcoxon(a, b)
    return (float(stat), float(p))
```

- [ ] **Step 4: Run tests + lint**

Run: `.venv/bin/pytest tests/test_significance.py -v && .venv/bin/ruff check src/eval/significance.py tests/test_significance.py && .venv/bin/black --check src/eval/significance.py tests/test_significance.py`
Expected: PASS (3 tests), ruff clean, black clean.

- [ ] **Step 5: Commit**

```bash
git add src/eval/significance.py tests/test_significance.py
git commit -m "feat: paired Wilcoxon significance wrapper with all-equal guard"
```

---

## Task 4: Grid runner — instances, strategies, run_grid

**Files:**
- Create: `src/eval/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: Task 1 `EvalConfig`/`load_eval_config`; Task 2 `compute_run_metrics`/`TimingStrategy`;
  `run_episode`, `ClusterEnv`, `make_cluster`, `DAGFactory`, strategies, `RLStrategy`,
  `TwoHeadPolicy`, `GNNEncoder`, `make_rng`, `derive_rng`, `load_config`.
- Produces:
  - `build_dags(eval_cfg, base_cfg) -> list[tuple[str, TaskDAG, int]]` — `(label, dag, seed)`;
    synthetic held-out DAGs + every `dag_benchmarks/*.json`.
  - `load_checkpoints(eval_cfg, base_cfg) -> list[tuple[str, TwoHeadPolicy]]`.
  - `build_strategies(base_cfg, checkpoints) -> list[tuple[str, BaseSchedulingStrategy]]`.
  - `run_grid(eval_cfg, base_cfg, checkpoints) -> pandas.DataFrame` — one row per
    `(regime, dag_label, noise_seed, strategy)` with metric columns + identifier columns
    `noise_std, beta, failures, dag_label, noise_seed, strategy`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
from src.eval.eval_config import EvalConfig
from src.eval.evaluate import build_dags, build_strategies, run_grid
from src.utils.config import load_config


def _eval_cfg() -> EvalConfig:
    return EvalConfig(
        noise_std=[0.0],
        beta=[2.0],
        failure_rate=0.0,
        failures=[False],
        n_dags=2,
        dag_sizes=[20],
        n_nodes=4,
        noise_seeds=[0, 1],
        dag_seed_base=100000,
        benchmark_dir="dag_benchmarks",
        checkpoint_glob="models/__none__*.pth",  # no checkpoints in this smoke test
        results_dir="results",
    )


def test_build_dags_includes_synthetic_and_benchmarks() -> None:
    e = _eval_cfg()
    dags = build_dags(e, load_config("config.yaml"))
    labels = [lbl for lbl, _dag, _seed in dags]
    assert sum(1 for lbl in labels if lbl.startswith("synthetic")) == 2
    assert any(lbl.startswith("bench:") for lbl in labels)  # committed dag_benchmarks/*.json


def test_run_grid_fairness_identical_instance_sets() -> None:
    e = _eval_cfg()
    base = load_config("config.yaml")
    strategies = build_strategies(base, checkpoints=[])
    df = run_grid(e, base, checkpoints=[])
    # Every strategy must have been run on the identical (dag_label, noise_seed) set.
    keysets = {
        name: set(map(tuple, df[df["strategy"] == name][["dag_label", "noise_seed"]].values))
        for name, _ in strategies
    }
    reference = next(iter(keysets.values()))
    assert all(ks == reference for ks in keysets.values())
    assert len(reference) >= 1


def test_run_grid_has_metric_columns() -> None:
    e = _eval_cfg()
    df = run_grid(e, load_config("config.yaml"), checkpoints=[])
    for col in ["makespan", "energy", "utilisation", "load_balance", "slr", "speedup",
                "overhead_ms", "noise_std", "beta", "failures", "dag_label", "noise_seed",
                "strategy"]:
        assert col in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.evaluate'`

- [ ] **Step 3: Write the implementation**

```python
# src/eval/evaluate.py
"""Regime-grid evaluation runner (TZ §7, §10). Fairness is structural via run_episode."""

import dataclasses
import glob
import os

import pandas as pd
import torch

from src.dag_factory.factory import DAGFactory
from src.env.cluster_env import ClusterEnv
from src.env.cluster_factory import make_cluster
from src.eval.eval_config import EvalConfig
from src.eval.metrics import TimingStrategy, compute_run_metrics
from src.rl.gnn_encoder import GNNEncoder
from src.rl.policy import TwoHeadPolicy
from src.rl.rl_strategy import RLStrategy
from src.scheduler.task_scheduler import run_episode
from src.strategies.base import BaseSchedulingStrategy
from src.strategies.cpop import CPOPStrategy
from src.strategies.heft import HEFTStrategy
from src.strategies.min_min import MinMinStrategy
from src.strategies.random_strategy import RandomStrategy
from src.strategies.weighted_sum_greedy import WeightedSumGreedyStrategy
from src.utils.config import Config
from src.utils.seeding import derive_rng, make_rng


def build_dags(eval_cfg: EvalConfig, base_cfg: Config) -> list[tuple[str, object, int]]:
    """Held-out synthetic DAGs (pinned seeds) + every committed benchmark JSON."""
    dags: list[tuple[str, object, int]] = []
    for k in range(eval_cfg.n_dags):
        seed = eval_cfg.dag_seed_base + k
        n_tasks = eval_cfg.dag_sizes[k % len(eval_cfg.dag_sizes)]
        dag = DAGFactory.create(
            "synthetic",
            make_rng(seed),
            n_tasks=n_tasks,
            n_layers=base_cfg.n_layers,
            edge_prob=base_cfg.edge_prob,
            ccr=base_cfg.ccr,
        )
        dags.append((f"synthetic:{n_tasks}:{seed}", dag, seed))
    for path in sorted(glob.glob(os.path.join(eval_cfg.benchmark_dir, "*.json"))):
        name = os.path.basename(path).replace(".json", "")
        recipe = name.split("_")[0]
        # rng is unused by the deterministic parser; seed it from dag_seed_base for clarity.
        dag = DAGFactory.load_from_wfcommons(path, make_rng(eval_cfg.dag_seed_base), recipe=recipe)
        dags.append((f"bench:{name}", dag, eval_cfg.dag_seed_base))
    return dags


def load_checkpoints(
    eval_cfg: EvalConfig, base_cfg: Config
) -> list[tuple[str, TwoHeadPolicy]]:
    """Load every checkpoint matching the glob into an eval-mode policy."""
    out: list[tuple[str, TwoHeadPolicy]] = []
    for path in sorted(glob.glob(eval_cfg.checkpoint_glob)):
        policy = TwoHeadPolicy(
            GNNEncoder(hidden=base_cfg.gnn_hidden, layers=base_cfg.gnn_layers),
            hidden=base_cfg.gnn_hidden,
        )
        policy.load_state_dict(torch.load(path, weights_only=True))
        policy.eval()
        out.append((os.path.basename(path).replace(".pth", ""), policy))
    return out


def build_strategies(
    base_cfg: Config, checkpoints: list[tuple[str, TwoHeadPolicy]]
) -> list[tuple[str, BaseSchedulingStrategy]]:
    """The fixed baseline set + one RLStrategy per loaded checkpoint."""
    strategies: list[tuple[str, BaseSchedulingStrategy]] = [
        ("heft", HEFTStrategy()),
        ("cpop", CPOPStrategy()),
        ("min_min", MinMinStrategy()),
        ("wsg", WeightedSumGreedyStrategy(base_cfg.w1, base_cfg.w2)),
        ("random", RandomStrategy(make_rng(base_cfg.seed))),
    ]
    for label, policy in checkpoints:
        strategies.append((f"rl@{label}", RLStrategy(policy)))
    return strategies


def run_grid(
    eval_cfg: EvalConfig,
    base_cfg: Config,
    checkpoints: list[tuple[str, TwoHeadPolicy]],
) -> pd.DataFrame:
    """Run every strategy on every (regime, instance, noise_seed); return raw rows."""
    dags = build_dags(eval_cfg, base_cfg)
    strategies = build_strategies(base_cfg, checkpoints)
    rows: list[dict] = []
    for noise_std in eval_cfg.noise_std:
        for beta in eval_cfg.beta:
            for failures in eval_cfg.failures:
                failure_rate = eval_cfg.failure_rate if failures else 0.0
                for dag_label, dag, dag_seed in dags:
                    # One cluster per (instance, beta); reused across noise seeds.
                    nodes = make_cluster(
                        derive_rng(dag_seed, f"cluster-beta{beta}"), eval_cfg.n_nodes, beta
                    )
                    for noise_seed in eval_cfg.noise_seeds:
                        cfg = dataclasses.replace(
                            base_cfg,
                            seed=noise_seed,
                            noise_std=noise_std,
                            failure_rate=failure_rate,
                        )
                        env = ClusterEnv(cfg)
                        for name, strategy in strategies:
                            timed = TimingStrategy(strategy)
                            schedule, info = run_episode(env, timed, dag=dag, nodes=nodes)
                            alive_ids = [n.node_id for n in env.state.nodes if n.alive]
                            metrics = compute_run_metrics(
                                schedule, info, dag, nodes, alive_ids, timed.predict_seconds
                            )
                            rows.append(
                                {
                                    **metrics,
                                    "noise_std": noise_std,
                                    "beta": beta,
                                    "failures": failures,
                                    "dag_label": dag_label,
                                    "noise_seed": noise_seed,
                                    "strategy": name,
                                }
                            )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests + lint**

Run: `.venv/bin/pytest tests/test_evaluate.py -v && .venv/bin/ruff check src/eval/evaluate.py tests/test_evaluate.py && .venv/bin/black --check src/eval/evaluate.py tests/test_evaluate.py`
Expected: PASS (3 tests), ruff clean, black clean.

- [ ] **Step 5: Commit**

```bash
git add src/eval/evaluate.py tests/test_evaluate.py
git commit -m "feat: regime-grid runner (instances, strategies, fair run_grid)"
```

---

## Task 5: Aggregation, significance roll-up, and output

**Files:**
- Modify: `src/eval/evaluate.py`
- Test: `tests/test_evaluate.py` (add tests)

**Interfaces:**
- Consumes: Task 4 `run_grid` output DataFrame; Task 3 `paired_wilcoxon`.
- Produces (added to `evaluate.py`):
  - `summarize(df) -> pandas.DataFrame` — mean+std per `(noise_std, beta, failures, strategy)`
    for each metric, plus a `robustness` column (mean over instances of the per-instance
    makespan std across noise seeds).
  - `compare_significance(df) -> pandas.DataFrame` — for each regime, each `rl@*` strategy vs
    each non-RL baseline, paired on `(dag_label, noise_seed)` over common instances; columns
    `noise_std, beta, failures, rl_strategy, baseline, n_pairs, wilcoxon_stat, p_value`.
  - `write_results(df, summary, significance, results_dir) -> None` — writes
    `eval_runs.csv`, `eval_summary.csv`, `eval_significance.csv`.
  - `print_tables(summary, significance) -> None` — one comparison table per regime + p-values.

- [ ] **Step 1: Write the failing test (append to tests/test_evaluate.py)**

```python
def test_summarize_and_significance_and_write(tmp_path) -> None:
    from src.eval.evaluate import compare_significance, run_grid, summarize, write_results

    base = load_config("config.yaml")
    # Build a tiny grid with ONE fake checkpoint so the RL-vs-baseline path runs.
    from src.rl.gnn_encoder import GNNEncoder
    from src.rl.policy import TwoHeadPolicy

    policy = TwoHeadPolicy(GNNEncoder(hidden=base.gnn_hidden, layers=base.gnn_layers),
                           hidden=base.gnn_hidden)
    df = run_grid(_eval_cfg(), base, checkpoints=[("rl_seed0", policy)])

    summary = summarize(df)
    assert {"makespan_mean", "makespan_std", "robustness"}.issubset(summary.columns)

    sig = compare_significance(df)
    assert {"rl_strategy", "baseline", "p_value", "n_pairs"}.issubset(sig.columns)
    assert len(sig) >= 1  # at least one rl-vs-baseline comparison

    write_results(df, summary, sig, str(tmp_path))
    import os
    for fname in ["eval_runs.csv", "eval_summary.csv", "eval_significance.csv"]:
        assert os.path.exists(os.path.join(str(tmp_path), fname))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_evaluate.py::test_summarize_and_significance_and_write -v`
Expected: FAIL with `ImportError: cannot import name 'summarize'`

- [ ] **Step 3: Write the implementation (append to src/eval/evaluate.py)**

```python
_METRIC_COLS = [
    "makespan", "energy", "utilisation", "load_balance", "slr", "speedup", "overhead_ms"
]
_REGIME_COLS = ["noise_std", "beta", "failures"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean+std per (regime, strategy) for each metric, plus a robustness column."""
    grouped = df.groupby([*_REGIME_COLS, "strategy"])
    summary = grouped[_METRIC_COLS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    # robustness = mean over instances of (makespan std across noise seeds).
    per_instance = (
        df.groupby([*_REGIME_COLS, "strategy", "dag_label"])["makespan"].std().reset_index()
    )
    robustness = (
        per_instance.groupby([*_REGIME_COLS, "strategy"])["makespan"]
        .mean()
        .reset_index()
        .rename(columns={"makespan": "robustness"})
    )
    return summary.merge(robustness, on=[*_REGIME_COLS, "strategy"], how="left")


def compare_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Per regime, each rl@* vs each non-RL baseline, paired on (dag_label, noise_seed)."""
    from src.eval.significance import paired_wilcoxon

    rl_names = sorted({s for s in df["strategy"].unique() if s.startswith("rl@")})
    baseline_names = sorted({s for s in df["strategy"].unique() if not s.startswith("rl@")})
    rows: list[dict] = []
    for regime, sub in df.groupby(_REGIME_COLS):
        for rl in rl_names:
            rl_df = sub[sub["strategy"] == rl].set_index(["dag_label", "noise_seed"])
            for base_name in baseline_names:
                b_df = sub[sub["strategy"] == base_name].set_index(["dag_label", "noise_seed"])
                common = rl_df.index.intersection(b_df.index)
                a = [float(rl_df.loc[k, "makespan"]) for k in common]
                b = [float(b_df.loc[k, "makespan"]) for k in common]
                stat, p = paired_wilcoxon(a, b)
                rows.append(
                    {
                        "noise_std": regime[0],
                        "beta": regime[1],
                        "failures": regime[2],
                        "rl_strategy": rl,
                        "baseline": base_name,
                        "n_pairs": len(common),
                        "wilcoxon_stat": stat,
                        "p_value": p,
                    }
                )
    return pd.DataFrame(rows)


def write_results(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    results_dir: str,
) -> None:
    """Write raw rows, summary, and significance CSVs to results_dir."""
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, "eval_runs.csv"), index=False)
    summary.to_csv(os.path.join(results_dir, "eval_summary.csv"), index=False)
    significance.to_csv(os.path.join(results_dir, "eval_significance.csv"), index=False)


def print_tables(summary: pd.DataFrame, significance: pd.DataFrame) -> None:
    """Print one comparison table per regime followed by RL-vs-baseline p-values."""
    for regime, sub in summary.groupby(_REGIME_COLS):
        print(f"\n=== regime noise_std={regime[0]} beta={regime[1]} failures={regime[2]} ===")
        cols = ["strategy", "makespan_mean", "makespan_std", "energy_mean",
                "load_balance_mean", "slr_mean", "speedup_mean", "overhead_ms_mean",
                "robustness"]
        print(sub[cols].to_string(index=False))
    if not significance.empty:
        print("\n=== RL vs baselines (Wilcoxon paired on makespan) ===")
        print(significance.to_string(index=False))
```

- [ ] **Step 4: Run tests + lint**

Run: `.venv/bin/pytest tests/test_evaluate.py -v && .venv/bin/ruff check src/eval/evaluate.py tests/test_evaluate.py && .venv/bin/black --check src/eval/evaluate.py tests/test_evaluate.py`
Expected: PASS (all evaluate tests), ruff clean, black clean.

- [ ] **Step 5: Commit**

```bash
git add src/eval/evaluate.py tests/test_evaluate.py
git commit -m "feat: eval aggregation, significance roll-up, CSV + console output"
```

---

## Task 6: CLI (`drl_scheduler.py`)

**Files:**
- Create: `drl_scheduler.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_config`, `load_eval_config`, `PPOTrainer`, `TwoHeadPolicy`, `GNNEncoder`,
  `ClusterEnv`, Task 4/5 `load_checkpoints`/`run_grid`/`summarize`/`compare_significance`/
  `write_results`/`print_tables`.
- Produces:
  - `cmd_train(seed: int, config_path: str = "config.yaml") -> str` — trains one model,
    saves `models/rl_seed{seed}.pth`, writes `results/train_history_seed{seed}.csv`,
    returns the checkpoint path.
  - `cmd_eval(config_path: str = "config.yaml") -> None` — runs the grid, writes CSVs,
    prints tables.
  - `main(argv: list[str] | None = None) -> None` — argparse dispatch for `train` / `eval`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import os

import pandas as pd


def _fast_config(tmp_path) -> str:
    """Write a tiny config.yaml copy with a 2-update training budget for speed."""
    import yaml

    with open("config.yaml", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["total_updates"] = 2
    raw["rollout_episodes"] = 1
    raw["gnn_hidden"] = 16
    raw["eval"] = {
        "noise_std": [0.0], "beta": [2.0], "failure_rate": 0.0, "failures": [False],
        "n_dags": 1, "dag_sizes": [20], "n_nodes": 4, "noise_seeds": [0],
        "dag_seed_base": 100000, "benchmark_dir": "dag_benchmarks",
        "checkpoint_glob": os.path.join(str(tmp_path), "models", "rl_seed*.pth"),
        "results_dir": os.path.join(str(tmp_path), "results"),
    }
    path = os.path.join(str(tmp_path), "config.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh)
    return path


def test_cmd_train_writes_checkpoint_and_history(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = _fast_config(tmp_path)
    from drl_scheduler import cmd_train

    ckpt = cmd_train(seed=0, config_path=cfg_path)
    assert os.path.exists(ckpt)
    assert os.path.exists(os.path.join(str(tmp_path), "results", "train_history_seed0.csv"))


def test_cmd_eval_writes_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # benchmark dir must exist for build_dags glob (empty is fine here).
    os.makedirs(os.path.join(str(tmp_path), "dag_benchmarks"), exist_ok=True)
    cfg_path = _fast_config(tmp_path)
    from drl_scheduler import cmd_eval, cmd_train

    cmd_train(seed=0, config_path=cfg_path)  # produce one checkpoint for the glob
    cmd_eval(config_path=cfg_path)
    summary = os.path.join(str(tmp_path), "results", "eval_summary.csv")
    assert os.path.exists(summary)
    assert len(pd.read_csv(summary)) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'drl_scheduler'`

- [ ] **Step 3: Write the implementation**

```python
# drl_scheduler.py
"""SmartDAG Scheduler CLI: train one RL model or run the evaluation grid (TZ §7, §12)."""

import argparse
import dataclasses
import os

import pandas as pd

from src.env.cluster_env import ClusterEnv
from src.eval.eval_config import load_eval_config
from src.eval.evaluate import (
    compare_significance,
    load_checkpoints,
    print_tables,
    run_grid,
    summarize,
    write_results,
)
from src.rl.gnn_encoder import GNNEncoder
from src.rl.policy import TwoHeadPolicy
from src.rl.ppo_trainer import PPOTrainer
from src.utils.config import load_config


def cmd_train(seed: int, config_path: str = "config.yaml") -> str:
    """Train one model on sampled instances; save checkpoint + history CSV."""
    base = load_config(config_path)
    cfg = dataclasses.replace(base, seed=seed)
    policy = TwoHeadPolicy(
        GNNEncoder(hidden=cfg.gnn_hidden, layers=cfg.gnn_layers), hidden=cfg.gnn_hidden
    )
    trainer = PPOTrainer(policy, cfg)
    env = ClusterEnv(cfg)
    history = trainer.train(env, n_updates=cfg.total_updates)  # dag=None => sampled instances
    ckpt = os.path.join("models", f"rl_seed{seed}.pth")
    trainer.save_checkpoint(ckpt)
    results_dir = load_eval_config(config_path).results_dir
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(
        os.path.join(results_dir, f"train_history_seed{seed}.csv"), index=False
    )
    return ckpt


def cmd_eval(config_path: str = "config.yaml") -> None:
    """Run the regime grid over loaded checkpoints; write CSVs + print tables."""
    base = load_config(config_path)
    eval_cfg = load_eval_config(config_path)
    checkpoints = load_checkpoints(eval_cfg, base)
    df = run_grid(eval_cfg, base, checkpoints)
    summary = summarize(df)
    significance = compare_significance(df)
    write_results(df, summary, significance, eval_cfg.results_dir)
    print_tables(summary, significance)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SmartDAG Scheduler train/eval CLI.")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train", help="train one RL model")
    p_train.add_argument("--seed", type=int, required=True)
    sub.add_parser("eval", help="run the evaluation grid")
    args = parser.parse_args(argv)
    if args.command == "train":
        path = cmd_train(args.seed, args.config)
        print(f"saved {path}")
    elif args.command == "eval":
        cmd_eval(args.config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + lint**

Run: `.venv/bin/pytest tests/test_cli.py -v && .venv/bin/ruff check drl_scheduler.py tests/test_cli.py && .venv/bin/black --check drl_scheduler.py tests/test_cli.py`
Expected: PASS (2 tests), ruff clean, black clean.

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `.venv/bin/pytest -q`
Expected: all tests pass (new M5 tests + all prior), 1 deselected (wfcommons marker).

- [ ] **Step 6: Commit**

```bash
git add drl_scheduler.py tests/test_cli.py
git commit -m "feat: drl_scheduler CLI (train/eval) driven by config.yaml"
```

---

## Self-Review

**Spec coverage:**
- §2.1 scipy core dep — Task 1 (requirements.txt + import check). ✓
- §2.2 eval consumes checkpoints — Task 4 `load_checkpoints`, Task 6 `cmd_eval`. ✓
- §2.3 config-driven grid + smoke defaults — Task 1 `eval:` block. ✓
- §2.4 no TensorBoard, history→CSV — Task 6 `cmd_train` writes `train_history_seed{N}.csv`. ✓
- §3 module layout — Tasks 1–6. ✓
- §4 metric contract (makespan/energy/utilisation/load_balance/slr/speedup/overhead_ms + TimingStrategy; robustness at aggregation) — Task 2 (per-run) + Task 5 (`robustness`). ✓
- §5 regime grid + fairness (regimes from config, noise seed = env seed, cluster per (instance,β) reused across noise seeds, all strategies via run_episode) — Task 4 `run_grid` + `test_run_grid_fairness`. ✓
- §6 aggregation/significance/output (eval_runs/eval_summary/eval_significance CSVs, mean±std, robustness, Wilcoxon per regime, console tables) — Task 5. ✓
- §7 CLI train/eval — Task 6. ✓
- §8 `eval:` config block + EvalConfig — Task 1. ✓
- §9 tests (golden metrics, significance, smoke-grid fairness, CLI, (0,0) regression) — Tasks 2,3,4,5,6. ✓
- §10 exit criteria — covered by Tasks 4–6 + full-suite step.

**Placeholder scan:** none. The `checkpoint_glob="models/__none__*.pth"` in `test_evaluate` is a deliberate no-match pattern (smoke test with zero checkpoints), not a placeholder.

**Type consistency:** `run_grid(eval_cfg, base_cfg, checkpoints)` signature identical in Tasks 4, 5 (test), 6. `summarize`/`compare_significance`/`write_results`/`print_tables` names consistent between Task 5 definition and Task 6 imports. `paired_wilcoxon(a, b) -> (stat, p)` consistent between Task 3 and Task 5. `compute_run_metrics(schedule, info, dag, nodes, alive_ids, predict_seconds)` consistent between Task 2 and Task 4. `load_checkpoints`/`build_dags`/`build_strategies` consistent between Task 4 and Task 6.

**Note on robustness column:** with a single noise seed (smoke default), per-instance makespan std is `NaN` (pandas std of one value); `robustness` will be NaN in smoke runs and meaningful only with ≥2 noise seeds. The full-grid preset uses ≥5 noise seeds. Tests use 2 noise seeds where robustness is asserted to exist as a column (not a specific value).
