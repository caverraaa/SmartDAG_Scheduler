# M5 — Evaluation + CLI (design)

**Date:** 2026-06-20
**Milestone:** M5 (final milestone). Depends on M1–M4 + WfCommons handling (all complete).
**Status:** approved design, ready for implementation plan
**Source of truth:** `SmartDAG_Scheduler_TZ.md` §7, §10, §12. Where this disagrees with the ТЗ, the ТЗ wins.

## 1. Goal & scope

The experiment runner, metrics, comparison output, and CLI entry point that turn the
trained agent + baselines into thesis evidence. One cohesive milestone, one spec:
metrics → evaluate → CLI.

**In scope:** per-run metrics; the regime-grid evaluation runner with paired
significance testing; the `drl_scheduler.py` CLI (`train` / `eval`), all driven by
`config.yaml`.

**Out of scope (non-goals):** GUI/dashboard/TensorBoard live curves (training history is
written to CSV instead); real-cluster execution; any change to the env / strategies /
fairness paths; production-scale (1000s of tasks) — eval stays in the trainable range.

## 2. Decisions (locked)

1. **Significance via scipy.** Add `scipy>=1.11` to core requirements and use
   `scipy.stats.wilcoxon`. (This promotes scipy from optional-transitive — it currently
   arrives only via the optional `wfcommons` extra — to a core dependency; a defensible
   deviation from the ТЗ stack listing, noted here.)
2. **Eval consumes pre-trained checkpoints.** `train` is run once per training seed and
   writes `models/rl_seed{N}.pth`; `eval` loads the checkpoint set and aggregates
   mean ± std across them. Training cost is decoupled from eval iteration.
3. **Config-driven grid, small smoke defaults.** All grid axes/counts live in a
   `config.yaml` `eval:` block. Defaults are a fast smoke-sized grid; the full thesis
   grid (ТЗ §10) is a documented preset.
4. **No TensorBoard.** `train` writes `results/train_history_seed{N}.csv` from the
   existing `PPOTrainer.train` history; curves can be plotted from CSV externally.

## 3. Module layout

```
src/eval/
  eval_config.py    # EvalConfig dataclass + load_eval_config(path) reads config.yaml `eval:` block
  metrics.py        # pure per-run metric fns + compute_run_metrics(...) -> dict[str, float]
  significance.py   # paired_wilcoxon(a, b) -> (stat, p)   (scipy wrapper + edge-case guard)
  evaluate.py       # regime-grid runner: build instances, run grid, aggregate, write CSV + tables
drl_scheduler.py    # CLI entry: `train` / `eval` subcommands (argparse), config-driven
config.yaml         # + `eval:` block
requirements.txt    # + scipy>=1.11
```

`eval/` is a new layer above `scheduler`/`rl` (it depends on them; nothing depends on it).
`metrics.py` and `significance.py` are pure and unit-testable in isolation; `evaluate.py`
orchestrates; `drl_scheduler.py` is a thin CLI shell. No back-dependencies (ТЗ layering).

## 4. Metric contract (`metrics.py`)

`compute_run_metrics(schedule, info, dag, nodes, alive_ids, predict_seconds) -> dict`
returns one row of metrics for a finished episode. Definitions (pinned, disclosed in
thesis):

- **makespan** `C_max` = `schedule.makespan()` (equals `info["makespan"]`).
- **energy** = `schedule.total_energy`.
- **utilisation** = `Σ_i busy_i / (C_max × |alive_ids|)` where `busy_i` is from
  `schedule.busy_time_by_node()`; `0.0` when `C_max == 0`.
- **load_balance** = `schedule.load_balance_index(alive_ids)` (existing; `1 − CV` of busy
  time over alive nodes).
- **slr** = `C_max / info["m_ref"]` (`m_ref` = fastest-exec critical-path lower bound,
  cached at reset). `m_ref > 0` always for a non-empty DAG; guard `m_ref == 0 → slr = 0`.
- **speedup** = `(min over nodes of Σ_i exec(task_i, node)) / C_max` — Topçuoğlu
  definition: serial time on the single fastest node ÷ parallel makespan. `exec` is
  `cost_model.exec_time(task, node)`. Computed over all DAG tasks and all (alive) nodes.
- **overhead_ms** = total wall-clock time spent in `strategy.predict` across the episode,
  in milliseconds. Captured by a `TimingStrategy(BaseSchedulingStrategy)` wrapper that
  delegates to the wrapped strategy and accumulates `time.perf_counter()` deltas, so
  `run_episode` is unchanged.
- **robustness** = std of `C_max` across the noise seeds of a fixed instance — computed at
  the **aggregation** stage (§6), not per run.

`compute_run_metrics` is side-effect-free and reads only the finished schedule + cached
refs; it never re-runs the episode.

## 5. Regime grid & fairness (`evaluate.py`)

### 5.1 Instances
- **Held-out synthetic DAGs:** generated via `DAGFactory.create("synthetic", rng, ...)`
  from a pinned **held-out seed stream** (a fixed offset, e.g. `eval.dag_seed_base`,
  disjoint from any training seed), sizes sampled in the 20–60 trainable range. Count =
  `eval.n_dags`.
- **WfCommons benchmarks:** every committed `dag_benchmarks/*.json`, loaded via
  `DAGFactory.load_from_wfcommons(path, rng, recipe=...)` — inherently held-out (never
  trained on).
- Each DAG is paired with **one cluster** built by `make_cluster(rng, n_nodes, β)` per
  `(instance, β)` and **reused across noise seeds** (robustness must vary only the noise,
  not the hardware).

### 5.2 Regimes
A **regime** = `(noise_std, β, failures)` drawn from the `eval:` config lists
(`noise_std ∈ {0, 0.1, 0.2}`, `β ∈ {2, 5, 10}`, `failures ∈ {off, on}` at full size).
`noise_std`/`failure_rate` are set on the `ClusterEnv` config; `β` feeds `make_cluster`.

### 5.3 Noise seeds & the fairness invariant
A **noise seed** = a distinct env `config.seed`. The env builds its noise/failure calendar
at `reset()` from isolated `derive_rng` streams keyed by `(seed, instance_signature)`
(M4). Therefore, for one `(regime, instance, noise_seed)`, **every strategy receives the
identical noise ε and identical failure events** — the thesis fairness invariant, enforced
structurally by running all strategies through the same `run_episode` + `env.reset`. No
strategy ever gets different instances/seeds/events. There is no per-strategy special
casing anywhere in the runner.

### 5.4 Loop
```
for regime in regimes:
  for instance in instances:                 # (dag, nodes) — nodes fixed per (instance, β)
    for noise_seed in eval.noise_seeds:
      env = ClusterEnv(config(seed=noise_seed, noise_std=…, failure_rate=…))
      for strategy in strategies:            # HEFT, CPOP, MinMin, WeightedSumGreedy, Random, RL×ckpts
        sched, info = run_episode(env, TimingStrategy(strategy), dag, nodes, monitor)
        row = compute_run_metrics(sched, info, dag, nodes, alive_ids, predict_seconds) + identifiers
```
`strategies` is built once: the heuristics, plus one `RLStrategy` per loaded checkpoint
(labelled `rl@seed{N}`).

## 6. Aggregation, significance & output

- All rows → a pandas `DataFrame` → `results/eval_runs.csv` (raw, one row per
  regime × instance × noise_seed × strategy).
- **Summary** → `results/eval_summary.csv`: mean ± std grouped by `regime × strategy`, per
  metric. `robustness` = std of `C_max` within each `(regime, instance, strategy)` group,
  then averaged. RL rows are aggregated **two ways**: (a) per-instance mean over training
  seeds (the comparison value), and (b) cross-training-seed std reported as a separate
  `train_std` column (training variability, ТЗ "report mean ± std over 3–5 training
  seeds").
- **Significance** (`significance.py`): `paired_wilcoxon(rl, baseline)` over common
  instances (paired on `(instance, noise_seed)`), RL vs each baseline, per regime, on
  makespan (and energy where the comparison is objective-level). The wrapper guards the
  degenerate all-equal / all-zero-difference case (returns `p = 1.0` rather than letting
  scipy raise).
- **Console**: one comparison table per regime — strategy rows × metric cols (mean±std) —
  followed by RL-vs-baseline p-values.

## 7. CLI (`drl_scheduler.py`)

- `train --seed N`: build cluster + `TwoHeadPolicy` from `config.yaml`, run
  `PPOTrainer.train` for `config.total_updates`, `save_checkpoint("models/rl_seed{N}.pth")`,
  and write the returned history to `results/train_history_seed{N}.csv`. Run once per
  training seed.
- `eval`: build instances + regimes from config, run the grid (§5–§6), write the CSVs, and
  print the per-regime tables. Checkpoint glob comes from `eval.checkpoint_glob`.
- Pure `argparse`; all hyperparameters/experiment settings from `config.yaml` (no magic
  numbers in code).

## 8. Config (`eval:` block)

A dedicated `EvalConfig` frozen dataclass + `load_eval_config(path="config.yaml")` reading
a new `eval:` block (mirrors the `WfcommonsParams` pattern; does **not** touch the flat
`Config` used by the env/trainer). Fields (smoke defaults; full-grid values in comments):

```yaml
eval:
  noise_std: [0.0, 0.1, 0.2]
  beta: [2.0, 5.0, 10.0]
  failure_rate: 0.0          # the "on" value when failures enabled; 0.0 disables
  failures: [false, true]
  n_dags: 5                  # full: 30–50
  dag_sizes: [20, 40, 60]    # sampled for held-out synthetic DAGs
  n_nodes: 8
  noise_seeds: [0, 1, 2]     # full: 5–10 distinct seeds
  dag_seed_base: 100000      # held-out stream offset (disjoint from training seeds)
  benchmark_dir: dag_benchmarks
  checkpoint_glob: models/rl_seed*.pth
  results_dir: results
```

## 9. Testing (TDD — failing test first)

- `test_metrics.py`: hand-checked metrics on the **golden instance** (the existing
  diamond: makespan 6, energy 1200, balance 0 on all-GPU). SLR/speedup/utilisation
  computed by hand from the fixture and asserted.
- `test_significance.py`: `paired_wilcoxon` vs known reference values; the all-equal guard
  returns `p = 1.0`.
- `test_evaluate.py`: a tiny 1-regime, 2-DAG, 2-noise-seed smoke grid with a single tiny
  RL checkpoint (so the RL-vs-baseline significance path runs) — asserts the **fairness
  invariant** (every strategy ran the identical `(instance, noise_seed)` set), the expected
  CSV columns exist, and Wilcoxon runs end-to-end.
- `test_cli.py`: `train --seed 0` (tiny config: few updates) writes a checkpoint + history
  CSV; `eval` (tiny grid, one checkpoint) writes `results/eval_summary.csv`. Kept light.
- **Regression anchor:** the `(noise_std=0, failures=off)` regime still reproduces the M1
  golden schedule.

## 10. Exit criteria (ТЗ §7, §10, §12)

1. Console comparison table + `results/*.csv` produced by `drl_scheduler eval`.
2. Wilcoxon paired test runs (RL vs each baseline, per regime).
3. The eval grid loads both held-out synthetic DAGs and the committed WfCommons benchmark
   set through `DAGFactory`.
4. Runs reproducibly from `config.yaml` (pinned held-out DAG seeds + noise seeds).
5. Improvement claims (e.g. "10–25% makespan over HEFT") are reported **qualified as under
   dynamic conditions** (ТЗ §14) — the runner reports per-regime so the qualification is
   structural.
6. `pytest` green (new + all prior tests), ruff + black clean, full type hints.

## 11. Notes / deviations

- scipy promoted to a core dependency (§2.1) — recorded deviation from the ТЗ stack list.
- `wfcommons` remains an optional extra; the eval runner loads the **committed** benchmark
  JSONs and never imports `wfcommons`.
- Multi-training-seed RL aggregation reports both the comparison mean (over training seeds,
  per instance) and the training-variability std — see §6.
