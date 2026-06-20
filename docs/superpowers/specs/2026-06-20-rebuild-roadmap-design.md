# SmartDAG Scheduler — Rebuild Roadmap (Design)

**Date:** 2026-06-20
**Status:** Approved decomposition; ready for `writing-plans` (M1 first).
**Scope of this document:** sequencing and milestone boundaries for rebuilding the
SmartDAG Scheduler from zero. It does **not** redefine the architecture — that is
fully specified and locked in [`SmartDAG_Scheduler_TZ.md`](../../../SmartDAG_Scheduler_TZ.md)
(the ТЗ, incl. **Appendix A**), which remains the single source of truth. Where this
roadmap and the ТЗ ever disagree about *what* the system does, the ТЗ wins; this
document only governs *the order in which it gets built*.

## Context: why a roadmap and not a design

As of 2026-06-20 the workspace contains **only three files** —
`SmartDAG_Scheduler_TZ.md`, `claude.md`, `.cursorrules`. There is no `src/`, no
tests, no `docs/`, no virtualenv, and it is not a git repository. A previous build
reached "M1 green, 38 tests" on a branch `m1-core-simulator`, but that code did not
survive to this machine and is treated as **lost and unrecoverable** (user decision:
rebuild fresh, do not attempt recovery).

Because the architecture is already locked (and the project's standing instruction is
"do not revisit locked decisions without a raised problem"), there is nothing to
*brainstorm at the design level*. The open work is purely **decomposition**: how to
slice the rebuild into independently testable milestones and in what order.

## Decomposition strategy

**Chosen: capability layers, deterministic-first.**
core+simulator → baselines+fairness → RL network → RL training → stochasticity →
eval/CLI. Each milestone is independently testable and maps cleanly to ТЗ sections.
This front-loads the deterministic, hand-checkable engine, which ТЗ §5.2 mandates
("validate the engine in this mode first… then turn the knobs on"). It is also the
slicing the prior (lost) build used successfully.

Alternatives considered and rejected:

- **Thin end-to-end slice first** (trivial DAG → random strategy → CSV across all
  layers, then deepen). Good for early integration, but the thesis value lives in
  *component correctness* (faithful HEFT, exact reward telescoping, correct masking);
  thin-slicing dilutes that and forces rework on the hardest parts (GNN/PPO).
- **Risk-first** (build GNN/PPO earliest against a stub env). De-risks the novel part
  but requires validating an RL agent against an *unvalidated* simulator with no
  correct baselines to compare against — two unknowns at once. The ТЗ deliberately
  orders deterministic-first.

## Universal exit gate (every milestone)

A milestone is "done" only when **all** of the following hold:

- `pytest` green (the milestone's named tests below, plus all prior milestones'
  tests still passing — no regressions).
- `ruff` and `black` clean.
- Full type hints on all new public surfaces.
- The milestone's specific exit criteria (below) are demonstrably met.

Tooling is invoked via `.venv/bin/...` (bare `python`/`pytest`/`ruff`/`black` are not
on PATH; PEP 668 environment).

## Milestones

### M1 — Core + deterministic simulator (`noise_std=0, failure_rate=0`)

**Goal:** a hand-checkable, deterministic, one-assignment-per-`step` environment.

**Deliverables (ТЗ §12):**
- `core/`: `task.py`, `dag.py`, `compute_node.py`, `schedule.py`
- `env/`: `cost_model.py`, `placement.py`, `observation.py`, `cluster_factory.py`,
  `cluster_env.py`
- `dag_factory/`: `factory.py`, `synthetic.py`
- `utils/`: `seeding.py`, `normalization.py`, `config.py`

**Locked invariants (ТЗ Appendix A):**
- `placement.weighted_cost(task, node, cluster_state)` is the **single**
  side-effect-free objective evaluator; it returns the *individual normalised
  components* `Δmakespan/M_ref` and `Δenergy/E_ref` (not a pre-summed scalar). The env
  negates+weight-sums them into the per-step reward (A.1).
- `Δmakespan_t = max(free_at_time)_after − _before` (running schedule horizon; `0` if
  the task fits under the current horizon) — **not** the task's own finish time (A.2).
- `M_ref` = fastest-exec critical-path lower bound; `E_ref = Σ_i min_node(energy_{i,node})`;
  both computed once at `reset()` and cached; emitted in the Gymnasium **`info` dict,
  not `obs`** (A.2, A.3).
- `node_id == index` enforced in `ClusterEnv.__init__`.
- Synthetic `base_cost` is wide / heavy-tailed (log-normal or wide uniform); uniform
  task sizes are forbidden (A.3).
- Layered-random DAG generation (acyclic by construction).
- Isolated `make_rng`-style generators, not global numpy seeding.

**Exit criteria / tests:** `test_dag`, `test_cost_model`, `test_env_step`
(asserts cross-step transitions: status `ready → done`, child
`#unscheduled_predecessors` decrements), `test_masking`, `test_reward_telescoping`
(Σ per-step reward components reconstruct total makespan/energy under γ=1), and a
committed **golden-schedule fixture** (tiny 3–4 task DAG, known-optimal deterministic
schedule) as a permanent regression anchor (A.4). A deterministic episode is exactly
N steps.

**Deliberately deferred:** GNN-facing normalization fixes (`N_SPEED` node mean speed
unnormalized; `T_MEM` normalized by wrong scale) → **M3a**, where the first consumer
(the GNN) exists. `T_DONE`/`T_SCHEDULED` columns intentionally coincide here and
**diverge under failures** → **M4** (do not "dedupe" them).

### M2 — Baselines + scheduler loop + fairness

**Goal:** classical strategies behind one `Strategy` interface, a decision-point loop,
and the fairness harness that guards the thesis invariant.

**Deliverables (ТЗ §8, §12):**
- `strategies/`: `base.py`, `heft.py`, `cpop.py`, `min_min.py`,
  `weighted_sum_greedy.py`, `random_strategy.py`, `ranking.py`
  (mean_exec/mean_comm/upward_rank/downward_rank — also reused as GNN features in M3a).
- `scheduler/task_scheduler.py`: decision-point loop, Observer subscriber.
- `SystemMonitor` (Observer) scaffold — **idle in deterministic mode**; becomes active
  in M4.

**Key constraints:**
- Weighted-Sum Greedy minimises `w1·(Δmakespan/M_ref) + w2·(Δenergy/E_ref)` over
  candidate `(task, node)` pairs via the **same** `placement.weighted_cost` the env
  uses for reward — parity is structural, not coincidental (ТЗ §8, A.1).
- Each heuristic is run **faithfully as itself**; no "upgrading" a heuristic to match
  agent capabilities. Placement is append-only EFT; baselines are the no-insertion
  HEFT/CPOP forms (stated honestly in the thesis).
- HEFT/CPOP structural caches keyed by DAG via `WeakKeyDictionary` (no leak across
  sampled DAGs).
- All strategies share the identical `predict(ready, cluster_state) -> (task, node)`
  interface. Strategies returning a node index use the node's list position; when
  filtering to `alive` nodes, the original index is preserved for the action.

**Exit criteria / tests:** `test_heft`, `test_cpop`, `test_min_min` (faithful, golden
reference values), `test_fairness` (same DAG + same seed ⇒ all strategies receive
identical inputs; greedy↔reward parity proven via the shared `weighted_cost`).

### M3a — RL network: encoder + two-head policy (no training)

**Goal:** the policy/value network, fully unit-tested for shapes, masking, and
log-prob — **before** any training loop exists.

**Deliverables (ТЗ §6.1–§6.3, §12 `rl/`):**
- `rl/gnn_encoder.py`: GraphSAGE (PyG, **not** DGL), bidirectional (forward+reverse
  edges, direction-flagged), 2–3 layers; per-task embeddings `h_i` + pooled graph
  embedding `g`; node-feature MLP → `n_j`, pooled → `c`.
- `rl/policy.py`: two-head autoregressive policy. Head 1 (ordering) scores ready tasks
  `score_i = MLP([h_i, globals])`, mask non-ready, softmax → τ. Head 2 (assignment)
  scores alive nodes `score_j = MLP([h_τ, n_j, globals])`, mask dead/unavailable,
  softmax → ν. Critic `V = MLP([g, c, globals])` scalar. Pointer scoring + masking
  over variable candidate sets — **never** fixed `Discrete`.
- Resolve the M1-deferred normalization (`N_SPEED`, `T_MEM`) now that the GNN consumes
  the observation.

**Exit criteria / tests:** `test_masking` at the policy level (never assigns a
probability to a dead/non-ready candidate); joint log-prob `= logπ(τ) + logπ(ν)`;
gradient flows through both heads and the critic; correct handling of variable
candidate-set sizes across a batch. No training is run in this milestone.

### M3b — RL training: custom PPO

**Goal:** train the M3a network on the deterministic env and clear the learning sanity
gates.

**Deliverables (ТЗ §6.5, §9, §12 `rl/`):**
- `rl/rollout_buffer.py`: **tensorises per-node features at the decision point** —
  never stores the live `Observation` object (it is a mutable reference). Per-step
  value targets must account for the terminal-only `w3·balance` reward term.
- `rl/ppo_trainer.py`: rollout buffer, GAE (λ≈0.95), clipped surrogate (ε≈0.2), value
  loss, entropy bonus, advantage normalisation, Adam. No DQN/PER/target network.
- Training budget + early stop: fixed env-step/epoch budget; every K updates run a
  frozen-policy eval on a fixed validation DAG set vs HEFT; stop on plateau or budget.
- Checkpoints → `models/*.pth` (and to Drive/Kaggle dataset when on cloud).

**Exit criteria / sanity gates (ТЗ §9):** beats Random quickly; reward rises then
plateaus; eval-vs-HEFT improves over training. A flat reward from the start is a bug
(reward scaling / masking / advantage), not "needs more training." Trains on the
target scale (20–60 tasks, 4–16 nodes).

### M4 — Stochasticity & failures (same env, knobs on)

**Goal:** turn on `noise_std` and `failure_rate` on the **same** environment; the
Observer becomes active; strategies react on live state.

**Deliverables (ТЗ §5.2, §3):**
- Reveal-at-completion noise: actual duration = nominal·(1+ε), ε sampled at task
  completion.
- Per-unit-time failure hazard: on failure the running task is lost and **requeued**
  into the ready set, the node is marked `alive=False`; `SystemMonitor` fires; the next
  decision point simply reflects the new state (no special intervention machinery).
- `T_DONE` vs `T_SCHEDULED` now **diverge** (done = executed vs scheduled =
  assignment committed).
- All strategies re-assign onto surviving nodes automatically via their existing
  reactive `predict` (no per-strategy failure code).
- Optional curriculum: clean → +noise → +failures (may be skipped).

**Exit criteria:** a failure correctly requeues the task and marks the node dead; the
**fairness invariant holds** under shared failure events + shared noise seed across all
strategies; with knobs at `(0, 0)` the env still reproduces the **M1 golden schedule**
(regression anchor).

### M5 — Evaluation + CLI (+ WfCommons)

**Goal:** the experiment runner, metrics, comparison output, CLI entry point, and
realistic-workflow benchmark source.

**Deliverables (ТЗ §7, §10, §12):**
- `eval/metrics.py`: makespan `C_max`, total energy, resource utilisation,
  load-balance index (`1 − CV` over **all alive** nodes), SLR, speedup, scheduling
  overhead (inference time), robustness (makespan std under noise).
- `eval/evaluate.py`: regime grid noise ∈ {0, 0.1, 0.2} × β ∈ {2, 5, 10} ×
  failures ∈ {off, on}; 30–50 held-out DAGs; 5–10 noise seeds per DAG; 3–5 independent
  training seeds (report mean ± std); Wilcoxon signed-rank (paired over common
  instances) for significance.
- `dag_factory/wfcommons_adapter.py`: `pip install wfcommons`; parse
  `Recipe.from_num_tasks(N).build_workflow()` WfFormat JSON → `TaskDAG`. Placed here
  (not M1) — it is a benchmark-credibility feature for evaluation and depends on
  nothing earlier (ТЗ §7: "after the synthetic path works").
- `drl_scheduler.py`: CLI entry point (`train` / `eval`); `config.yaml`
  (all hyperparams + experiment config); optional TensorBoard curves.

**Exit criteria:** console comparison table + `results/*.csv`; Wilcoxon paired test
runs; runs reproducibly from `config.yaml`. Claims like "10–25% makespan improvement
over HEFT" are reported **qualified as under dynamic conditions** (ТЗ §14).

## Dependency order

```
M1 ──> M2 ──> M3a ──> M3b ──> M4 ──> M5
                              (M5 also depends on M2 baselines for comparison
                               and on M3b for the trained agent under eval)
```

Strict linear order; no parallelism assumed. `writing-plans` will be invoked
**per milestone, starting with M1** — each milestone gets its own implementation plan.

## Non-goals (ТЗ §13 — do NOT implement)

Task deadlines; GUI/dashboard/Streamlit/web/REST/gRPC; database/Kubernetes/Mesos/
MLflow/real-cluster/human-in-the-loop; DQN/PER/target networks; DGL (use PyG); fixed
`Discrete` action space (use pointer scoring + masking); anything that breaks the
fairness invariant (§8).
