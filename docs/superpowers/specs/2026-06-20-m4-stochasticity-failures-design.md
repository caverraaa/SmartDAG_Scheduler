# M4 — Stochasticity & Failures (Design)

**Date:** 2026-06-20
**Status:** Approved; ready for `writing-plans`.
**Scope:** Add execution-time **noise** and node **failures** to the existing `ClusterEnv`, activate the `SystemMonitor` Observer, and ensure all strategies (heuristics + RL via `RLStrategy`) react to the new live state through their existing `predict` interface. Source: roadmap M4 section + `SmartDAG_Scheduler_TZ.md` §5.2/§5.3/§3/§8/§14 + the M2→M4 carry-forward. Builds on merged M1–M3b.

## Context & the resolved tension

Spec §3/§5.3 describe an **event-driven** simulator (running clock + event queue, failures firing between decision points across concurrently in-flight tasks). M1 deliberately built a **simplified append-only-EFT** env (one `step` = one full placement; no running clock). M4 keeps the append-only env ("same env, knobs on") and expresses node loss, requeue, re-assignment onto survivors, and shared adversity within it. The one thing the event-queue would add — failures firing mid-execution across concurrent in-flight tasks with in-flight replanning — is a production concern, **out of scope**, and disclosed honestly (§14), consistent with the existing append-only / no-insertion simplifications.

**Rejected alternative (per-placement hazard):** sampling failure over the *assigned* task's duration (`p = 1 − exp(−rate·duration)`) makes failures a function of each strategy's own choices (which task, how long, which node, plus RNG desync from differing placement order). That yields strategy-dependent failure realizations, which (a) contradict the M4 exit gate's "shared failure events", (b) violate the §8 fairness invariant, and (c) reinterpret §5.2's per-unit-*simulated-time* hazard as a per-task hazard. **Replaced by the exogenous failure calendar below.**

## Architecture

Two config knobs on the one env: `noise_std`, `failure_rate` (already present in `Config`). `(0,0)` = deterministic = the M1 behaviour, reproduced **bit-for-bit**. Both sources draw from **isolated RNG streams** so zeroing the knobs cannot perturb the DAG/cluster-generation stream (the golden-schedule regression depends on this).

### 1. Exogenous failure calendar

- At `reset()`, from an **isolated failure RNG** (`make_rng` keyed on `seed` + a "failure" salt), draw one time-to-failure per node: `t_f[node_id] ~ Exponential(failure_rate)`. One failure per node per episode (a dead node stays dead; no cascades/streams). `failure_rate == 0` ⇒ `t_f = +inf` (never fires).
- The calendar is computed **before any strategy runs**, is **strategy-independent**, and is cached in `ClusterState` (`failure_times: dict[int, float]`).
- **Placement-failure rule (precise):** when a strategy places task `T` on node `X` with EFT window `[start, finish]` (finish using the *actual* noisy duration — see §2):
  - `finish <= t_f[X]` → **success**: commit `T` (normal append-only placement).
  - `finish > t_f[X]` → **failure**: `X.alive = False`; `T` is **not** committed (stays in the ready set / requeued); `SystemMonitor` records the event; nothing is charged (no partial energy/time). The next decision point sees the survivor set; the strategy re-assigns `T` via its existing `predict`.
- This single rule subsumes all cases: a task spanning `t_f` is lost; a task placed entirely after `t_f` (idle-gap or post-death) has `finish > t_f` and correctly fails on the now-dead node; a node whose `t_f` falls after its last use is never exercised again ⇒ "no impact" (it is simply never marked dead, equivalent to having finished its work before dying).
- **Detection is lazy**: a node is marked dead the first time a placement on it would cross its `t_f`. A strategy that picks such a node "discovers" the death via one failed placement (a realistic, bounded adaptation cost — at most one wasted decision per node, since the node is dead thereafter).
- **"Longer occupancy → more likely hit" is preserved** at the impact level (a node kept busy longer has later finishes, more likely `finish > t_f`), while the *event* itself stays fixed and exogenous. A fixed calendar is **not hackable**: the agent cannot make a node fail later by idling it (which a per-occupancy hazard would have rewarded, confounding robustness with exposure-avoidance).

### 2. Noise (reveal-at-completion)

- At `reset()`, from an **isolated noise RNG**, draw `ε[task_id]` per task (keyed by `task_id`, **not** placement/draw order), cached in `ClusterState` (`noise_eps: dict[int, float]`).
- Actual execution time = `nominal_exec · (1 + ε)`, clamped to `≥ 0`. Planning is on nominal: the observation features, `placement.weighted_cost`, and the strategies all use **nominal** `exec_time`; only the committed `Assignment.finish` and the node's `free_at_time` use the **actual** (noisy) duration. The agent plans on nominal values and reality differs — this is what makes robustness meaningful.
- Same `ε` per task across every strategy ⇒ "same noise seed" fairness holds. `noise_std == 0` ⇒ all `ε = 0` ⇒ actual == nominal.

### 3. Reward & accounting under stochasticity

- **Successful step** (unchanged shape): `r = −(w1·Δmakespan/M_ref + w2·Δenergy/E_ref)`, where `Δmakespan` is the horizon delta and `Δenergy` is the **actual** committed energy. Terminal adds `+w3·balance`.
- **Failed step:** reward `0`. Nothing is committed; **no** Δmakespan is computed from the horizon — dropping a node from `max(free_at_time)` could otherwise *spuriously reward* the failure. The cost is emergent: the requeued task pushes the horizon later, and the survivor set is smaller.
- **Horizon (for reward + makespan metric):** `max(free_at_time)` over **all nodes that have committed work** (a dead node's *completed* tasks still elapsed and count toward makespan). Keeps the horizon monotone non-decreasing across successful steps. **Placement candidates remain alive-only.**
- **Energy telescoping stays exact:** Σ energy over successfully committed tasks = total energy of the final schedule, so the M1–M3b reward machinery is untouched. (Makespan telescoping is exact only at `(0,0)`; under noise/failures it is approximate, which is expected and acceptable.)
- **Disclosure (thesis §14):** the no-partial-charge-on-failure simplification is disclosed alongside the existing append-only / no-insertion notes.

### 4. `T_DONE` vs `T_SCHEDULED` (do not dedupe)

Both observation columns are kept and wired to **distinct sources** — `T_SCHEDULED` from the committed set, `T_DONE` from the completed set — and each `Assignment` keeps both `start` (commit) and `finish` (noise-revealed actual completion). In this append-only / no-partial model a commit completes atomically, so the two *flags* still coincide at every decision point; the genuine divergence lives in the `Assignment` times (`start ≠ finish`, and the actual finish ≠ nominal, later after a requeue) and in the metrics. We keep the columns distinct (no dedupe, per the M1 carry-forward) but do **not** manufacture a fake in-flight flag state.

### 5. SystemMonitor activation

`SystemMonitor.check(state)` now returns the failure event(s) realized at the current decision point (the nodes that just died). `run_episode` already calls `monitor.check` each decision point (the uniform Observer trigger established in M2). The failure *mechanism* lives in `env.step` (the calendar check); the monitor is the observability/event channel — uniform across all strategies, preserving fairness.

### 6. Load-balance index fix (M2→M4 carry-forward)

`Schedule.load_balance_index` currently iterates `range(n_alive_nodes)` (a dense `0..k-1` prefix), which is wrong once a mid-range node dies (e.g. alive `{0,2}`, dead `1`). M4 changes it to compute `1 − CV` over the **actual alive node ids** (busy time keyed by node id, restricted to the survivors at terminal).

### 7. Deadlock handling

If unscheduled tasks remain but **zero nodes are alive**, the episode terminates `done=True` with `info["deadlocked"]=True`; the makespan metric is the partial makespan so far (M5 eval records completion-rate as a robustness axis). `run_episode` additionally guards: it never calls `predict` when there are no alive nodes (it ends the episode instead). At normal `failure_rate` this is rare; it is handled so it fails gracefully rather than crashing.

## Components touched

- `src/env/cluster_env.py`: `reset` builds the failure calendar + noise map from isolated RNGs and stores them in `ClusterState`; `step` applies noise to the committed duration, applies the placement-failure rule (success/failure branch), emits failure info, handles the deadlock terminal. Horizon over committed nodes.
- `src/env/placement.py`: `ClusterState` gains `failure_times: dict[int,float]` and `noise_eps: dict[int,float]`. `weighted_cost` stays nominal (planning). A helper exposes the actual (noisy) finish for the commit path without disturbing the nominal evaluator.
- `src/core/schedule.py`: `load_balance_index` over actual alive node ids.
- `src/scheduler/system_monitor.py`: `check` returns realized failure events.
- `src/scheduler/task_scheduler.py`: deadlock guard (no alive nodes ⇒ end).
- `src/utils/seeding.py`: isolated per-concern RNGs (a small helper to derive named sub-streams from the base seed) if not already expressible.

## Testing

- `test_noise`: with `noise_std>0`, committed finishes differ from nominal by the per-task `ε`; planning/`weighted_cost` unchanged; same `ε` per task across two strategies.
- `test_failures`: a node whose `t_f` is crossed by a placement → node `alive=False`, task requeued (re-appears in `ready_set`), `SystemMonitor.check` returns the event; the requeued task completes later on a survivor; episode still completes.
- `test_fairness` (extended): the failure calendar and noise map are **bit-identical across all strategies** for a fixed instance+seed.
- `test_deadlock`: all nodes dead with tasks remaining ⇒ `done=True`, `info["deadlocked"]=True`, no crash.
- `test_load_balance_alive_ids`: a dead mid-range node is excluded from the balance CV (iterates actual alive ids).
- **Golden regression:** at `(noise_std=0, failure_rate=0)` the env reproduces the **M1 golden schedule bit-for-bit** (RNG isolation guarantees the deterministic stream is unperturbed) — the permanent regression anchor.

## Non-goals (M4)

Event-queue / running-clock simulator; failures firing mid-execution across concurrent in-flight tasks; partial-work energy/time charging; cascading/streamed failures (one per node); retraining the agent under the regimes (that is M5 eval); the full regime grid / Wilcoxon / CLI / WfCommons (M5).
