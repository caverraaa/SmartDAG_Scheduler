# SmartDAG Scheduler — Project Specification (Engineering ТЗ)

**Thesis topic:** «Дослідження методів, моделей та інформаційних технологій для розробки оптимізуючого планувальника задач для гетерогенних обчислювальних систем з використанням reinforcement learning.»

This document is the single source of truth for the implementation. It describes the research prototype we are actually building, not the broader production vision. All earlier lab/course diagrams are treated as background only; where they conflict with this document, this document wins.

## 1. Purpose & scope

**What it is:** a CLI-only, research-grade program-modelling complex (ПМК) that
- trains a Deep RL scheduling agent on simulated heterogeneous clusters,
- benchmarks it against classical heuristics on the same instances, and
- emits metrics (console comparison table + CSV, optional TensorBoard curves) used for the thesis analysis.

**Scientific claim (what the experiments must support):** RL does not "win everything." It wins structurally on multi-criteria objectives (heuristics like HEFT ignore energy and load balance) and empirically on robustness under execution-time noise and node failures. The deliverable characterises where and why RL beats heuristics.

**In scope:** simulation environment, DRL agent, baselines, experiment runner, metrics, reproducible config.

**Out of scope (hard non-goals):** GUI/dashboard, web/REST/gRPC, database, Kubernetes/Mesos, human-in-the-loop approval, real hardware, task deadlines. See §13.

## 2. Problem formulation (MDP)

DAG scheduling on heterogeneous nodes, formalised as a sequential-decision MDP ⟨S, A, P, R, γ⟩, solved by constructing one full schedule per episode via N decisions (N = number of tasks).

- **State s_t** — the DAG with per-task features + the live cluster state (see §6.2).
- **Action a_t** — a two-part decision: pick a ready task (ordering) and a compute node (assignment). See §6.3.
- **Transition P** — deterministic placement + event-driven time advance; stochastic execution times and node failures inject randomness (see §5.2, §7).
- **Reward R** — hybrid telescoping multi-criteria (see §6.4).
- **γ = 1.0** — finite-horizon episodic; γ=1 is required so the dense per-step reward telescopes exactly to the totals.

## 3. Architecture overview

Event-driven, decision-point simulation. The agent is queried only at decision points (a moment in simulated time where a ready task and an available node co-exist). Between decision points the clock fast-forwards through the event queue (task finishes, node frees, node failures). One `env.step(action)` = one assignment, not one time tick.

**Design patterns:**
- **Factory (DAGFactory)** — isolates DAG generation/parsing from training.
- **Strategy (BaseSchedulingStrategy)** — RL agent and every heuristic implement the same `predict(ready_tasks, cluster_state) -> (task, node)` interface; swappable at runtime.
- **Observer (SystemMonitor)** — detects failure/overload events and triggers reoptimisation uniformly for all strategies (fairness invariant, §8).

**Layering:** core data → environment → strategies → RL → scheduler loop → evaluation. Designed so a presentation layer (CLI now, optional Streamlit later) and, in principle, a real-cluster backend could be added without touching the core. Not built now.

## 4. Core data structures

- **Task** — `id`, `base_cost`, `mem_required`, `task_class` (`data_parallel | sequential | streaming`), `out_data` (per-edge volumes live on the DAG edges). No deadline field.
- **TaskDAG** — wraps `networkx.DiGraph`; checks acyclicity; computes critical path, `b_level` (longest path to sink) and `t_level`; tracks the ready set (tasks whose predecessors are all scheduled/finished).
- **ComputeNode** — `node_id`, `node_type` (`CPU | GPU | FPGA | TPU`), `speed_by_class`, `power_w`, `bandwidth`, `free_at_time`, `alive` flag.
- **Schedule / Assignment** — the result object: ordered Assignments (task→node with start/finish) plus integral metrics (makespan, energy, load-balance index).

## 5. Environment & cost model

### 5.1 Cost / time / energy / comms

**Heterogeneous cost matrix:** `exec_time = base_cost / speed[task_class][node_type]`. The speed table is generated so that the max/min ratio across node types ≈ heterogeneity factor β (default range 2–10). Task classes make the right node non-trivial (data-parallel → GPU/TPU, sequential/branchy → CPU, streaming → FPGA).

**Energy:** `E = power_w[node_type] * exec_time`. Power ratings in plausible TDP ranges (GPU ~250–400 W, TPU ~200–450 W, CPU ~65–150 W, FPGA ~30–75 W). This is the source of the genuine makespan↔energy conflict (fast node = less time, more watts).

**Communication:** each DAG edge i→j carries data volume `d_ij`; cross-node transfer = `d_ij / bandwidth` (+ latency), intra-node ≈ 0. Edge volumes scale with the communication-to-computation ratio (CCR) parameter.

**Load-balance index (terminal):** `1 − CV(per-node busy time)` (coefficient of variation), normalised to 0..1, higher = more even.

### 5.2 Stochasticity & failures (both are config knobs on ONE env)

- **Noise (`noise_std`):** actual duration = nominal * (1 + ε), ε sampled at task completion (reveal-at-completion: the agent plans on nominal values, reality differs — this is what makes robustness meaningful).
- **Failures (`failure_rate`):** per-unit-simulated-time hazard. On failure at t_f, the running task is lost and requeued into the ready set, the node is marked `alive=False`. SystemMonitor fires; the next decision point simply reflects the new state — no special intervention machinery.
- **Determinism** = `noise_std=0, failure_rate=0`. Same code, knobs at zero. Validate the engine in this mode first (hand-checkable), then turn the knobs on.

### 5.3 Gymnasium-style interface

`reset()`, `step(action) -> (obs, reward, done, info)`. `step` applies the action, samples the duration, schedules the finish event, fast-forwards to the next decision point (failures may fire in between), and returns the new observation + per-step reward.

## 6. The RL agent (the core contribution)

### 6.1 Encoder — GraphSAGE

GraphSAGE (torch_geometric) over the DAG, bidirectional (forward + reverse edges, direction flagged) so each task aggregates from both predecessors (readiness/incoming data) and successors (downstream criticality / b-level). 2–3 layers (= hops). Produces per-task embeddings `h_i` and a pooled graph embedding `g`. Node features go through a small MLP → `n_j`, pooled → `c`.

### 6.2 Observation (per decision point)

- **Per-task features:** normalised `base_cost`; cost vector across the 4 node types; `mem_required`; status one-hot {done, ready, blocked} + scheduled flag; #unscheduled predecessors; `b_level`, `t_level` (critical-path signal — the key feature that lets the ordering head learn/beat HEFT's upward-rank); out-degree; output-data volume.
- **Per-node features:** type one-hot; `free_at_time` relative to current sim time; current utilisation; `power_w`; speed coef; `alive` flag.
- **Globals:** `[g, c, current_makespan, fraction_done]`.

Normalisation is mandatory (costs by mean cost, times by critical-path estimate) — keeps features stable and prevents reward-scale blow-ups.

### 6.3 Policy — two-headed autoregressive (MAIN FEATURE)

Factorised: `π(a) = π(task | s) · π(node | s, task)`.

- **Head 1 (ordering):** pointer/score over the ready task set — `score_i = MLP([h_i, globals])`, mask non-ready, softmax → τ. The agent owns task ordering — this is the differentiator vs assignment-only RL schedulers and the place it can outperform HEFT's hard-coded ranking.
- **Head 2 (assignment):** pointer/score over alive nodes — `score_j = MLP([h_τ, n_j, globals])`, mask dead/unavailable, softmax → ν.

Variable candidate-set sizes handled by score + mask + softmax (NOT fixed `Discrete`). This is the reason PPO (policy-gradient) is used and DQN is not.

**Critic:** `V = MLP([g, c, globals])` scalar.

Joint log-prob for PPO = `logπ(τ) + logπ(ν)`.

### 6.4 Reward — hybrid telescoping multi-criteria

```
per step:   r_t  = − w1 · (Δmakespan_t / M_ref) − w2 · (Δenergy_t / E_ref)
terminal:   r_T += + w3 · balance_index            # balance is global → terminal only
```

`Δmakespan_t`, `Δenergy_t` = increments caused by the current assignment. With γ=1 they telescope to total makespan / total energy → the dense per-step signal is mathematically the same objective as the totals, just distributed (faster, more stable learning).

`M_ref`, `E_ref` = per-instance references (critical-path / HEFT estimate) for normalisation so the weighted terms are comparable.

Weights `w1, w2, w3` live in `config.yaml` (defaults to be tuned in early runs, e.g. 1.0 / 0.3 / 0.2). No deadline term, no w4.

### 6.5 PPO trainer

Rollout buffer, GAE (λ≈0.95), clip loss (ε≈0.2), value loss, entropy bonus, advantage normalisation, Adam. Standard hyperparams; no DQN, no PER, no target network.

## 7. DAG generation (DAGFactory, Factory pattern)

Two interchangeable sources behind one interface:

- **Synthetic (`generate_synthetic`)** — networkx-based random valid DAGs; params: #tasks, CCR, density/shape, β heterogeneity. Built first; full control.
- **WfCommons adapter (`load_from_wfcommons`)** — `pip install wfcommons`; realistic workflows from real recipes (Montage, CyberShake, Blast, Seismology, Cycles…) via `Recipe.from_num_tasks(N).build_workflow()` → WfFormat JSON. The adapter parses WfFormat JSON → TaskDAG. Adds benchmark credibility. Added after the synthetic path works.

## 8. Baselines & fairness invariant

**Strategies:**

- **Time-only canonical heuristics — HEFT, CPOP, Min-Min** (faithful, unit-tested). Represent standard practice: they ignore energy and load balance *by design* — that is the point, not a defect. They are the fair competitors for the robustness leg (noise / failures), where the axis is makespan, not energy.
- **Multi-criteria control — Weighted-Sum Greedy.** At every decision point, over all (ready task, alive node) pairs, picks the argmin of the SAME per-step objective as the agent's reward: `w1·Δmakespan/M_ref + w2·Δenergy/E_ref` (same weights, same per-instance references). Balance is terminal-only, so the greedy does not chase it — balance is measured, not optimised (mirrors the reward, §6.4; the learned agent can anticipate the terminal balance term via γ=1 credit assignment, the myopic greedy cannot — that asymmetry is a result to show, not unfairness). This is the scientific control that isolates **"RL wins by learning / sequential planning"** from **"RL wins only because heuristics ignore energy."** If RL beats this on the identical objective, the multi-criteria claim is non-tautological.
- **Random floor** — sanity baseline.

What each comparison isolates: RL vs time-only heuristics → cost of ignoring energy/balance (objective-level win); RL vs Weighted-Sum Greedy → value of learning + sequential planning on the *same* objective (algorithm-level win); RL vs all under noise/failures → robustness / adaptivity.

They are reactive, not static: implemented as `Strategy.predict(ready, cluster_state)` called at every decision point on the live state, so they automatically re-assign onto surviving nodes after a failure. HEFT's upward-rank (structural) may be precomputed; EFT assignment uses live node state. (Optionally also keep a static HEFT as an extra point to show the cost of non-adaptivity.)

**Ordering fairness (defense note):** all baselines are full list-schedulers — they select BOTH task ordering and node (HEFT/CPOP via rank-based priority, Min-Min via the min-min rule, Weighted-Sum Greedy via argmin over (task, node) pairs), not assignment-only. The agent's edge is a *learned* ordering policy, not extra authority in the action space; the `predict(ready, cluster_state) -> (task, node)` interface is identical for all strategies. Each heuristic is run faithfully as itself — "upgrading" a heuristic to match the agent's capabilities would destroy the canonical baseline and is explicitly not done.

**Fairness invariant:** all strategies run on the same DAG instances, same noise seed, same failure events, same Observer trigger. Breaking this invalidates the thesis — guard it in tests.

## 9. Training

- **Scale (trainability lever):** train on 20–60 tasks, 4–16 nodes. Production-scale (1000s of tasks) is eval-only/out of scope.
- **Curriculum (optional):** ramp the same env's knobs across phases — clean → +noise → +failures. Can be skipped (train directly on the full env); it only aids convergence.
- **Budget + early stop (anti-infinite-training):** fixed env-step/epoch budget; every K updates run a frozen-policy eval on a fixed validation DAG set vs HEFT; stop on plateau or budget.
- **Sanity gates:** beats Random fast; reward rises then plateaus; eval-vs-HEFT improves. Flat reward from start ⇒ bug (reward scaling / masking / advantage), not "needs more training."
- **Checkpoints** saved to `models/*.pth` (also to Drive/Kaggle dataset when on cloud).

## 10. Evaluation methodology

- **Regime grid:** noise ∈ {0, 0.1, 0.2} × β ∈ {2, 5, 10} × failures ∈ {off, on}.
- **Per config:** 30–50 held-out DAGs; 5–10 noise seeds per DAG (→ robustness = std of makespan).
- **Training seeds:** 3–5 independent agents; report mean ± std across seeds.
- **Significance:** Wilcoxon signed-rank (paired over common instances) for "RL significantly better."
- **Metrics:** makespan C_max, total energy, resource utilisation, load-balance index, SLR, speedup, scheduling overhead (inference time), robustness (makespan std under noise).
- **Output:** console comparison table + `results/*.csv`; optional TensorBoard curves.

## 11. Tech stack & runtime

- Python 3.10+, PyTorch, PyTorch Geometric (PyG) — not DGL, networkx, gymnasium, wfcommons, numpy, pandas, pytest.
- Lint/format: ruff + black. Typing: full type hints.
- Dev: locally on CPU (no ROCm needed; AMD GPU irrelevant for dev).
- Training: Kaggle Notebooks (30 h/week) or Colab free (T4) — the workload is light (small GNN, CPU-bound simulator), comfortably inside free tiers. Save checkpoints externally (sessions wipe local disk).

## 12. Module / file structure

```
drl-scheduler/
  drl_scheduler.py          # CLI entry point (train / eval)
  config.yaml               # all hyperparams + experiment config
  src/
    core/        task.py  dag.py  compute_node.py  schedule.py
    env/         cluster_env.py  cost_model.py  events.py        # events.py: queue, failures, SystemMonitor (Observer)
    dag_factory/ factory.py  synthetic.py  wfcommons_adapter.py
    strategies/  base.py  rl_agent.py  heft.py  cpop.py  min_min.py  weighted_sum_greedy.py  random_strategy.py
    rl/          gnn_encoder.py  policy.py  ppo_trainer.py  rollout_buffer.py
    scheduler/   task_scheduler.py                              # decision-point loop, Observer subscriber
    eval/        metrics.py  evaluate.py
    utils/       seeding.py  normalization.py
  tests/         test_dag.py  test_cost_model.py  test_heft.py  test_cpop.py  test_min_min.py
                 test_reward_telescoping.py  test_env_step.py  test_masking.py  test_fairness.py
  dag_benchmarks/   results/   models/
```

## 13. Non-goals (explicit — do NOT implement)

- ❌ Task deadlines (cut entirely — no field, no reward term, no w4).
- ❌ GUI / dashboard / Streamlit / web / REST / gRPC.
- ❌ Database, Kubernetes/Mesos, MLflow, real-cluster execution, human-in-the-loop approval.
- ❌ DQN / PER / target networks (PPO Actor-Critic only).
- ❌ DGL (use PyG).
- ❌ Fixed `Discrete` action space (use pointer scoring + masking).
- ❌ Anything that breaks the fairness invariant (§8).

## 14. Scientific framing (keep in conclusions)

RL's advantage is multi-criteria (HEFT/CPOP optimise time only) and adaptive/robust under stochasticity + failures. The level of dynamism reached = the strength of the claim: +noise ⇒ "robust to uncertainty"; +failures ⇒ "adapts to failures."

Claims like "10–25% makespan improvement over HEFT" must be qualified as **under dynamic conditions**, not on static instances (where reactive HEFT is near-optimal).

---

## Appendix A — M1 architectural refinements & locked invariants (added 2026-06-18, rebuild)

These sharpen §4–§6 for the M1 (core + deterministic simulator) build and are binding on M2–M5.

### A.1 Single source of truth for the objective (`env/placement.py`)
All per-step cost decomposition lives in **one** side-effect-free function, `placement.weighted_cost(task, node, cluster_state)`, which evaluates a candidate assignment against the current state **without mutating it** and returns the *individual normalised components* `Δmakespan/M_ref` and `Δenergy/E_ref` (not a pre-summed scalar).
- `cluster_env.step` negates and weight-sums these components → per-step reward `−w1·(Δmakespan/M_ref) − w2·(Δenergy/E_ref)`.
- The M2 Weighted-Sum Greedy minimises `w1·(Δmakespan/M_ref) + w2·(Δenergy/E_ref)` over candidate `(task, node)` pairs via the **same** function — mathematical parity with the reward is structural, not coincidental (§8 fairness invariant).
- The terminal `+w3·balance` term is **NOT** part of `weighted_cost` (the greedy measures balance, never chases it); the env adds it once at episode end.
- Benefit: the objective function gets real coverage via `test_reward_telescoping` in M1, instead of being dead code until M2.

### A.2 Locked Δ and per-instance reference definitions
- **Δmakespan_t** = `makespan_after − makespan_before`, where `makespan = max(free_at_time)` across all (alive) nodes — the running schedule horizon. It is **NOT** the finish time of the just-scheduled task. If the task fits entirely under the current horizon, `Δmakespan_t = 0`. (Horizon is monotone non-decreasing ⇒ `Σ Δmakespan = final makespan`; telescopes exactly under γ=1.)
- **M_ref** = critical-path lower bound (fastest-exec critical path of the instance).
- **E_ref** = absolute energy lower bound = `Σ_i min_node(energy_{i,node})` (per-task minimum energy summed; energy has no structural critical path).
- Both refs are computed once at `reset()` and cached for the episode.

### A.3 Configuration & observation safeguards
- **Synthetic `base_cost` distribution (`dag_factory/synthetic.py`):** wide / heavy-tailed (log-normal or wide uniform). Uniform task sizes flatten the multi-criteria trade-off and make queue ordering irrelevant — forbidden.
- **Load-balance CV scope:** the `1 − CV` index is evaluated over **all alive nodes**; nodes left idle by small DAGs count as 0 busy time, so structural imbalance is correctly reflected.
- **Reference metadata placement:** static per-instance constants (`M_ref`, `E_ref`, static config flags) are emitted in the Gymnasium **`info` dict, not the `obs`**. `obs` stays lean — dynamic per-task / per-node / global features only (for the GNN policy). *(This overrides any earlier note that the Observation object carries `m_ref`/`e_ref`.)*

### A.4 M1 gating-test enhancements
- **Observation consistency:** `test_env_step` asserts cross-step state transitions — a task's status bit flips `ready → done`, and the `#unscheduled_predecessors` counter decrements for child tasks when a parent completes.
- **Golden schedule fixture:** a tiny hardcoded 3–4 task DAG with a known-optimal deterministic schedule is committed to the suite as a permanent regression anchor for M2–M5.