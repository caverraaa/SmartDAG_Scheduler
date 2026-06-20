# CLAUDE.md — SmartDAG Scheduler
 
> Project context for Claude Code. This file documents **project-specific conventions and invariants** — the things you cannot infer from the codebase alone. The full design spec lives in `SmartDAG_Scheduler_TZ.md`; read it before non-trivial work.
 
## Interplay with Superpowers
 
The Superpowers plugin injects its own `brainstorm → plan → implement` workflow, TDD skills, and review skills at session start via the bootstrap hook. **Do not re-document or re-implement that workflow here.** This file supplies the project context those skills consult.
 
**Priority:** project-specific instructions in this file and in `SmartDAG_Scheduler_TZ.md` override generic Superpowers skill defaults. When a skill's generic suggestion conflicts with a rule or invariant below, the rule below wins. Surface the conflict instead of silently following the generic default.
 
## What this project is (anti-drift anchor)
 
A CLI-only, research-grade ПМК that trains a Deep RL DAG-scheduling agent on simulated heterogeneous clusters, benchmarks it against classical heuristics on identical instances, and emits metrics for a thesis.
 
**The scientific claim — keep it intact in every decision:** RL does *not* win everything. It wins (a) structurally on multi-criteria objectives (heuristics ignore energy + load balance) and (b) empirically on robustness under execution-time noise and node failures. If a change would weaken, bypass, or pre-bias this claim, stop and flag it.
 
## Tech stack & Context7
 
Python 3.10+. Core: **PyTorch**, **PyTorch Geometric (PyG — not DGL)**, **NetworkX**, **Gymnasium**, **WfCommons**, **NumPy**, **pandas**. Test: **pytest**. Lint/format: **ruff** + **black**. Full type hints everywhere.
 
**Pulling docs via Context7 — do this whenever you touch a library API, before writing the code:** call `resolve-library-id` for the library name, then `get-library-docs` with a focused topic. Always resolve first (IDs can change); the canonical IDs are:
 
| Library | Context7 ID | Typical topics to request |
|---|---|---|
| PyTorch | `/pytorch/pytorch` | autograd, nn modules, optim, save/load |
| PyTorch Geometric | `/pyg-team/pytorch_geometric` | SAGEConv, message passing, Data/Batch, pooling |
| Gymnasium | `/farama-foundation/gymnasium` | Env API, reset/step contract, spaces, wrappers |
| NetworkX | `/networkx/networkx` | DiGraph, DAG checks, longest_path, topological_sort |
| WfCommons | `/wfcommons/wfcommons` | WfFormat schema, Recipe, workflow generation |
| NumPy | `/numpy/numpy` | random Generator, vectorization |
| pandas | `/pandas-dev/pandas` | DataFrame I/O, groupby, to_csv |
| pytest | `/pytest-dev/pytest` | fixtures, parametrize, monkeypatch |
 
Pin doc lookups to the version actually installed (check `pyproject.toml` / lockfile) rather than trusting memory — several of these (PyG, Gymnasium) have breaking API shifts across minor versions.
 
## Hard invariants — NEVER violate (these are the thesis, not style)
 
1. **Fairness invariant.** All strategies (RL + every heuristic) run on the *same* DAG instances, *same* noise seed, *same* failure events, *same* Observer trigger, *same* `predict(ready, cluster_state) -> (task, node)` interface. Any code path that gives one strategy different instances/seeds/events/information than another is a bug, no matter how it improves results. Guard it in `tests/test_fairness.py`.
2. **Baselines stay authentic.** HEFT, CPOP, Min-Min are run faithfully as the published algorithms. Do **not** "upgrade" a heuristic to match the agent's capabilities (no learned ordering bolted onto HEFT, etc.) — that destroys the canonical baseline. All baselines are full list-schedulers (they choose task ordering *and* node); the agent's only edge is a *learned* ordering policy, not extra action-space authority.
3. **Multi-criteria control = Weighted-Sum Greedy.** It optimizes the SAME per-step objective as the reward (`w1·Δmakespan/M_ref + w2·Δenergy/E_ref`, identical weights and per-instance references). Balance is terminal-only; greedy measures it, does not chase it. Do not let it drift to a different objective than the agent's reward.
4. **γ = 1.0 and reward telescoping.** Per-step reward must telescope exactly to total makespan / total energy under γ=1; the terminal balance term is added once. Do not introduce discounting or per-step balance terms — it breaks the equivalence between the dense signal and the true objective.
5. **Reveal-at-completion noise.** The agent plans on nominal durations; the noise ε is sampled and revealed only at task completion. Never leak realized durations into the observation — that is what makes the robustness result meaningful.
6. **Determinism = knobs at zero.** `noise_std=0, failure_rate=0` must be the *same* code path with knobs off, and must produce hand-checkable, reproducible schedules. Seed everything through `utils/seeding.py`.
## Conventions
 
- **File structure:** as defined in §12 of the spec. New modules go in the matching `src/<layer>/` directory; respect the layering (core → env → strategies → rl → scheduler → eval). Do not create cross-layer back-dependencies (e.g. `core` importing from `rl`).
- **Strategies** implement `BaseSchedulingStrategy` (`src/strategies/base.py`). Every new strategy is a drop-in behind `predict(ready, cluster_state) -> (task, node)`.
- **DAG sources** go behind `DAGFactory`; never construct DAGs ad hoc in training/eval code.
- **Config:** all hyperparams and experiment settings live in `config.yaml`. No magic numbers in code — read from config.
- **Testing (TDD via Superpowers):** new behavior gets a failing test first. Required coverage that must stay green: faithful heuristics (`test_heft`, `test_cpop`, `test_min_min`), `test_reward_telescoping`, `test_env_step`, `test_masking`, `test_fairness`. A flat reward curve from step 0 means a bug (reward scaling / masking / advantage) — investigate, do not "train longer."
- **Style:** `ruff` + `black` clean before any review step. Full type hints; no untyped public functions.
## Non-goals — do NOT implement (will be rejected in review)
 
GUI / dashboard / Streamlit / web / REST / gRPC; database; Kubernetes / Mesos / MLflow; real-cluster execution; human-in-the-loop approval; **task deadlines** (no field, no reward term, no `w4`); DQN / PER / target networks (PPO Actor-Critic only); DGL (use PyG); fixed `Discrete` action space (use pointer scoring + masking); anything that breaks the fairness invariant.
 
## When in doubt
 
Prefer the smallest change that satisfies the spec. If a request pushes toward the non-goals, toward capability-parity hacks on baselines, or toward weakening an invariant — pause and raise it rather than implementing. Over-engineering is a failure mode here, not a virtue.
