# WfCommons handling — WfFormat → TaskDAG (design)

**Date:** 2026-06-20
**Milestone:** M5 (WfCommons leg; depends on nothing earlier — TZ §7: "after the synthetic path works")
**Status:** approved design, ready for implementation plan

## 1. Decision & rationale

**Option 1: a fixture-tested parser plus an optional generator.** The load-bearing
artifact is the **parser** (WfFormat JSON → `TaskDAG`); it must be deterministic and
hand-checkable, exactly like the golden-schedule fixture. The `wfcommons` library is only
a one-off *producer* of WfFormat JSON.

Rejected alternatives:
- **Run `Recipe.build_workflow()` live in the test suite.** Pulls a heavy dependency into
  the default suite and runs random (possibly network / recipe-data) generation live,
  violating the determinism/reproducibility invariant (.cursorrules §6). Its "most
  faithful" framing is illusory: faithfulness comes from running on real
  wfcommons-generated workflows, which we get by **committing them as data**, not by
  regenerating them in tests.
- **Drop WfCommons entirely.** Needlessly discards a cheap thesis-credibility leg
  (real Montage/CyberShake/Blast workflows).

## 2. Parser scope — modeling decisions (disclose in thesis, not plumbing)

Resolved against **WfFormat v1.5** (`schemaVersion: "1.5"`), confirmed from the WfFormat
schema. The format splits `workflow.specification` (logical structure) from
`workflow.execution` (timings/resources), joined by task `id`.

Relevant fields:
- `specification.tasks[]`: `id` (str), `name` (str), `parents[]`, `children[]`,
  `inputFiles[]`, `outputFiles[]`.
- `specification.files[]`: `id` (str), `sizeInBytes` (int).
- `execution.tasks[]`: `id` (str), `runtimeInSeconds` (num, **required**),
  `memoryInBytes` (optional), `coreCount`, energy fields, etc.
- **No `category` field exists on tasks** — `name` is the only class signal.

### 2.1 id → integer index
WfFormat ids are strings; `TaskDAG` requires `node_id == index` over `0..N-1`. Assign
indices deterministically: **topological order, ties broken by string id ascending**.
Keep a `str → int` map for edge wiring. (Topological-first keeps indices roughly aligned
with dependency depth, which is convenient for debugging and stable across regenerations.)

### 2.2 runtime → base_cost
`base_cost ← execution.runtimeInSeconds`, taken **verbatim** as the nominal cost on a
reference unit. The parser reads **no** machine speed — heterogeneity comes from the
existing speed table (`base_cost / speed[type][class]`, `cost_model.exec_time`). Do not
bake machine-specific timings into the cost matrix.

Fail loud if a `specification` task has no matching `execution` task (Recipe always emits
both; a missing join is corrupt input, not a default-able condition).

### 2.3 name → task_class (curated per-recipe tables + hash fallback)
WfFormat has no `{data_parallel, sequential, streaming}`. Assign deterministically:

1. Extract the **abstract task name** by stripping the numeric/ID suffix
   (e.g. `mProject_00000007 → mProject`, `individuals_ID0001 → individuals`). The
   stripping rule is a single documented regex.
2. Look up the abstract name in a **curated per-recipe table**
   (`MONTAGE`, `CYBERSHAKE`, `BLAST`, …) mapping abstract name → `TaskClass`. The recipe
   is selected by the `recipe` argument, else inferred from the dominant abstract-name
   prefix across the workflow.
3. **Fallback:** any abstract name not in the selected table maps to
   `CLASSES[stable_hash(abstract_name) % 3]` (a fixed deterministic hash, not Python's
   salted `hash()`). This guarantees: no `KeyError` on unknown recipes/versions, no silent
   collapse of every task into one class, and full reproducibility.

Curated tables are authored so a benchmark spans **at least two** classes — if a recipe
collapses to one class, the realistic benchmarks lose the class-heterogeneity that makes
node selection non-trivial. Both the tables and the fallback are disclosed in the thesis.

**Maintenance/drift note:** curated tables are the brittle part; the
`@pytest.mark.wfcommons` integration test (§5) plus the hash fallback bound the blast
radius of wfcommons recipe drift.

### 2.4 memoryInBytes → mem_required
`mem_required ← rescale(execution.memoryInBytes)` into the synthetic generator's range
(~1–8 units) when the field is present, else a pinned `DEFAULT_MEM` constant. The rescale
rule (a pinned linear `bytes → unit` map) lives in config/constants, not inline.

### 2.5 files → edge volumes
Precedence is **explicit** in the schema, so do not reverse-engineer edges from files:

- **Edge set** = the `parents`/`children` relation (authoritative DAG structure).
- **Volume** `d_ij = (Σ sizeInBytes(f) for f in outputFiles(i) ∩ inputFiles(j)) * BYTES_TO_UNIT`.
- Precedence edges with no shared file get `d_ij = EPS` (a small pinned constant), so the
  control dependency survives in the graph.

`BYTES_TO_UNIT` is a single **pinned config constant** that makes byte-scale volumes
commensurate with second-scale `base_cost` in the cost model
(`comm_time = volume/bandwidth`). It **preserves each benchmark's native comm/comp ratio**
(Montage stays comm-light, CyberShake comm-heavy) rather than re-normalizing every DAG to
a fixed synthetic CCR — preserving the realism that justifies using WfCommons at all. The
constant and its derivation are documented in the thesis.

### 2.6 Invariants of the produced TaskDAG
The result re-validates the **same invariants as synthetic DAGs**: acyclic, `node_id ==
index` over `0..N-1`, full per-task fields (`base_cost`, `mem_required`, `task_class`),
and per-edge volumes. Enforced by `TaskDAG.__init__` plus parser-level assertions.

## 3. Module layout

```
src/dag_factory/
  wfcommons_adapter.py   # PURE parser: parse_wfformat(doc: dict, rng, recipe=None) -> TaskDAG
                         #   no file I/O, no `import wfcommons`
  wfcommons_classes.py   # curated per-recipe name->TaskClass tables, abstract-name regex,
                         #   stable_hash fallback; pinned DEFAULT_MEM / EPS / BYTES_TO_UNIT refs
  factory.py             # + DAGFactory.load_from_wfcommons(path, rng, recipe=None)
                         # + create(source="wfcommons", ...) delegates to it
tools/
  gen_wfcommons.py       # OPTIONAL CLI: lazy `import wfcommons`; Recipe.from_num_tasks(N)
                         #   .build_workflow() -> WfFormat JSON -> dag_benchmarks/
dag_benchmarks/
  README.md              # exact generation command + pinned seed (frozen, regenerable)
  *.json                 # committed realistic set (Montage/CyberShake/Blast @ 20-60 tasks)
tests/fixtures/
  wfformat_tiny.json     # tiny hand-built WfFormat v1.5 doc for test_wfcommons
tests/
  test_wfcommons.py      # parser correctness; no network, no wfcommons import
  test_wfcommons_live.py # @pytest.mark.wfcommons integration test, deselected by default
```

Pinned numeric constants (`DEFAULT_MEM`, `EPS`, `BYTES_TO_UNIT`, mem rescale bounds) go in
`config.yaml` under a `wfcommons:` block — no magic numbers in code (.cursorrules).

## 4. Two committed-data roles (kept separate)

**(a) Parser fixture** — `tests/fixtures/wfformat_tiny.json`: a tiny hand-authored
WfFormat v1.5 document (~4 tasks, e.g. a diamond `0 -> {1,2} -> 3`) with explicit
`specification` + `execution` and a couple of files, sized so every mapped field is
hand-verifiable. Exercises parser correctness only. **No network, no wfcommons import.**
This is the WfCommons analog of `tests/conftest.py::golden_instance`.

**(b) Benchmark set** — `dag_benchmarks/*.json`: the frozen realistic workflows for the M5
eval grid. Produced **once** by `tools/gen_wfcommons.py` with a **pinned seed**; the exact
command and seed are recorded in `dag_benchmarks/README.md` so the set is frozen and
regenerable. The eval grid loads these via `DAGFactory.load_from_wfcommons` /
`create(source="wfcommons", ...)`.

## 5. Optional generation path & dependency hygiene

`wfcommons` is an **optional extra**: runtime and the default test suite must not import
it. `tools/gen_wfcommons.py` imports it **lazily** inside the function, guarded by
`try/except ImportError` with a clear `pip install wfcommons` message.

One `@pytest.mark.wfcommons` integration test (`test_wfcommons_live.py`) exercises live
`Recipe.from_num_tasks(N).build_workflow()` → parser, to catch wfcommons **API/schema
drift**. It is **deselected by default**:
- register the marker in `pyproject.toml` `[tool.pytest.ini_options].markers`;
- add `addopts = "-m 'not wfcommons'"` so the default `pytest` run skips it.
Run it manually (`pytest -m wfcommons`) only when regenerating the benchmark set.

This **refines** the TZ §11 / .cursorrules "core stack" listing — `wfcommons` is demoted
from core to an optional extra. Recorded here as a deliberate, defensible test-hygiene
deviation (the default suite stays light and offline; reproducibility invariant intact).

## 6. Routing & invariants

Single entry point preserves "DAGs only via `DAGFactory`" (.cursorrules):

```python
DAGFactory.load_from_wfcommons(path, rng, recipe=None)  # reads JSON, calls pure parser
DAGFactory.create(source="wfcommons", path=..., recipe=...)  # delegates to the above
```

The produced `TaskDAG` flows through the **identical** env / cost model / strategies /
fairness invariant as synthetic DAGs — **no special-casing** anywhere downstream. WfCommons
DAGs are just another `source` to the factory; everything past the factory is source-blind.
The fairness invariant (.cursorrules §1, `tests/test_fairness.py`) is unaffected: all
strategies still receive the same instances/seeds/events.

## 7. Testing strategy (TDD — failing test first)

- `test_wfcommons.py` (default suite):
  - parses `wfformat_tiny.json` into the expected `TaskDAG` — hand-checked node count,
    `node_id == index` ordering, `base_cost` (= runtime), `task_class` (curated + a
    fallback case), `mem_required` (derived + a default case), and edge volumes
    (shared-file sum, plus an `EPS` control-only edge);
  - asserts **no `wfcommons` import** is required (no import in the module under test) and
    no network access;
  - asserts the result satisfies the synthetic invariants (acyclic, full fields).
- `test_wfcommons_live.py` (`@pytest.mark.wfcommons`, deselected by default): live Recipe →
  parser smoke check for API drift.
- ruff + black clean; full type hints; no untyped public functions.

## 8. Exit criteria

1. `test_wfcommons` parses the committed fixture into the expected `TaskDAG` (hand-checked
   structure, costs, mapped fields) with **no network and no wfcommons import**.
2. The eval grid can load the committed `dag_benchmarks/` set via the factory.
3. The optional generation path is **documented and seed-pinned**
   (`dag_benchmarks/README.md`), and `@pytest.mark.wfcommons` is deselected by default.
4. The dependency-hygiene deviation from TZ §11 (wfcommons = optional extra) is recorded.

## 9. Out of scope

- The rest of M5 (metrics, `eval/evaluate.py` regime grid, `drl_scheduler.py` CLI) — this
  spec covers only the WfCommons DAG source they consume.
- Reading machine-specific timings / energy from `execution` into the cost matrix
  (explicitly rejected — see §2.2).
- Any change to env / strategies / fairness paths.
