# WfCommons Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, fixture-tested parser that turns WfFormat v1.5 JSON into a `TaskDAG`, plus an optional (lazily-imported) generator CLI for producing a frozen realistic benchmark set.

**Architecture:** A pure parser (`parse_wfformat(doc, rng, params, recipe=None) -> TaskDAG`) does the semantic mapping with no I/O and no `wfcommons` import. Modeling constants live in `config.yaml` under a `wfcommons:` block, loaded by a small dedicated `WfcommonsParams` loader and passed into the parser. `DAGFactory` is the only entry point (`load_from_wfcommons` + `create(source="wfcommons", ...)`). The `wfcommons` library is an optional extra used only by a generation CLI and one default-deselected integration test.

**Tech Stack:** Python 3.10+, NetworkX, NumPy, PyYAML, pytest. `wfcommons` is an OPTIONAL extra (never imported by runtime or the default test suite).

## Global Constraints

- Python >= 3.10; full type hints on all public functions (.cursorrules).
- ruff + black clean (`line-length = 100`); ruff lint select `E,F,I,UP,B`.
- Run tools via `.venv/bin/` (bare `python`/`pytest`/`ruff`/`black` are not on PATH; PEP 668).
- TDD: write the failing test first, watch it fail, then implement.
- No magic numbers in code — modeling constants live in `config.yaml` (.cursorrules).
- Respect layering: `core -> dag_factory`. No back-dependencies.
- DAGs only via `DAGFactory` — never construct ad hoc in training/eval.
- `TaskDAG` invariants (enforced by `TaskDAG.__init__`): `node_id == index` over `0..N-1`, acyclic, per-edge `data` volume.
- `Task` fields are exactly `id: int, base_cost: float, mem_required: float, task_class: TaskClass`.
- `TaskClass` members: `DATA_PARALLEL`, `SEQUENTIAL`, `STREAMING`. `list(TaskClass)` order = `[DATA_PARALLEL, SEQUENTIAL, STREAMING]`.
- The parser module must NOT `import wfcommons` and must NOT do file/network I/O.
- WfFormat target version: **v1.5** (`schemaVersion: "1.5"`). Structure in `workflow.specification`, timings in `workflow.execution`, joined by task `id`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/dag_factory/wfcommons_config.py` (create) | `WfcommonsParams` frozen dataclass + `load_wfcommons_params(path)` reading the `wfcommons:` block. |
| `config.yaml` (modify) | Add the `wfcommons:` constants block. |
| `src/dag_factory/wfcommons_classes.py` (create) | Abstract-name regex, `stable_hash`, curated per-recipe `name -> TaskClass` tables, `infer_recipe`, `assign_task_class`. |
| `src/dag_factory/wfcommons_adapter.py` (create) | Pure `parse_wfformat(doc, rng, params, recipe=None) -> TaskDAG`. |
| `src/dag_factory/factory.py` (modify) | `DAGFactory.load_from_wfcommons(...)` + `create(source="wfcommons", ...)`. |
| `tests/fixtures/wfformat_tiny.json` (create) | Hand-authored 4-task WfFormat v1.5 doc for parser correctness. |
| `tests/test_wfcommons_config.py` (create) | Loader test. |
| `tests/test_wfcommons_classes.py` (create) | Class-assignment tests. |
| `tests/test_wfcommons.py` (create) | Parser correctness against the fixture (no network, no wfcommons). |
| `tests/test_wfcommons_factory.py` (create) | Factory routing test. |
| `tools/gen_wfcommons.py` (create) | Optional CLI: lazy `import wfcommons`; WfFormat JSON -> `dag_benchmarks/`. |
| `tests/test_wfcommons_live.py` (create) | `@pytest.mark.wfcommons` integration test, deselected by default. |
| `pyproject.toml` (modify) | Register `wfcommons` marker + `addopts = "-m 'not wfcommons'"`. |
| `dag_benchmarks/README.md` (create) | Exact generation command + pinned seed. |

---

## Task 1: WfcommonsParams config

**Files:**
- Create: `src/dag_factory/wfcommons_config.py`
- Modify: `config.yaml`
- Test: `tests/test_wfcommons_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `WfcommonsParams` — frozen dataclass with fields:
    `default_mem: float, eps: float, bytes_to_unit: float, mem_min: float, mem_max: float, memory_ref_bytes: float`.
  - `load_wfcommons_params(path: str = "config.yaml") -> WfcommonsParams`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wfcommons_config.py
from src.dag_factory.wfcommons_config import WfcommonsParams, load_wfcommons_params


def test_load_wfcommons_params_from_config() -> None:
    p = load_wfcommons_params("config.yaml")
    assert isinstance(p, WfcommonsParams)
    assert p.mem_min < p.mem_max
    assert p.eps > 0.0
    assert p.bytes_to_unit > 0.0
    assert p.memory_ref_bytes > 0.0


def test_wfcommons_params_is_frozen() -> None:
    p = WfcommonsParams(
        default_mem=4.0, eps=0.01, bytes_to_unit=1e-6,
        mem_min=1.0, mem_max=8.0, memory_ref_bytes=8e9,
    )
    import dataclasses
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.eps = 0.5  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_wfcommons_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dag_factory.wfcommons_config'`

- [ ] **Step 3: Add the config block**

Append to `config.yaml`:

```yaml

# WfCommons parser modeling constants (M5). Disclosed in thesis.
wfcommons:
  default_mem: 4.0          # mem_required when execution.memoryInBytes is absent
  eps: 0.01                 # volume on a precedence edge that shares no file
  bytes_to_unit: 1.0e-6     # file sizeInBytes -> cost-unit volume (preserves native CCR)
  mem_min: 1.0              # rescale floor for mem_required (matches synthetic ~1-8)
  mem_max: 8.0              # rescale ceiling for mem_required
  memory_ref_bytes: 8.0e9   # memoryInBytes that maps to mem_max
```

- [ ] **Step 4: Write minimal implementation**

```python
# src/dag_factory/wfcommons_config.py
"""Modeling constants for the WfCommons parser, loaded from config.yaml (TZ §5, §7)."""

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class WfcommonsParams:
    default_mem: float
    eps: float
    bytes_to_unit: float
    mem_min: float
    mem_max: float
    memory_ref_bytes: float


def load_wfcommons_params(path: str = "config.yaml") -> WfcommonsParams:
    """Parse the ``wfcommons:`` block of config.yaml into a typed, frozen params object."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    wf = raw["wfcommons"]
    return WfcommonsParams(
        default_mem=float(wf["default_mem"]),
        eps=float(wf["eps"]),
        bytes_to_unit=float(wf["bytes_to_unit"]),
        mem_min=float(wf["mem_min"]),
        mem_max=float(wf["mem_max"]),
        memory_ref_bytes=float(wf["memory_ref_bytes"]),
    )
```

- [ ] **Step 5: Run tests + lint**

Run: `.venv/bin/pytest tests/test_wfcommons_config.py -v && .venv/bin/ruff check src/dag_factory/wfcommons_config.py tests/test_wfcommons_config.py && .venv/bin/black --check src/dag_factory/wfcommons_config.py tests/test_wfcommons_config.py`
Expected: PASS, ruff clean, black clean.

- [ ] **Step 6: Commit**

```bash
git add src/dag_factory/wfcommons_config.py config.yaml tests/test_wfcommons_config.py
git commit -m "feat: WfcommonsParams config loader for parser constants"
```

---

## Task 2: task_class assignment (curated tables + hash fallback)

**Files:**
- Create: `src/dag_factory/wfcommons_classes.py`
- Test: `tests/test_wfcommons_classes.py`

**Interfaces:**
- Consumes: `src.core.task.TaskClass`.
- Produces:
  - `abstract_name(name: str) -> str` — strips trailing `_<digits>` / `_ID<digits>` groups.
  - `stable_hash(text: str) -> int` — deterministic (md5-based), not Python's salted `hash`.
  - `RECIPE_TABLES: dict[str, dict[str, TaskClass]]` — keys `"montage"`, `"cybershake"`, `"blast"`.
  - `infer_recipe(abstract_names: list[str]) -> str | None` — best-matching recipe key, else `None`.
  - `assign_task_class(name: str, table: dict[str, TaskClass]) -> TaskClass`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wfcommons_classes.py
from src.core.task import TaskClass
from src.dag_factory.wfcommons_classes import (
    RECIPE_TABLES,
    abstract_name,
    assign_task_class,
    infer_recipe,
    stable_hash,
)


def test_abstract_name_strips_numeric_and_id_suffixes() -> None:
    assert abstract_name("mProject_00000001") == "mProject"
    assert abstract_name("individuals_ID0000007_0") == "individuals"
    assert abstract_name("mConcatFit") == "mConcatFit"


def test_stable_hash_is_deterministic_and_nonnegative() -> None:
    assert stable_hash("mProject") == stable_hash("mProject")
    assert stable_hash("mProject") >= 0


def test_curated_lookup_uses_table() -> None:
    table = RECIPE_TABLES["montage"]
    assert assign_task_class("mProject_00000001", table) is TaskClass.DATA_PARALLEL
    assert assign_task_class("mConcatFit_00000001", table) is TaskClass.SEQUENTIAL


def test_unmatched_name_uses_deterministic_fallback() -> None:
    table = RECIPE_TABLES["montage"]
    classes = list(TaskClass)
    expected = classes[stable_hash("totallyUnknownTask") % 3]
    assert assign_task_class("totallyUnknownTask_42", table) is expected


def test_every_table_spans_at_least_two_classes() -> None:
    for key, table in RECIPE_TABLES.items():
        assert len(set(table.values())) >= 2, f"{key} collapses to one class"


def test_infer_recipe_picks_best_match() -> None:
    names = ["mProject", "mDiffFit", "mConcatFit", "unknown"]
    assert infer_recipe(names) == "montage"
    assert infer_recipe(["nothing", "matches"]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_wfcommons_classes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dag_factory.wfcommons_classes'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/dag_factory/wfcommons_classes.py
"""Deterministic WfFormat task-name -> TaskClass mapping (TZ §4, §7).

WfFormat has no task category; ``name`` is the only class signal. We use curated
per-recipe tables on the abstract task name, with a deterministic hash fallback so
unknown recipes/versions never KeyError or silently collapse to one class.
Both the tables and the fallback are disclosed in the thesis.
"""

import hashlib
import re

from src.core.task import TaskClass

_CLASSES = list(TaskClass)
_SUFFIX_RE = re.compile(r"(_(ID)?\d+)+$")


def abstract_name(name: str) -> str:
    """Strip trailing ``_<digits>`` / ``_ID<digits>`` groups (e.g. mProject_00000001)."""
    return _SUFFIX_RE.sub("", name)


def stable_hash(text: str) -> int:
    """Deterministic, cross-run hash (Python's built-in hash is salted)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


# Curated per-recipe tables. Montage names are confirmed; cybershake/blast are
# best-effort and verified/extended against generated JSON during benchmark
# regeneration (Task 6). Unknown names fall back deterministically, so an
# out-of-date table degrades gracefully rather than failing.
RECIPE_TABLES: dict[str, dict[str, TaskClass]] = {
    "montage": {
        "mProject": TaskClass.DATA_PARALLEL,
        "mDiffFit": TaskClass.STREAMING,
        "mConcatFit": TaskClass.SEQUENTIAL,
        "mBgModel": TaskClass.SEQUENTIAL,
        "mBackground": TaskClass.DATA_PARALLEL,
        "mImgtbl": TaskClass.SEQUENTIAL,
        "mAdd": TaskClass.DATA_PARALLEL,
        "mShrink": TaskClass.STREAMING,
        "mViewer": TaskClass.SEQUENTIAL,
    },
    "cybershake": {
        "PreCVM": TaskClass.SEQUENTIAL,
        "ExtractSGT": TaskClass.STREAMING,
        "SeismogramSynthesis": TaskClass.DATA_PARALLEL,
        "PeakValCalcOkaya": TaskClass.DATA_PARALLEL,
        "ZipSeis": TaskClass.SEQUENTIAL,
        "ZipPSA": TaskClass.SEQUENTIAL,
    },
    "blast": {
        "blastall": TaskClass.DATA_PARALLEL,
        "split": TaskClass.SEQUENTIAL,
        "cat": TaskClass.SEQUENTIAL,
    },
}


def infer_recipe(abstract_names: list[str]) -> str | None:
    """Return the recipe key whose table matches the most abstract names, or None."""
    best_key: str | None = None
    best_hits = 0
    for key, table in RECIPE_TABLES.items():
        hits = sum(1 for a in abstract_names if a in table)
        if hits > best_hits:
            best_key, best_hits = key, hits
    return best_key


def assign_task_class(name: str, table: dict[str, TaskClass]) -> TaskClass:
    """Curated lookup on the abstract name; deterministic hash fallback otherwise."""
    abstract = abstract_name(name)
    if abstract in table:
        return table[abstract]
    return _CLASSES[stable_hash(abstract) % 3]
```

- [ ] **Step 4: Run tests + lint**

Run: `.venv/bin/pytest tests/test_wfcommons_classes.py -v && .venv/bin/ruff check src/dag_factory/wfcommons_classes.py tests/test_wfcommons_classes.py && .venv/bin/black --check src/dag_factory/wfcommons_classes.py tests/test_wfcommons_classes.py`
Expected: PASS, ruff clean, black clean.

- [ ] **Step 5: Commit**

```bash
git add src/dag_factory/wfcommons_classes.py tests/test_wfcommons_classes.py
git commit -m "feat: curated per-recipe task_class tables with deterministic fallback"
```

---

## Task 3: Pure parser + tiny fixture

**Files:**
- Create: `tests/fixtures/wfformat_tiny.json`
- Create: `src/dag_factory/wfcommons_adapter.py`
- Test: `tests/test_wfcommons.py`

**Interfaces:**
- Consumes: `WfcommonsParams` (Task 1); `assign_task_class`, `infer_recipe`, `abstract_name`, `RECIPE_TABLES` (Task 2); `src.core.dag.TaskDAG`; `src.core.task.Task`.
- Produces: `parse_wfformat(doc: dict, rng: np.random.Generator, params: WfcommonsParams, recipe: str | None = None) -> TaskDAG`.

**Fixture design (hand-checked).** Diamond `task-1 -> {task-2, task-3} -> task-4`.
- Index order = deterministic topological order, ties by string id ascending:
  `task-1 -> 0`, `task-2 -> 1`, `task-3 -> 2`, `task-4 -> 3`.
- `base_cost` = `runtimeInSeconds`: `[5, 3, 4, 2]` by index.
- Names: `mProject` (DATA_PARALLEL), `mDiffFit` (STREAMING) ×2, `mConcatFit` (SEQUENTIAL).
- Files: `f_ra=100B` (task-1→task-2), `f_as=200B` (task-2→task-4), `f_bs=50B` (task-3→task-4).
  `task-1 -> task-3` shares NO file -> EPS edge.
- With test params `bytes_to_unit=1.0`, `eps=0.01`: edge volumes
  `(0,1)=100`, `(0,2)=0.01`, `(1,3)=200`, `(2,3)=50`.
- `mem_required` with `mem_min=1, mem_max=8, memory_ref_bytes=8e9, default_mem=4`:
  task-1 `memoryInBytes=8e9 -> 8.0`; task-2 (no field) `-> 4.0`; task-3 `4e9 -> 4.5`; task-4 `0 -> 1.0`.

- [ ] **Step 1: Write the fixture**

```json
{
  "name": "tiny-montage",
  "description": "Hand-authored WfFormat v1.5 fixture for parser tests.",
  "schemaVersion": "1.5",
  "workflow": {
    "specification": {
      "tasks": [
        {"id": "task-1", "name": "mProject_00000001", "parents": [], "children": ["task-2", "task-3"], "inputFiles": [], "outputFiles": ["f_ra"]},
        {"id": "task-2", "name": "mDiffFit_00000001", "parents": ["task-1"], "children": ["task-4"], "inputFiles": ["f_ra"], "outputFiles": ["f_as"]},
        {"id": "task-3", "name": "mDiffFit_00000002", "parents": ["task-1"], "children": ["task-4"], "inputFiles": [], "outputFiles": ["f_bs"]},
        {"id": "task-4", "name": "mConcatFit_00000001", "parents": ["task-2", "task-3"], "children": [], "inputFiles": ["f_as", "f_bs"], "outputFiles": []}
      ],
      "files": [
        {"id": "f_ra", "sizeInBytes": 100},
        {"id": "f_as", "sizeInBytes": 200},
        {"id": "f_bs", "sizeInBytes": 50}
      ]
    },
    "execution": {
      "makespanInSeconds": 14.0,
      "tasks": [
        {"id": "task-1", "runtimeInSeconds": 5.0, "memoryInBytes": 8000000000},
        {"id": "task-2", "runtimeInSeconds": 3.0},
        {"id": "task-3", "runtimeInSeconds": 4.0, "memoryInBytes": 4000000000},
        {"id": "task-4", "runtimeInSeconds": 2.0, "memoryInBytes": 0}
      ],
      "machines": [
        {"nodeName": "ref", "cpu": {"coreCount": 1, "speedInMHz": 1000}}
      ]
    }
  }
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_wfcommons.py
import json
import sys
from pathlib import Path

import numpy as np

from src.core.task import TaskClass
from src.dag_factory.wfcommons_adapter import parse_wfformat
from src.dag_factory.wfcommons_config import WfcommonsParams

_FIXTURE = Path(__file__).parent / "fixtures" / "wfformat_tiny.json"


def _doc() -> dict:
    with open(_FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


def _params() -> WfcommonsParams:
    return WfcommonsParams(
        default_mem=4.0, eps=0.01, bytes_to_unit=1.0,
        mem_min=1.0, mem_max=8.0, memory_ref_bytes=8e9,
    )


def test_parser_does_not_import_wfcommons() -> None:
    parse_wfformat(_doc(), np.random.default_rng(0), _params())
    assert "wfcommons" not in sys.modules


def test_indices_topological_tiebreak_by_id() -> None:
    dag = parse_wfformat(_doc(), np.random.default_rng(0), _params())
    assert dag.n_tasks == 4
    # base_cost in index order proves the mapping task-1->0 ... task-4->3
    assert [dag.task(i).base_cost for i in range(4)] == [5.0, 3.0, 4.0, 2.0]


def test_task_class_from_curated_montage_table() -> None:
    dag = parse_wfformat(_doc(), np.random.default_rng(0), _params())
    assert dag.task(0).task_class is TaskClass.DATA_PARALLEL   # mProject
    assert dag.task(1).task_class is TaskClass.STREAMING       # mDiffFit
    assert dag.task(2).task_class is TaskClass.STREAMING       # mDiffFit
    assert dag.task(3).task_class is TaskClass.SEQUENTIAL      # mConcatFit


def test_mem_required_derive_and_default() -> None:
    dag = parse_wfformat(_doc(), np.random.default_rng(0), _params())
    assert dag.task(0).mem_required == 8.0   # 8e9 -> mem_max
    assert dag.task(1).mem_required == 4.0   # absent -> default
    assert dag.task(2).mem_required == 4.5   # 4e9 -> midpoint
    assert dag.task(3).mem_required == 1.0   # 0 -> mem_min


def test_edges_and_volumes_from_shared_files_with_eps() -> None:
    dag = parse_wfformat(_doc(), np.random.default_rng(0), _params())
    assert sorted(dag.edge_index()) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert dag.edge_data(0, 1) == 100.0   # f_ra
    assert dag.edge_data(0, 2) == 0.01    # no shared file -> eps
    assert dag.edge_data(1, 3) == 200.0   # f_as
    assert dag.edge_data(2, 3) == 50.0    # f_bs


def test_result_satisfies_taskdag_invariants() -> None:
    dag = parse_wfformat(_doc(), np.random.default_rng(0), _params())
    # node_id == index over 0..N-1 and acyclic are enforced by TaskDAG.__init__;
    # critical_path_length being finite proves a valid DAG was built.
    assert dag.critical_path_length() > 0.0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_wfcommons.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dag_factory.wfcommons_adapter'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/dag_factory/wfcommons_adapter.py
"""Pure WfFormat v1.5 -> TaskDAG parser (TZ §7). No I/O, no `import wfcommons`.

Semantic mapping (disclosed in thesis):
  runtimeInSeconds -> base_cost (nominal, reference unit; no machine speed read)
  name             -> task_class (curated per-recipe table + hash fallback)
  memoryInBytes    -> mem_required (rescaled into [mem_min, mem_max], else default)
  parents/children -> edges; shared outputFiles n inputFiles sizes -> volume (else eps)
"""

import numpy as np

from src.core.dag import TaskDAG
from src.core.task import Task
from src.dag_factory.wfcommons_classes import (
    RECIPE_TABLES,
    abstract_name,
    assign_task_class,
    infer_recipe,
)
from src.dag_factory.wfcommons_config import WfcommonsParams


def _topo_order(ids: list[str], parents: dict[str, list[str]]) -> list[str]:
    """Kahn's algorithm with smallest-id tie-break -> deterministic ordering."""
    indeg = {i: len(parents[i]) for i in ids}
    children: dict[str, list[str]] = {i: [] for i in ids}
    for i in ids:
        for p in parents[i]:
            children[p].append(i)
    ready = sorted(i for i in ids if indeg[i] == 0)
    order: list[str] = []
    while ready:
        cur = ready.pop(0)
        order.append(cur)
        for c in children[cur]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
        ready.sort()
    if len(order) != len(ids):
        raise ValueError("WfFormat workflow is cyclic or has dangling parents.")
    return order


def _rescale_mem(mem_bytes: float, params: WfcommonsParams) -> float:
    frac = mem_bytes / params.memory_ref_bytes
    mem = params.mem_min + frac * (params.mem_max - params.mem_min)
    return float(min(max(mem, params.mem_min), params.mem_max))


def parse_wfformat(
    doc: dict,
    rng: np.random.Generator,  # noqa: ARG001 (reserved for future stochastic mapping)
    params: WfcommonsParams,
    recipe: str | None = None,
) -> TaskDAG:
    spec = doc["workflow"]["specification"]
    execu = doc["workflow"]["execution"]

    spec_tasks = {t["id"]: t for t in spec["tasks"]}
    file_size = {f["id"]: float(f["sizeInBytes"]) for f in spec.get("files", [])}
    runtime = {t["id"]: float(t["runtimeInSeconds"]) for t in execu["tasks"]}
    mem_bytes = {
        t["id"]: float(t["memoryInBytes"])
        for t in execu["tasks"]
        if t.get("memoryInBytes") is not None
    }

    ids = list(spec_tasks)
    parents = {i: list(spec_tasks[i]["parents"]) for i in ids}
    order = _topo_order(ids, parents)
    index = {sid: k for k, sid in enumerate(order)}

    table_key = recipe or infer_recipe([abstract_name(spec_tasks[i]["name"]) for i in ids])
    table = RECIPE_TABLES.get(table_key, {}) if table_key is not None else {}

    tasks: list[Task] = []
    for sid in order:
        if sid not in runtime:
            raise ValueError(f"Task {sid!r} has no execution.runtimeInSeconds entry.")
        mem = _rescale_mem(mem_bytes[sid], params) if sid in mem_bytes else params.default_mem
        tasks.append(
            Task(
                id=index[sid],
                base_cost=runtime[sid],
                mem_required=mem,
                task_class=assign_task_class(spec_tasks[sid]["name"], table),
            )
        )

    edges: list[tuple[int, int, float]] = []
    for sid in order:
        out_files = set(spec_tasks[sid]["outputFiles"])
        for child in spec_tasks[sid]["children"]:
            shared = out_files & set(spec_tasks[child]["inputFiles"])
            volume = sum(file_size[f] for f in shared) * params.bytes_to_unit
            edges.append((index[sid], index[child], volume if shared else params.eps))

    return TaskDAG(tasks, edges)
```

Note: this module deliberately does not import `TaskClass` (it never references it directly — class selection is fully delegated to `assign_task_class`). It also never imports `wfcommons` or `json`. `parse_wfformat`'s `rng` is unused for now (reserved for future stochastic mapping); the `# noqa: ARG001` keeps ruff quiet without dropping the parameter from the public signature.

- [ ] **Step 5: Run tests + lint**

Run: `.venv/bin/pytest tests/test_wfcommons.py -v && .venv/bin/ruff check src/dag_factory/wfcommons_adapter.py tests/test_wfcommons.py && .venv/bin/black --check src/dag_factory/wfcommons_adapter.py tests/test_wfcommons.py`
Expected: PASS (6 tests), ruff clean, black clean.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/wfformat_tiny.json src/dag_factory/wfcommons_adapter.py tests/test_wfcommons.py
git commit -m "feat: pure WfFormat v1.5 -> TaskDAG parser + hand-checked fixture"
```

---

## Task 4: Factory routing

**Files:**
- Modify: `src/dag_factory/factory.py`
- Test: `tests/test_wfcommons_factory.py`

**Interfaces:**
- Consumes: `parse_wfformat` (Task 3), `load_wfcommons_params` (Task 1).
- Produces:
  - `DAGFactory.load_from_wfcommons(path: str, rng: np.random.Generator, recipe: str | None = None) -> TaskDAG`.
  - `DAGFactory.create(source="wfcommons", rng=..., path=<str>, recipe=<str|None>)` delegates to it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wfcommons_factory.py
from pathlib import Path

import numpy as np

from src.dag_factory.factory import DAGFactory

_FIXTURE = str(Path(__file__).parent / "fixtures" / "wfformat_tiny.json")


def test_load_from_wfcommons_reads_file() -> None:
    dag = DAGFactory.load_from_wfcommons(_FIXTURE, np.random.default_rng(0))
    assert dag.n_tasks == 4
    assert sorted(dag.edge_index()) == [(0, 1), (0, 2), (1, 3), (2, 3)]


def test_create_source_wfcommons_delegates() -> None:
    dag = DAGFactory.create("wfcommons", np.random.default_rng(0), path=_FIXTURE)
    assert dag.n_tasks == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_wfcommons_factory.py -v`
Expected: FAIL with `AttributeError: type object 'DAGFactory' has no attribute 'load_from_wfcommons'`

- [ ] **Step 3: Write implementation (full new factory.py)**

```python
# src/dag_factory/factory.py
"""DAGFactory: Factory pattern over interchangeable DAG sources (TZ §3, §7)."""

import json

import numpy as np

from src.core.dag import TaskDAG
from src.dag_factory.synthetic import generate_synthetic
from src.dag_factory.wfcommons_adapter import parse_wfformat
from src.dag_factory.wfcommons_config import load_wfcommons_params


class DAGFactory:
    @classmethod
    def create(cls, source: str, rng: np.random.Generator, **params: object) -> TaskDAG:
        if source == "synthetic":
            return generate_synthetic(
                rng,
                n_tasks=int(params["n_tasks"]),  # type: ignore[arg-type]
                n_layers=int(params["n_layers"]),  # type: ignore[arg-type]
                edge_prob=float(params["edge_prob"]),  # type: ignore[arg-type]
                ccr=float(params["ccr"]),  # type: ignore[arg-type]
            )
        if source == "wfcommons":
            recipe = params.get("recipe")
            return cls.load_from_wfcommons(
                str(params["path"]),
                rng,
                recipe=str(recipe) if recipe is not None else None,
            )
        raise ValueError(f"Unknown DAG source: {source!r}")

    @classmethod
    def load_from_wfcommons(
        cls,
        path: str,
        rng: np.random.Generator,
        recipe: str | None = None,
        config_path: str = "config.yaml",
    ) -> TaskDAG:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        params = load_wfcommons_params(config_path)
        return parse_wfformat(doc, rng, params, recipe=recipe)
```

- [ ] **Step 4: Run tests + full suite + lint**

Run: `.venv/bin/pytest tests/test_wfcommons_factory.py tests/test_synthetic.py -v && .venv/bin/ruff check src/dag_factory/factory.py tests/test_wfcommons_factory.py && .venv/bin/black --check src/dag_factory/factory.py tests/test_wfcommons_factory.py`
Expected: PASS (existing synthetic tests still green — `create` signature widened, behavior unchanged), ruff clean, black clean.

- [ ] **Step 5: Commit**

```bash
git add src/dag_factory/factory.py tests/test_wfcommons_factory.py
git commit -m "feat: DAGFactory.load_from_wfcommons + wfcommons source routing"
```

---

## Task 5: Optional generation CLI + deselected integration test

**Files:**
- Create: `tools/gen_wfcommons.py`
- Create: `tests/test_wfcommons_live.py`
- Modify: `pyproject.toml`
- Create: `dag_benchmarks/README.md`

**Interfaces:**
- Consumes: `DAGFactory.load_from_wfcommons` (Task 4); `wfcommons` library (lazy, optional).
- Produces: `generate(recipe: str, n_tasks: int, seed: int, out_dir: str) -> str` (returns written path); CLI `python -m tools.gen_wfcommons ...`.

- [ ] **Step 1: Register the marker (so the deselect step is meaningful)**

Append to `pyproject.toml` under `[tool.pytest.ini_options]` (which currently has `pythonpath` and `testpaths`):

```toml
addopts = "-m 'not wfcommons'"
markers = [
    "wfcommons: live wfcommons-library integration tests (deselected by default; run with -m wfcommons)",
]
```

- [ ] **Step 2: Write the failing integration test**

```python
# tests/test_wfcommons_live.py
"""Live wfcommons integration — deselected by default (run: pytest -m wfcommons).

Catches wfcommons API/schema drift by generating a real workflow and feeding it
through the parser. Requires `pip install wfcommons`.
"""

import numpy as np
import pytest

wfcommons = pytest.importorskip("wfcommons")


@pytest.mark.wfcommons
def test_live_recipe_parses(tmp_path) -> None:
    from tools.gen_wfcommons import generate

    path = generate(recipe="montage", n_tasks=20, seed=42, out_dir=str(tmp_path))

    from src.dag_factory.factory import DAGFactory

    dag = DAGFactory.load_from_wfcommons(path, np.random.default_rng(0), recipe="montage")
    assert dag.n_tasks >= 1
    assert dag.critical_path_length() > 0.0
```

- [ ] **Step 3: Verify it is deselected by default**

Run: `.venv/bin/pytest tests/test_wfcommons_live.py -v`
Expected: `deselected` (and/or skipped via `importorskip` if wfcommons absent) — 0 tests run, exit code 0. The default suite never imports `wfcommons`.

Run: `.venv/bin/pytest -m wfcommons tests/test_wfcommons_live.py -v`
Expected (wfcommons not installed): the whole module is skipped by `importorskip`. This is the expected state on this machine; the test only executes after `pip install wfcommons`.

- [ ] **Step 4: Write the generation CLI**

```python
# tools/gen_wfcommons.py
"""Optional CLI: generate WfFormat JSON benchmarks via wfcommons (TZ §7, §11).

wfcommons is an OPTIONAL extra; it is imported lazily here only. Runtime and the
default test suite never import it. Run:
    .venv/bin/python -m tools.gen_wfcommons --recipe montage --n-tasks 20 --seed 42
"""

import argparse
import os

_RECIPES = {
    "montage": "MontageRecipe",
    "cybershake": "CyberShakeRecipe",
    "blast": "BlastRecipe",
}


def _load_recipe_class(recipe: str):
    try:
        from wfcommons import recipes  # noqa: PLC0415  (lazy, optional dependency)
    except ImportError as exc:  # pragma: no cover - exercised only without wfcommons
        raise ImportError(
            "wfcommons is required to generate benchmarks: pip install wfcommons"
        ) from exc
    if recipe not in _RECIPES:
        raise ValueError(f"Unknown recipe {recipe!r}; choices: {sorted(_RECIPES)}")
    return getattr(recipes, _RECIPES[recipe])


def generate(recipe: str, n_tasks: int, seed: int, out_dir: str) -> str:
    """Build one workflow and write it as WfFormat JSON; return the written path."""
    import random

    random.seed(seed)
    recipe_cls = _load_recipe_class(recipe)
    workflow = recipe_cls.from_num_tasks(n_tasks).build_workflow()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{recipe}_{n_tasks}_seed{seed}.json")
    workflow.write_json(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WfFormat JSON benchmarks.")
    parser.add_argument("--recipe", required=True, choices=sorted(_RECIPES))
    parser.add_argument("--n-tasks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="dag_benchmarks")
    args = parser.parse_args()
    path = generate(args.recipe, args.n_tasks, args.seed, args.out_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

Note: confirm `recipes.MontageRecipe` / `from_num_tasks` / `workflow.write_json` against the installed wfcommons when running the live test; adjust the `_RECIPES` map and `write_json` call if the API differs. The live test (Step 3) is the drift detector for exactly this.

- [ ] **Step 5: Write `dag_benchmarks/README.md`**

```markdown
# dag_benchmarks

Frozen, committed WfFormat v1.5 workflows for the M5 evaluation grid. Loaded via
`DAGFactory.load_from_wfcommons(path, rng, recipe=...)`. These are DATA, not
generated at test time.

## Regenerating (requires `pip install wfcommons`)

Pinned seed = 42. Exact commands:

    .venv/bin/python -m tools.gen_wfcommons --recipe montage    --n-tasks 30 --seed 42
    .venv/bin/python -m tools.gen_wfcommons --recipe cybershake --n-tasks 30 --seed 42
    .venv/bin/python -m tools.gen_wfcommons --recipe blast      --n-tasks 30 --seed 42

After regenerating, run the drift check and verify/extend the curated class tables
in `src/dag_factory/wfcommons_classes.py` against the real task names:

    .venv/bin/pytest -m wfcommons
```

- [ ] **Step 6: Run the default suite to confirm hygiene**

Run: `.venv/bin/pytest -q`
Expected: entire suite passes; the `wfcommons` marker is deselected; no `wfcommons` import occurs. Confirm with:
`.venv/bin/python -c "import tests.test_wfcommons, sys; assert 'wfcommons' not in sys.modules; print('clean')"`
Expected: `clean`.

- [ ] **Step 7: Lint**

Run: `.venv/bin/ruff check tools/gen_wfcommons.py tests/test_wfcommons_live.py && .venv/bin/black --check tools/gen_wfcommons.py tests/test_wfcommons_live.py`
Expected: ruff clean, black clean.

- [ ] **Step 8: Commit**

```bash
git add tools/gen_wfcommons.py tests/test_wfcommons_live.py pyproject.toml dag_benchmarks/README.md
git commit -m "feat: optional wfcommons generation CLI + deselected drift test"
```

---

## Task 6: Generate and commit the benchmark set (manual, requires wfcommons)

> This task is performed once, by a human/agent with `wfcommons` installed, to freeze
> the realistic benchmark set. It produces committed data, not code. If `wfcommons`
> cannot be installed in this environment, STOP after Task 5 and hand this task off —
> the parser, factory, and CLI are complete and tested without it.

**Files:**
- Create: `dag_benchmarks/*.json` (generated)
- Possibly modify: `src/dag_factory/wfcommons_classes.py` (extend tables to real names)

- [ ] **Step 1: Install the optional extra**

Run: `.venv/bin/pip install wfcommons`

- [ ] **Step 2: Run the drift test**

Run: `.venv/bin/pytest -m wfcommons -v`
Expected: PASS. If it fails, fix `tools/gen_wfcommons.py` (`_RECIPES` names / `write_json`) per the actual wfcommons API, then re-run.

- [ ] **Step 3: Generate the frozen set (seed 42)**

Run:
```bash
.venv/bin/python -m tools.gen_wfcommons --recipe montage    --n-tasks 30 --seed 42
.venv/bin/python -m tools.gen_wfcommons --recipe cybershake --n-tasks 30 --seed 42
.venv/bin/python -m tools.gen_wfcommons --recipe blast      --n-tasks 30 --seed 42
```
Expected: three `dag_benchmarks/*.json` files written.

- [ ] **Step 4: Verify the curated tables against real task names**

Run:
```bash
.venv/bin/python -c "import json,glob; from src.dag_factory.wfcommons_classes import abstract_name; \
print(sorted({abstract_name(t['name']) for p in glob.glob('dag_benchmarks/*.json') \
for t in json.load(open(p))['workflow']['specification']['tasks']}))"
```
Compare the printed abstract names to `RECIPE_TABLES`. Add any missing names to the
relevant table so each benchmark spans >= 2 classes via curated entries (not just the
fallback). Re-run `.venv/bin/pytest tests/test_wfcommons_classes.py -v` after edits.

- [ ] **Step 5: Confirm the committed set loads through the factory**

```python
# scratch check — run via: .venv/bin/python -
import glob
import numpy as np
from src.dag_factory.factory import DAGFactory
for p in sorted(glob.glob("dag_benchmarks/*.json")):
    dag = DAGFactory.load_from_wfcommons(p, np.random.default_rng(0))
    assert dag.n_tasks >= 20
    assert dag.critical_path_length() > 0.0
    print(p, dag.n_tasks, "ok")
```
Expected: each file loads, prints `ok`.

- [ ] **Step 6: Commit the frozen set**

```bash
git add dag_benchmarks/*.json src/dag_factory/wfcommons_classes.py
git commit -m "data: frozen WfCommons benchmark set (seed 42) + curated table updates"
```

---

## Self-Review

**Spec coverage:**
- §1 Decision (Option 1) — Tasks 3 (parser) + 5 (optional generator), wfcommons never in default suite. ✓
- §2.1 id→index (topo, tie-break by id) — Task 3 `_topo_order` + `test_indices_topological_tiebreak_by_id`. ✓
- §2.2 runtime→base_cost verbatim, fail loud — Task 3 parser + `ValueError` on missing runtime. ✓
- §2.3 curated tables + abstract name + hash fallback + ≥2 classes — Task 2 (+ tests). ✓
- §2.4 mem rescale else default — Task 3 `_rescale_mem` + `test_mem_required_derive_and_default`. ✓
- §2.5 edges from parents/children, volume from shared files, EPS, BYTES_TO_UNIT — Task 3 + `test_edges_and_volumes_from_shared_files_with_eps`. ✓
- §2.6 TaskDAG invariants — enforced by `TaskDAG.__init__`, asserted in Task 3. ✓
- §3 module layout + config block — Tasks 1–5. ✓
- §4 two data roles (fixture vs benchmark set) — fixture in Task 3, benchmark set in Tasks 5/6. ✓
- §5 optional CLI, lazy import, marker deselected, drift test — Task 5. ✓
- §6 single factory entry, no special-casing downstream — Task 4. ✓
- §7 testing strategy — Tasks 2–5. ✓
- §8 exit criteria — Tasks 3 (fixture parse), 6 (eval grid load), 5 (documented/seed-pinned/deselected). ✓

**Placeholder scan:** Curated cybershake/blast names are best-effort, with Task 6 Step 4 as the concrete verification action and the hash fallback guaranteeing correctness meanwhile — not a TODO. No "TBD"/"implement later" remain.

**Type consistency:** `parse_wfformat(doc, rng, params, recipe=None)` signature is identical in Tasks 3 and 4. `WfcommonsParams` field names match between Task 1 definition and Tasks 3/4 usage. `assign_task_class(name, table)` / `infer_recipe(list)` / `abstract_name(str)` signatures consistent across Tasks 2 and 3. `load_from_wfcommons(path, rng, recipe=None, config_path=...)` consistent between Tasks 4 and 5.
