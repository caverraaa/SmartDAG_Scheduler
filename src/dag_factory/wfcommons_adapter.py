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
