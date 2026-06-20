"""Observation builder: per-task / per-node / global features + masks (TZ §6.2).

Per-task and per-node features are normalized to O(1) scale for consumption by the
GNN in M3a (graph/node pooled embeddings g,c are prepended to globals). Normalization
rules: T_MEM/T_OUTDATA by task means, T_OUTDEG/T_UNSCHED_PREDS by n_tasks,
N_SPEED/N_POWER by node means, N_UTIL clamped [0,1]. Unchanged: T_BASE_COST, T_COST_*,
T_BLEVEL, T_TLEVEL, flags, masks, N_FREE_REL, globals.
"""

from dataclasses import dataclass

import numpy as np

from src.core.compute_node import ComputeNode, NodeType
from src.env.cost_model import exec_time
from src.env.placement import ClusterState

# --- task feature columns ---
T_BASE_COST = 0
T_COST_CPU = 1
T_COST_GPU = 2
T_COST_FPGA = 3
T_COST_TPU = 4
T_MEM = 5
T_DONE = 6
T_READY = 7
T_BLOCKED = 8
T_SCHEDULED = 9
T_UNSCHED_PREDS = 10
T_BLEVEL = 11
T_TLEVEL = 12
T_OUTDEG = 13
T_OUTDATA = 14
N_TASK_FEATURES = 15

# --- node feature columns ---
N_TYPE_CPU = 0
N_TYPE_GPU = 1
N_TYPE_FPGA = 2
N_TYPE_TPU = 3
N_FREE_REL = 4
N_UTIL = 5
N_POWER = 6
N_SPEED = 7
N_ALIVE = 8
N_NODE_FEATURES = 9

_TYPE_COL = {
    NodeType.CPU: N_TYPE_CPU,
    NodeType.GPU: N_TYPE_GPU,
    NodeType.FPGA: N_TYPE_FPGA,
    NodeType.TPU: N_TYPE_TPU,
}
_COST_COL = {
    NodeType.CPU: T_COST_CPU,
    NodeType.GPU: T_COST_GPU,
    NodeType.FPGA: T_COST_FPGA,
    NodeType.TPU: T_COST_TPU,
}


@dataclass
class Observation:
    task_features: np.ndarray
    node_features: np.ndarray
    globals: np.ndarray
    edge_index: np.ndarray
    ready_mask: np.ndarray
    alive_mask: np.ndarray
    nodes: list[ComputeNode]


def _mean_cost(state: ClusterState) -> float:
    costs = [state.dag.task(i).base_cost for i in range(state.dag.n_tasks)]
    return sum(costs) / len(costs) if costs else 1.0


def build_observation(
    state: ClusterState, scheduled: set[int], current_makespan: float
) -> Observation:
    dag = state.dag
    n = dag.n_tasks
    m = len(state.nodes)
    mean_cost = _mean_cost(state)
    cp = dag.critical_path_length() or 1.0
    ready = set(dag.ready_set(scheduled))

    # Compute instance-wise means for normalization (eps-guarded with 1e-8)
    mems = [dag.task(i).mem_required for i in range(n)]
    mean_mem = (sum(mems) / n) + 1e-8 if n else 1.0
    outs = [dag.out_data(i) for i in range(n)]
    mean_out = (sum(outs) / n) + 1e-8 if n else 1.0
    node_speeds = [float(np.mean(list(node.speed_by_class.values()))) for node in state.nodes]
    mean_speed = (sum(node_speeds) / m) + 1e-8 if m else 1.0
    mean_power = (sum(node.power_w for node in state.nodes) / m) + 1e-8 if m else 1.0
    denom_tasks = float(n) if n else 1.0

    task_features = np.zeros((n, N_TASK_FEATURES), dtype=np.float32)
    ready_mask = np.zeros(n, dtype=bool)
    # one representative node per present type for the per-task cost vector
    type_repr: dict[NodeType, ComputeNode] = {}
    for node in state.nodes:
        type_repr.setdefault(node.node_type, node)

    for tid in range(n):
        task = dag.task(tid)
        f = task_features[tid]
        f[T_BASE_COST] = task.base_cost / mean_cost
        for nt, node in type_repr.items():
            f[_COST_COL[nt]] = exec_time(task, node) / cp
        f[T_MEM] = task.mem_required / mean_mem
        is_scheduled = tid in scheduled
        is_ready = tid in ready
        f[T_DONE] = 1.0 if is_scheduled else 0.0
        f[T_READY] = 1.0 if (is_ready and not is_scheduled) else 0.0
        f[T_BLOCKED] = 1.0 if (not is_ready and not is_scheduled) else 0.0
        f[T_SCHEDULED] = 1.0 if is_scheduled else 0.0
        f[T_UNSCHED_PREDS] = (
            float(sum(1 for p in dag.predecessors(tid) if p not in scheduled)) / denom_tasks
        )
        f[T_BLEVEL] = dag.b_level(tid) / cp
        f[T_TLEVEL] = dag.t_level(tid) / cp
        f[T_OUTDEG] = float(dag.out_degree(tid)) / denom_tasks
        f[T_OUTDATA] = dag.out_data(tid) / mean_out
        ready_mask[tid] = is_ready and not is_scheduled

    node_features = np.zeros((m, N_NODE_FEATURES), dtype=np.float32)
    alive_mask = np.zeros(m, dtype=bool)
    for j, node in enumerate(state.nodes):
        nf = node_features[j]
        nf[_TYPE_COL[node.node_type]] = 1.0
        nf[N_FREE_REL] = (node.free_at_time - state.sim_time) / cp
        nf[N_UTIL] = (
            min(1.0, max(0.0, node.free_at_time / current_makespan))
            if current_makespan > 0
            else 0.0
        )
        nf[N_POWER] = node.power_w / mean_power
        nf[N_SPEED] = float(np.mean(list(node.speed_by_class.values()))) / mean_speed
        nf[N_ALIVE] = 1.0 if node.alive else 0.0
        alive_mask[j] = node.alive

    edges = dag.edge_index()
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    globals_ = np.array([current_makespan / cp, len(scheduled) / n if n else 0.0], dtype=np.float32)
    return Observation(
        task_features=task_features,
        node_features=node_features,
        globals=globals_,
        edge_index=edge_index,
        ready_mask=ready_mask,
        alive_mask=alive_mask,
        nodes=state.nodes,
    )
