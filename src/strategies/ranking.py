"""Cluster-aware HEFT/CPOP rank functions (averaged over processors) (TZ §8).

These are the canonical list-scheduling priorities: averaged exec/comm costs,
distinct from TaskDAG's structural base_cost-weighted b_level/t_level.
"""

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.env.cost_model import exec_time


def _mean_bandwidth(nodes: list[ComputeNode]) -> float:
    return sum(n.bandwidth for n in nodes) / len(nodes)


def mean_exec(dag: TaskDAG, tid: int, nodes: list[ComputeNode]) -> float:
    task = dag.task(tid)
    return sum(exec_time(task, n) for n in nodes) / len(nodes)


def mean_comm(dag: TaskDAG, src: int, dst: int, nodes: list[ComputeNode]) -> float:
    return dag.edge_data(src, dst) / _mean_bandwidth(nodes)


def upward_rank(dag: TaskDAG, nodes: list[ComputeNode]) -> dict[int, float]:
    memo: dict[int, float] = {}

    def ru(i: int) -> float:
        if i in memo:
            return memo[i]
        succ = dag.successors(i)
        if not succ:
            value = mean_exec(dag, i, nodes)
        else:
            value = mean_exec(dag, i, nodes) + max(
                mean_comm(dag, i, j, nodes) + ru(j) for j in succ
            )
        memo[i] = value
        return value

    for i in range(dag.n_tasks):
        ru(i)
    return memo


def downward_rank(dag: TaskDAG, nodes: list[ComputeNode]) -> dict[int, float]:
    memo: dict[int, float] = {}

    def rd(i: int) -> float:
        if i in memo:
            return memo[i]
        pred = dag.predecessors(i)
        if not pred:
            value = 0.0
        else:
            value = max(
                rd(p) + mean_exec(dag, p, nodes) + mean_comm(dag, p, i, nodes) for p in pred
            )
        memo[i] = value
        return value

    for i in range(dag.n_tasks):
        rd(i)
    return memo
