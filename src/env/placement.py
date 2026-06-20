"""The single side-effect-free objective evaluator (TZ Appendix A.1/A.2)."""

from dataclasses import dataclass

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.core.task import Task
from src.env.cost_model import comm_time, energy, exec_time


@dataclass
class ClusterState:
    nodes: list[ComputeNode]
    dag: TaskDAG
    task_finish: dict[int, float]
    task_node: dict[int, int]
    m_ref: float
    e_ref: float
    sim_time: float = 0.0


@dataclass(frozen=True)
class CostComponents:
    d_makespan_norm: float
    d_energy_norm: float


def horizon(nodes: list[ComputeNode]) -> float:
    """Running schedule horizon = max free_at_time over alive nodes."""
    alive = [n.free_at_time for n in nodes if n.alive]
    return max(alive) if alive else 0.0


def earliest_start_finish(
    task: Task, node: ComputeNode, state: ClusterState
) -> tuple[float, float]:
    """Append-only EFT: when can this task start/finish on this node, given state.

    Precondition: every predecessor of `task` must already be present in
    `state.task_finish` (i.e. the task is ready); otherwise a KeyError is raised.
    """
    ready_time = 0.0
    for pred in state.dag.predecessors(task.id):
        pred_finish = state.task_finish[pred]
        if state.task_node[pred] != node.node_id:
            pred_finish += comm_time(state.dag.edge_data(pred, task.id), node.bandwidth)
        ready_time = max(ready_time, pred_finish)
    start = max(node.free_at_time, ready_time)
    return start, start + exec_time(task, node)


def weighted_cost(task: Task, node: ComputeNode, state: ClusterState) -> CostComponents:
    """Normalised (Δmakespan/M_ref, Δenergy/E_ref) for a candidate assignment.

    Δmakespan is the change in the schedule HORIZON (A.2), not the task finish.
    Side-effect-free: never mutates node/state.
    Inherits the predecessor-readiness precondition from earliest_start_finish.
    """
    before = horizon(state.nodes)
    _, finish = earliest_start_finish(task, node, state)
    after = max(before, finish)
    d_makespan = after - before
    d_energy = energy(task, node)
    return CostComponents(
        d_makespan_norm=d_makespan / state.m_ref,
        d_energy_norm=d_energy / state.e_ref,
    )
