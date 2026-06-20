from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.placement import (
    ClusterState,
    earliest_start_finish,
    horizon,
    weighted_cost,
)


def _setup() -> ClusterState:
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 4.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return ClusterState(nodes=nodes, dag=dag, task_finish={}, task_node={}, m_ref=6.0, e_ref=600.0)


def test_eft_first_task_starts_at_zero() -> None:
    st = _setup()
    start, finish = earliest_start_finish(st.dag.task(0), st.nodes[0], st)
    assert (start, finish) == (0.0, 2.0)


def test_eft_respects_predecessor_and_cross_node_comm() -> None:
    st = _setup()
    # task 0 finished at t=2 on node 0
    st.task_finish[0] = 2.0
    st.task_node[0] = 0
    # task 1 on node 1 (cross-node): comm = 10/10 = 1.0 -> ready at 3.0
    start, finish = earliest_start_finish(st.dag.task(1), st.nodes[1], st)
    assert start == 3.0 and finish == 7.0
    # task 1 on node 0 (same node): no comm -> ready at 2.0
    start0, finish0 = earliest_start_finish(st.dag.task(1), st.nodes[0], st)
    assert start0 == 2.0 and finish0 == 6.0


def test_weighted_cost_returns_normalised_components() -> None:
    st = _setup()
    comp = weighted_cost(st.dag.task(0), st.nodes[0], st)
    # horizon before = 0, after = 2 -> d_makespan = 2 ; /m_ref(6) = 1/3
    assert abs(comp.d_makespan_norm - (2.0 / 6.0)) < 1e-9
    # energy = 100 * 2 = 200 ; /e_ref(600) = 1/3
    assert abs(comp.d_energy_norm - (200.0 / 600.0)) < 1e-9


def test_horizon_zero_delta_when_task_fits_under_existing_horizon() -> None:
    st = _setup()
    # push node 0 horizon to 10 by pretending it is busy
    st.nodes[0].free_at_time = 10.0
    assert horizon(st.nodes) == 10.0
    # schedule task 0 as predecessor
    st.task_finish[0] = 2.0
    st.task_node[0] = 0
    # task 1 on node 1: comm = 10/10 = 1 -> starts at 3, finishes at 7 (< 10)
    # d_makespan = 0 (task fits under horizon)
    comp = weighted_cost(st.dag.task(1), st.nodes[1], st)
    assert comp.d_makespan_norm == 0.0
