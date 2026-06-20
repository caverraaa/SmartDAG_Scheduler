from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.placement import ClusterState, horizon


def _nodes() -> list[ComputeNode]:
    return [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]


def test_cluster_state_calendar_fields_default_empty() -> None:
    dag = TaskDAG([Task(0, 1.0, 1.0, TaskClass.SEQUENTIAL)], [])
    st = ClusterState(nodes=_nodes(), dag=dag, task_finish={}, task_node={}, m_ref=1.0, e_ref=1.0)
    assert st.failure_times == {} and st.noise_eps == {}


def test_horizon_includes_dead_node_committed_work() -> None:
    nodes = _nodes()
    nodes[1].free_at_time = 9.0
    nodes[1].alive = False  # dead, but its committed work elapsed to t=9
    nodes[0].free_at_time = 4.0
    assert horizon(nodes) == 9.0  # dead node still counts toward the horizon
