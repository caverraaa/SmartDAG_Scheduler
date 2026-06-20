from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.observation import (
    N_ALIVE,
    N_NODE_FEATURES,
    N_TASK_FEATURES,
    T_READY,
    T_SCHEDULED,
    build_observation,
)
from src.env.placement import ClusterState


def _state() -> ClusterState:
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 4.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 300.0, 10.0),
    ]
    return ClusterState(nodes=nodes, dag=dag, task_finish={}, task_node={}, m_ref=6.0, e_ref=600.0)


def test_shapes() -> None:
    st = _state()
    obs = build_observation(st, scheduled=set(), current_makespan=0.0)
    assert obs.task_features.shape == (3, N_TASK_FEATURES)
    assert obs.node_features.shape == (2, N_NODE_FEATURES)
    assert obs.globals.shape == (2,)


def test_ready_mask_only_true_for_ready_unscheduled_tasks() -> None:
    st = _state()
    obs = build_observation(st, scheduled=set(), current_makespan=0.0)
    assert obs.ready_mask.tolist() == [True, False, False]  # only task 0 ready
    obs2 = build_observation(st, scheduled={0}, current_makespan=2.0)
    assert obs2.ready_mask.tolist() == [False, True, True]  # 1 and 2 now ready, 0 scheduled


def test_alive_mask_reflects_node_alive_flag() -> None:
    st = _state()
    st.nodes[1].alive = False
    obs = build_observation(st, scheduled=set(), current_makespan=0.0)
    assert obs.alive_mask.tolist() == [True, False]
    assert obs.node_features[1, N_ALIVE] == 0.0


def test_scheduled_flag_set() -> None:
    st = _state()
    obs = build_observation(st, scheduled={0}, current_makespan=2.0)
    assert obs.task_features[0, T_SCHEDULED] == 1.0
    assert obs.task_features[0, T_READY] == 0.0
