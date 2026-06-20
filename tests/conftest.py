import pytest

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass


@pytest.fixture
def golden_instance() -> tuple[TaskDAG, list[ComputeNode]]:
    """Diamond DAG (0->{1,2}->3) on 1 CPU + 1 GPU; hand-verified schedule.

    base_costs: t0=2, t1=4, t2=4, t3=2. CPU speed 1 (power 100),
    GPU speed 2 (power 200). All edges carry data=10, bandwidth=10 (comm=1
    cross-node, 0 intra-node).
    """
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 4.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 4.0, 1.0, TaskClass.SEQUENTIAL),
        Task(3, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes
