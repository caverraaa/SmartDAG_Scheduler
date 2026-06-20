from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.strategies.ranking import downward_rank, mean_comm, mean_exec, upward_rank


def _hetero_diamond() -> tuple[TaskDAG, list[ComputeNode]]:
    # 0->1->3, 0->2->3 ; base costs 2,4,4,2 ; edges all data=10
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


def test_mean_exec_and_comm() -> None:
    dag, nodes = _hetero_diamond()
    # task0 base 2: (2/1 + 2/2)/2 = 1.5
    assert mean_exec(dag, 0, nodes) == 1.5
    assert mean_exec(dag, 1, nodes) == 3.0
    # edge data 10 / mean bandwidth 10 = 1.0
    assert mean_comm(dag, 0, 1, nodes) == 1.0


def test_upward_rank_golden() -> None:
    dag, nodes = _hetero_diamond()
    ru = upward_rank(dag, nodes)
    assert ru[3] == 1.5
    assert ru[1] == 5.5
    assert ru[2] == 5.5
    assert ru[0] == 8.0


def test_downward_rank_golden() -> None:
    dag, nodes = _hetero_diamond()
    rd = downward_rank(dag, nodes)
    assert rd[0] == 0.0
    assert rd[1] == 2.5
    assert rd[2] == 2.5
    assert rd[3] == 6.5
