from src.core.compute_node import ComputeNode, NodeType
from src.core.task import Task, TaskClass


def _node() -> ComputeNode:
    return ComputeNode(
        node_id=0,
        node_type=NodeType.GPU,
        speed_by_class={
            TaskClass.DATA_PARALLEL: 4.0,
            TaskClass.SEQUENTIAL: 1.0,
            TaskClass.STREAMING: 2.0,
        },
        power_w=300.0,
        bandwidth=10.0,
    )


def test_task_is_immutable() -> None:
    t = Task(id=1, base_cost=5.0, mem_required=2.0, task_class=TaskClass.SEQUENTIAL)
    assert t.id == 1 and t.task_class is TaskClass.SEQUENTIAL


def test_node_speed_lookup_and_reset() -> None:
    n = _node()
    assert n.speed(TaskClass.DATA_PARALLEL) == 4.0
    assert n.free_at_time == 0.0 and n.alive is True
    n.free_at_time = 12.0
    n.alive = False
    n.reset()
    assert n.free_at_time == 0.0 and n.alive is True


def test_four_node_types_exist() -> None:
    assert {nt.name for nt in NodeType} == {"CPU", "GPU", "FPGA", "TPU"}
