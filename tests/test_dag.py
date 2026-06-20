import pytest

from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass


def _diamond() -> TaskDAG:
    # 0 -> 1 -> 3 ; 0 -> 2 -> 3   (base_costs: 2,4,4,2)
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 4.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 4.0, 1.0, TaskClass.SEQUENTIAL),
        Task(3, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    edges = [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)]
    return TaskDAG(tasks, edges)


def test_rejects_cycle() -> None:
    tasks = [Task(0, 1.0, 1.0, TaskClass.SEQUENTIAL), Task(1, 1.0, 1.0, TaskClass.SEQUENTIAL)]
    with pytest.raises(ValueError):
        TaskDAG(tasks, [(0, 1, 1.0), (1, 0, 1.0)])


def test_structure_queries() -> None:
    d = _diamond()
    assert d.n_tasks == 4
    assert d.predecessors(3) == [1, 2]
    assert d.successors(0) == [1, 2]
    assert d.out_degree(0) == 2
    assert d.out_data(0) == 20.0
    assert d.edge_data(1, 3) == 10.0


def test_ready_set_tracks_predecessors() -> None:
    d = _diamond()
    assert d.ready_set(set()) == [0]
    assert d.ready_set({0}) == [1, 2]
    assert d.ready_set({0, 1}) == [2]
    assert d.ready_set({0, 1, 2}) == [3]
    assert d.ready_set({0, 1, 2, 3}) == []


def test_levels_and_critical_path() -> None:
    d = _diamond()
    # b_level(3)=2 ; b_level(1)=4+2=6 ; b_level(0)=2+6=8
    assert d.b_level(3) == 2.0
    assert d.b_level(1) == 6.0
    assert d.b_level(0) == 8.0
    # t_level(0)=0 ; t_level(1)=2 ; t_level(3)=2+4=6
    assert d.t_level(0) == 0.0
    assert d.t_level(1) == 2.0
    assert d.t_level(3) == 6.0
    assert d.critical_path_length() == 8.0


def test_rejects_noncontiguous_ids() -> None:
    tasks = [Task(0, 1.0, 1.0, TaskClass.SEQUENTIAL), Task(5, 1.0, 1.0, TaskClass.SEQUENTIAL)]
    with pytest.raises(ValueError):
        TaskDAG(tasks, [])
