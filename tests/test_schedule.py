from src.core.schedule import Assignment, Schedule


def test_makespan_and_energy_accumulate() -> None:
    s = Schedule(n_nodes=2)
    s.add(Assignment(0, 0, 0.0, 2.0), energy=200.0)
    s.add(Assignment(1, 0, 2.0, 6.0), energy=400.0)
    assert s.makespan() == 6.0
    assert s.total_energy == 600.0
    assert s.busy_time_by_node() == {0: 6.0}


def test_load_balance_perfectly_even_is_one() -> None:
    s = Schedule(n_nodes=2)
    s.add(Assignment(0, 0, 0.0, 5.0), energy=1.0)
    s.add(Assignment(1, 1, 0.0, 5.0), energy=1.0)
    assert s.load_balance_index(2) == 1.0


def test_load_balance_fully_skewed_is_zero() -> None:
    s = Schedule(n_nodes=2)
    s.add(Assignment(0, 0, 0.0, 10.0), energy=1.0)
    # node 1 idle -> busy times [10, 0] -> CV == 1 -> index 0
    assert s.load_balance_index(2) == 0.0


def test_empty_schedule_makespan_zero() -> None:
    assert Schedule(n_nodes=3).makespan() == 0.0
