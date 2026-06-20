from src.core.schedule import Assignment, Schedule


def test_balance_over_noncontiguous_alive_ids() -> None:
    # nodes 0 and 2 each busy 5.0; node 1 is dead and absent.
    s = Schedule(n_nodes=3)
    s.add(Assignment(0, 0, 0.0, 5.0), energy=1.0)
    s.add(Assignment(1, 2, 0.0, 5.0), energy=1.0)
    # Over the actual alive ids {0,2} the two are perfectly balanced -> 1.0
    assert s.load_balance_index([0, 2]) == 1.0


def test_dead_midrange_node_excluded() -> None:
    # node 0 busy 10, node 2 idle; alive ids {0,2}. busy=[10,0] -> CV 1 -> 0.0
    s = Schedule(n_nodes=3)
    s.add(Assignment(0, 0, 0.0, 10.0), energy=1.0)
    assert s.load_balance_index([0, 2]) == 0.0


def test_empty_alive_ids_returns_zero() -> None:
    assert Schedule(n_nodes=2).load_balance_index([]) == 0.0
