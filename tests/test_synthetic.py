import numpy as np

from src.core.dag import TaskDAG
from src.dag_factory.factory import DAGFactory
from src.dag_factory.synthetic import generate_synthetic
from src.utils.seeding import make_rng


def test_generates_valid_acyclic_dag_of_requested_size() -> None:
    dag = generate_synthetic(make_rng(0), n_tasks=30, n_layers=6, edge_prob=0.4, ccr=0.5)
    assert isinstance(dag, TaskDAG)
    assert dag.n_tasks == 30  # construction would raise if cyclic


def test_base_costs_are_heavy_tailed_not_uniform() -> None:
    dag = generate_synthetic(make_rng(1), n_tasks=60, n_layers=8, edge_prob=0.4, ccr=0.5)
    costs = np.array([dag.task(i).base_cost for i in range(dag.n_tasks)])
    # heavy-tailed: coefficient of variation clearly above a uniform spread
    assert costs.std() / costs.mean() > 0.5
    assert costs.min() > 0.0


def test_factory_dispatches_synthetic_and_rejects_unknown() -> None:
    dag = DAGFactory.create(
        "synthetic", make_rng(2), n_tasks=20, n_layers=5, edge_prob=0.4, ccr=0.5
    )
    assert dag.n_tasks == 20
    try:
        DAGFactory.create("does-not-exist", make_rng(0))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_reproducible_with_same_seed() -> None:
    a = generate_synthetic(make_rng(5), n_tasks=25, n_layers=5, edge_prob=0.4, ccr=0.5)
    b = generate_synthetic(make_rng(5), n_tasks=25, n_layers=5, edge_prob=0.4, ccr=0.5)
    assert a.edge_index() == b.edge_index()
    assert [a.task(i).base_cost for i in range(25)] == [b.task(i).base_cost for i in range(25)]
