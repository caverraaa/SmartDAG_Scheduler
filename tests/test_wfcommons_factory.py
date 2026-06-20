# tests/test_wfcommons_factory.py
from pathlib import Path

import numpy as np

from src.dag_factory.factory import DAGFactory

_FIXTURE = str(Path(__file__).parent / "fixtures" / "wfformat_tiny.json")


def test_load_from_wfcommons_reads_file() -> None:
    dag = DAGFactory.load_from_wfcommons(_FIXTURE, np.random.default_rng(0))
    assert dag.n_tasks == 4
    assert sorted(dag.edge_index()) == [(0, 1), (0, 2), (1, 3), (2, 3)]


def test_create_source_wfcommons_delegates() -> None:
    dag = DAGFactory.create("wfcommons", np.random.default_rng(0), path=_FIXTURE)
    assert dag.n_tasks == 4
