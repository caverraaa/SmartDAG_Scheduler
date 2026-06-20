# tests/test_wfcommons.py
import json
import sys
from pathlib import Path

import numpy as np

from src.core.task import TaskClass
from src.dag_factory.wfcommons_adapter import parse_wfformat
from src.dag_factory.wfcommons_config import WfcommonsParams

_FIXTURE = Path(__file__).parent / "fixtures" / "wfformat_tiny.json"


def _doc() -> dict:
    with open(_FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


def _params() -> WfcommonsParams:
    return WfcommonsParams(
        default_mem=4.0,
        eps=0.01,
        bytes_to_unit=1.0,
        mem_min=1.0,
        mem_max=8.0,
        memory_ref_bytes=8e9,
    )


def test_parser_does_not_import_wfcommons() -> None:
    parse_wfformat(_doc(), np.random.default_rng(0), _params())
    assert "wfcommons" not in sys.modules


def test_indices_topological_tiebreak_by_id() -> None:
    dag = parse_wfformat(_doc(), np.random.default_rng(0), _params())
    assert dag.n_tasks == 4
    # base_cost in index order proves the mapping task-1->0 ... task-4->3
    assert [dag.task(i).base_cost for i in range(4)] == [5.0, 3.0, 4.0, 2.0]


def test_task_class_from_curated_montage_table() -> None:
    dag = parse_wfformat(_doc(), np.random.default_rng(0), _params())
    assert dag.task(0).task_class is TaskClass.DATA_PARALLEL  # mProject
    assert dag.task(1).task_class is TaskClass.STREAMING  # mDiffFit
    assert dag.task(2).task_class is TaskClass.STREAMING  # mDiffFit
    assert dag.task(3).task_class is TaskClass.SEQUENTIAL  # mConcatFit


def test_mem_required_derive_and_default() -> None:
    dag = parse_wfformat(_doc(), np.random.default_rng(0), _params())
    assert dag.task(0).mem_required == 8.0  # 8e9 -> mem_max
    assert dag.task(1).mem_required == 4.0  # absent -> default
    assert dag.task(2).mem_required == 4.5  # 4e9 -> midpoint
    assert dag.task(3).mem_required == 1.0  # 0 -> mem_min


def test_edges_and_volumes_from_shared_files_with_eps() -> None:
    dag = parse_wfformat(_doc(), np.random.default_rng(0), _params())
    assert sorted(dag.edge_index()) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert dag.edge_data(0, 1) == 100.0  # f_ra
    assert dag.edge_data(0, 2) == 0.01  # no shared file -> eps
    assert dag.edge_data(1, 3) == 200.0  # f_as
    assert dag.edge_data(2, 3) == 50.0  # f_bs


def test_result_satisfies_taskdag_invariants() -> None:
    dag = parse_wfformat(_doc(), np.random.default_rng(0), _params())
    # node_id == index over 0..N-1 and acyclic are enforced by TaskDAG.__init__;
    # critical_path_length being finite proves a valid DAG was built.
    assert dag.critical_path_length() > 0.0
