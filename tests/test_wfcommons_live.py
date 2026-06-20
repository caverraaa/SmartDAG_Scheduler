"""Live wfcommons integration — deselected by default (run: pytest -m wfcommons).

Catches wfcommons API/schema drift by generating a real workflow and feeding it
through the parser. Requires `pip install wfcommons`.
"""

import numpy as np
import pytest


@pytest.mark.wfcommons
def test_live_recipe_parses(tmp_path) -> None:
    pytest.importorskip("wfcommons")
    from tools.gen_wfcommons import generate

    path = generate(recipe="montage", n_tasks=20, seed=42, out_dir=str(tmp_path))

    from src.dag_factory.factory import DAGFactory

    dag = DAGFactory.load_from_wfcommons(path, np.random.default_rng(0), recipe="montage")
    assert dag.n_tasks >= 1
    assert dag.critical_path_length() > 0.0
