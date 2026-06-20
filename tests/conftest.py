import pathlib
import shutil

import pytest

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass

_REPO_ROOT = pathlib.Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def _seed_config_in_tmp(tmp_path: pathlib.Path, request: pytest.FixtureRequest) -> None:
    """Copy the project-root config.yaml into every test's tmp_path.

    Tests that use monkeypatch.chdir(tmp_path) (e.g. test_cli.py) then open
    "config.yaml" relative to that directory and find a valid copy.
    """
    # Only copy when the test actually receives tmp_path; skip session/module scope.
    if "tmp_path" in request.fixturenames:
        src = _REPO_ROOT / "config.yaml"
        shutil.copy(src, tmp_path / "config.yaml")


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
