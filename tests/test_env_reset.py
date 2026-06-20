from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.utils.config import load_config


def _golden_instance() -> tuple[TaskDAG, list[ComputeNode]]:
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


def test_reset_emits_refs_in_info_not_obs() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    obs, info = env.reset(dag=dag, nodes=nodes)
    assert "m_ref" in info and "e_ref" in info
    # M_ref = fastest-exec critical path: min exec per task (GPU speed 2) on path 0->1->3
    # = 1 + 2 + 1 = 4
    assert info["m_ref"] == 4.0
    # E_ref = sum min energy per task; CPU=100*c, GPU=200*(c/2)=100*c -> equal -> 100*(2+4+4+2)
    assert info["e_ref"] == 1200.0


def test_reset_enforces_node_id_equals_index() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    nodes[1].node_id = 7  # break the invariant
    try:
        env.reset(dag=dag, nodes=nodes)
        raise AssertionError("expected ValueError for node_id != index")
    except ValueError:
        pass


def test_reset_samples_from_config_when_no_override() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    obs, info = env.reset()
    assert obs.task_features.shape[0] == 30  # n_tasks from config
    assert obs.node_features.shape[0] == 8  # n_nodes from config


def test_reset_guards_against_degenerate_e_ref() -> None:
    """Verify reset() raises ValueError if e_ref would be zero or negative."""
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    # Set all nodes' power_w to 0.0 so every task's minimum energy is 0 -> e_ref == 0
    nodes[0].power_w = 0.0
    nodes[1].power_w = 0.0
    try:
        env.reset(dag=dag, nodes=nodes)
        raise AssertionError("expected ValueError for degenerate e_ref == 0")
    except ValueError as e:
        assert "Degenerate instance" in str(e)
        assert "e_ref=0.0" in str(e)
