from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.env.observation import T_DONE, T_UNSCHED_PREDS
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


def test_deterministic_episode_is_n_steps() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    env.reset(dag=dag, nodes=nodes)
    order = [(0, 0), (1, 0), (2, 0), (3, 0)]
    done = False
    steps = 0
    for action in order:
        _, _, done, _ = env.step(action)
        steps += 1
    assert steps == 4 and done is True


def test_step_transitions_done_and_unsched_preds() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    env.reset(dag=dag, nodes=nodes)
    obs, _, _, _ = env.step((0, 0))
    # task 0 now done
    assert obs.task_features[0, T_DONE] == 1.0
    # children 1 and 2 had 1 unscheduled pred (task 0) -> now 0
    assert obs.task_features[1, T_UNSCHED_PREDS] == 0.0
    assert obs.task_features[2, T_UNSCHED_PREDS] == 0.0


def test_step_rejects_unready_task() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    env.reset(dag=dag, nodes=nodes)
    try:
        env.step((3, 0))  # task 3 not ready (preds 1,2 unscheduled)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_all_on_cpu_matches_hand_computed_schedule() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _golden_instance()
    env.reset(dag=dag, nodes=nodes)
    info = {}
    for action in [(0, 0), (1, 0), (2, 0), (3, 0)]:
        _, _, done, info = env.step(action)
    # CPU speed 1 -> exec == base_cost; intra-node comm = 0
    # finishes: t0=2, t1=6, t2=10, t3=12 -> makespan 12 ; energy=100*12=1200
    assert info["makespan"] == 12.0
    assert info["energy"] == 1200.0
    assert info["balance"] == 0.0  # node1 idle -> fully skewed
