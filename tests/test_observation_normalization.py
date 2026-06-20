import numpy as np

from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.env.observation import (
    N_POWER,
    N_SPEED,
    N_UTIL,
    T_MEM,
    T_OUTDATA,
    T_OUTDEG,
    T_UNSCHED_PREDS,
)
from src.utils.config import load_config


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [
        Task(0, 2.0, 4.0, TaskClass.SEQUENTIAL),
        Task(1, 4.0, 8.0, TaskClass.SEQUENTIAL),
        Task(2, 4.0, 2.0, TaskClass.SEQUENTIAL),
        Task(3, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 300.0, 10.0),
    ]
    return dag, nodes


def test_normalized_columns_are_order_one() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    obs, _ = env.reset(dag=dag, nodes=nodes)
    # mem mean = (4+8+2+1)/4 = 3.75 ; task0 mem 4 -> 4/3.75 ~ 1.07
    assert abs(obs.task_features[0, T_MEM] - (4.0 / 3.75)) < 1e-5
    # out-degree of task0 is 2 ; /n_tasks(4) = 0.5
    assert obs.task_features[0, T_OUTDEG] == 0.5
    # mean power = (100+300)/2 = 200 ; node1 power 300 -> 1.5
    assert abs(obs.node_features[1, N_POWER] - 1.5) < 1e-5
    # speed normalized to ~O(1): mean of N_SPEED column near 1
    assert abs(float(np.mean(obs.node_features[:, N_SPEED])) - 1.0) < 1e-5


def test_unsched_preds_normalized_and_outdata_present() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    obs, _ = env.reset(dag=dag, nodes=nodes)
    # task3 has 2 unscheduled preds at reset -> 2/4 = 0.5
    assert obs.task_features[3, T_UNSCHED_PREDS] == 0.5
    # out_data normalized: column mean ~ 1 over tasks with outgoing edges
    vals = obs.task_features[:, T_OUTDATA]
    assert vals.max() > 0.0


def test_util_clamped_unit_interval() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    obs, _ = env.reset(dag=dag, nodes=nodes)
    col = obs.node_features[:, N_UTIL]
    assert col.min() >= 0.0 and col.max() <= 1.0
