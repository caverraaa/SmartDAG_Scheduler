from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.env.placement import weighted_cost
from src.utils.config import load_config


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
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


def test_makespan_and_energy_telescope_to_totals() -> None:
    env = ClusterEnv(load_config("config.yaml"))
    dag, nodes = _instance()
    obs, info = env.reset(dag=dag, nodes=nodes)
    m_ref, e_ref = info["m_ref"], info["e_ref"]

    sum_makespan = 0.0
    sum_energy = 0.0
    actions = [(0, 1), (1, 0), (2, 1), (3, 0)]  # mixed nodes
    final_info: dict = {}
    for action in actions:
        task = env.state.dag.task(action[0])
        node = env.state.nodes[action[1]]
        comp = weighted_cost(task, node, env.state)  # measure BEFORE applying
        sum_makespan += comp.d_makespan_norm * m_ref
        sum_energy += comp.d_energy_norm * e_ref
        _, _, _, final_info = env.step(action)

    assert abs(sum_makespan - final_info["makespan"]) < 1e-6
    assert abs(sum_energy - final_info["energy"]) < 1e-6
