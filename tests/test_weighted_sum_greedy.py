from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.env.placement import weighted_cost
from src.scheduler.task_scheduler import run_episode
from src.strategies.weighted_sum_greedy import WeightedSumGreedyStrategy
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


def test_greedy_picks_argmin_of_weighted_cost() -> None:
    dag, nodes = _instance()
    env = ClusterEnv(load_config("config.yaml"))
    env.reset(dag=dag, nodes=nodes)
    w1, w2 = 1.0, 0.3
    chosen = WeightedSumGreedyStrategy(w1, w2).predict([0], env.state)
    # brute-force the same objective over (task 0, each node)
    costs = {}
    for node in env.state.nodes:
        c = weighted_cost(env.state.dag.task(0), node, env.state)
        costs[node.node_id] = w1 * c.d_makespan_norm + w2 * c.d_energy_norm
    best_node = min(costs, key=costs.get)
    assert chosen == (0, best_node)


def test_greedy_completes_episode() -> None:
    dag, nodes = _instance()
    env = ClusterEnv(load_config("config.yaml"))
    schedule, info = run_episode(env, WeightedSumGreedyStrategy(1.0, 0.3), dag=dag, nodes=nodes)
    assert len(schedule.assignments) == 4
    assert info["makespan"] > 0.0
