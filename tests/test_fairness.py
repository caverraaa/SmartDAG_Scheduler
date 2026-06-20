from src.core.compute_node import ComputeNode, NodeType
from src.core.dag import TaskDAG
from src.core.task import Task, TaskClass
from src.env.cluster_env import ClusterEnv
from src.env.placement import weighted_cost
from src.scheduler.task_scheduler import run_episode
from src.strategies.cpop import CPOPStrategy
from src.strategies.heft import HEFTStrategy
from src.strategies.min_min import MinMinStrategy
from src.strategies.random_strategy import RandomStrategy
from src.strategies.weighted_sum_greedy import WeightedSumGreedyStrategy
from src.utils.config import load_config
from src.utils.seeding import make_rng


def _instance() -> tuple[TaskDAG, list[ComputeNode]]:
    tasks = [
        Task(0, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(1, 6.0, 1.0, TaskClass.SEQUENTIAL),
        Task(2, 2.0, 1.0, TaskClass.SEQUENTIAL),
        Task(3, 2.0, 1.0, TaskClass.SEQUENTIAL),
    ]
    dag = TaskDAG(tasks, [(0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0)])
    nodes = [
        ComputeNode(0, NodeType.CPU, {tc: 1.0 for tc in TaskClass}, 100.0, 10.0),
        ComputeNode(1, NodeType.GPU, {tc: 2.0 for tc in TaskClass}, 200.0, 10.0),
    ]
    return dag, nodes


def _all_strategies() -> list:
    cfg = load_config("config.yaml")
    return [
        HEFTStrategy(),
        CPOPStrategy(),
        MinMinStrategy(),
        WeightedSumGreedyStrategy(cfg.w1, cfg.w2),
        RandomStrategy(make_rng(0)),
    ]


def test_all_strategies_produce_valid_complete_schedules_on_same_instance() -> None:
    dag, nodes = _instance()
    for strategy in _all_strategies():
        env = ClusterEnv(load_config("config.yaml"))
        schedule, info = run_episode(env, strategy, dag=dag, nodes=nodes)
        assert sorted(a.task_id for a in schedule.assignments) == [0, 1, 2, 3], strategy
        assert info["makespan"] > 0.0


def test_greedy_choice_matches_env_reward_objective() -> None:
    # The action the greedy picks must be the global argmin of the SAME objective
    # the env reward negates; on a non-terminal step reward == -(w1*dmk + w2*den).
    dag, nodes = _instance()
    cfg = load_config("config.yaml")
    env = ClusterEnv(load_config("config.yaml"))
    env.reset(dag=dag, nodes=nodes)

    greedy = WeightedSumGreedyStrategy(cfg.w1, cfg.w2)
    ready = env.state.dag.ready_set(env.scheduled)
    task_id, node_id = greedy.predict(ready, env.state)

    # objective the greedy minimised for its choice
    comp = weighted_cost(env.state.dag.task(task_id), env.state.nodes[node_id], env.state)
    greedy_obj = cfg.w1 * comp.d_makespan_norm + cfg.w2 * comp.d_energy_norm

    # it must be the global minimum over all (ready, alive node) pairs
    all_objs = []
    for t in ready:
        for n in env.state.nodes:
            c = weighted_cost(env.state.dag.task(t), n, env.state)
            all_objs.append(cfg.w1 * c.d_makespan_norm + cfg.w2 * c.d_energy_norm)
    assert greedy_obj == min(all_objs)

    # stepping that action yields reward == -greedy_obj (first step is non-terminal)
    _, reward, done, _ = env.step((task_id, node_id))
    assert done is False
    assert abs(reward - (-greedy_obj)) < 1e-9


def test_same_instance_gives_identical_inputs_across_strategies() -> None:
    # Fairness: the env builds the identical first decision-point state regardless
    # of strategy, because dag+nodes are the same objects and reset is deterministic.
    dag, nodes = _instance()
    env_a = ClusterEnv(load_config("config.yaml"))
    env_b = ClusterEnv(load_config("config.yaml"))
    _, info_a = env_a.reset(dag=dag, nodes=nodes)
    _, info_b = env_b.reset(dag=dag, nodes=nodes)
    assert info_a["m_ref"] == info_b["m_ref"]
    assert info_a["e_ref"] == info_b["e_ref"]
    assert env_a.state.dag.ready_set(env_a.scheduled) == env_b.state.dag.ready_set(env_b.scheduled)
