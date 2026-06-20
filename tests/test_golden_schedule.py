from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.env.cluster_env import ClusterEnv
from src.utils.config import load_config


def test_golden_all_gpu_schedule(
    golden_instance: tuple[TaskDAG, list[ComputeNode]],
) -> None:
    dag, nodes = golden_instance
    env = ClusterEnv(load_config("config.yaml"))
    env.reset(dag=dag, nodes=nodes)
    info: dict = {}
    # All tasks on GPU (node 1, speed 2): exec = base/2 -> t0=1,t1=2,t2=2,t3=1
    # intra-node comm 0; serialized on one node:
    # t0:0-1, t1:1-3, t2:3-5, t3:5-6 -> makespan 6 ; energy = 200*(1+2+2+1)=1200
    for action in [(0, 1), (1, 1), (2, 1), (3, 1)]:
        _, _, done, info = env.step(action)
    assert done is True
    assert info["makespan"] == 6.0
    assert info["energy"] == 1200.0
    assert info["balance"] == 0.0  # only GPU busy, CPU idle


def test_golden_refs_are_stable(
    golden_instance: tuple[TaskDAG, list[ComputeNode]],
) -> None:
    dag, nodes = golden_instance
    env = ClusterEnv(load_config("config.yaml"))
    _, info = env.reset(dag=dag, nodes=nodes)
    assert info["m_ref"] == 4.0
    assert info["e_ref"] == 1200.0
