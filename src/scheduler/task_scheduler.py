"""Decision-point scheduler loop driving any strategy through ClusterEnv (TZ §3, §8)."""

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.core.schedule import Schedule
from src.env.cluster_env import ClusterEnv
from src.scheduler.system_monitor import SystemMonitor
from src.strategies.base import BaseSchedulingStrategy


def run_episode(
    env: ClusterEnv,
    strategy: BaseSchedulingStrategy,
    dag: TaskDAG | None = None,
    nodes: list[ComputeNode] | None = None,
    monitor: SystemMonitor | None = None,
) -> tuple[Schedule, dict]:
    """Run one full episode: reset, then assign tasks one decision point at a time.

    Every strategy is driven through the identical ClusterEnv.step, which is the
    fairness invariant (§8). `monitor.check` is invoked at each decision point as
    the uniform Observer trigger (no-op in deterministic M2).
    """
    env.reset(dag=dag, nodes=nodes)
    done = False
    info: dict = {}
    while not done:
        if monitor is not None:
            monitor.check(env.state)
        if not any(n.alive for n in env.state.nodes):
            break  # deadlock: no surviving node can run the remaining tasks
        ready = env.state.dag.ready_set(env.scheduled)
        action = strategy.predict(ready, env.state)
        _, _, done, info = env.step(action)
    assert env.schedule is not None
    return env.schedule, info
