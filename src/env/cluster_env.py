"""Gymnasium-style decision-point environment (TZ §5.3, §6.4, Appendix A)."""

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.core.schedule import Schedule
from src.dag_factory.factory import DAGFactory
from src.env.cluster_factory import make_cluster
from src.env.cost_model import energy, exec_time
from src.env.observation import Observation, build_observation
from src.env.placement import ClusterState
from src.utils.config import Config
from src.utils.seeding import make_rng


class ClusterEnv:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._rng = make_rng(config.seed)
        self.state: ClusterState | None = None
        self.schedule: Schedule | None = None
        self.scheduled: set[int] = set()

    @staticmethod
    def _compute_m_ref(dag: TaskDAG, nodes: list[ComputeNode]) -> float:
        """Fastest-exec critical-path lower bound (min exec per task, comm-free)."""

        def node_weight(tid: int) -> float:
            task = dag.task(tid)
            return min(exec_time(task, node) for node in nodes)

        return dag.longest_path_length(node_weight=node_weight, edge_weight=lambda u, v: 0.0)

    @staticmethod
    def _compute_e_ref(dag: TaskDAG, nodes: list[ComputeNode]) -> float:
        """Absolute energy lower bound = sum of per-task minimum energy."""
        total = 0.0
        for tid in range(dag.n_tasks):
            task = dag.task(tid)
            total += min(energy(task, node) for node in nodes)
        return total

    def reset(
        self,
        dag: TaskDAG | None = None,
        nodes: list[ComputeNode] | None = None,
    ) -> tuple[Observation, dict]:
        if dag is None:
            dag = DAGFactory.create(
                "synthetic",
                self._rng,
                n_tasks=self.config.n_tasks,
                n_layers=self.config.n_layers,
                edge_prob=self.config.edge_prob,
                ccr=self.config.ccr,
            )
        if nodes is None:
            nodes = make_cluster(self._rng, self.config.n_nodes, self.config.beta)

        for i, node in enumerate(nodes):
            if node.node_id != i:
                raise ValueError(f"node_id must equal index: nodes[{i}].node_id={node.node_id}")
            node.reset()

        m_ref = self._compute_m_ref(dag, nodes)
        e_ref = self._compute_e_ref(dag, nodes)
        self.state = ClusterState(
            nodes=nodes, dag=dag, task_finish={}, task_node={}, m_ref=m_ref, e_ref=e_ref
        )
        self.schedule = Schedule(n_nodes=len(nodes))
        self.scheduled = set()
        obs = build_observation(self.state, self.scheduled, current_makespan=0.0)
        info = {"m_ref": m_ref, "e_ref": e_ref}
        return obs, info
