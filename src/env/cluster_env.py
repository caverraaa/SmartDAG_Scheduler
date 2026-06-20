"""Gymnasium-style decision-point environment (TZ §5.3, §6.4, Appendix A)."""

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.core.schedule import Assignment, Schedule
from src.dag_factory.factory import DAGFactory
from src.env.cluster_factory import make_cluster
from src.env.cost_model import energy, exec_time
from src.env.observation import Observation, build_observation
from src.env.placement import ClusterState, earliest_start_finish, weighted_cost
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

    def step(self, action: tuple[int, int]) -> tuple[Observation, float, bool, dict]:
        if self.state is None or self.schedule is None:
            raise RuntimeError("Call reset() before step().")
        task_id, node_id = action
        state = self.state
        if not (0 <= node_id < len(state.nodes)) or state.nodes[node_id].node_id != node_id:
            raise ValueError(f"Invalid node_id {node_id}")
        node = state.nodes[node_id]
        if not node.alive:
            raise ValueError(f"Node {node_id} is dead")
        if task_id in self.scheduled:
            raise ValueError(f"Task {task_id} already scheduled")
        if task_id not in set(state.dag.ready_set(self.scheduled)):
            raise ValueError(f"Task {task_id} is not ready")

        task = state.dag.task(task_id)
        components = weighted_cost(task, node, state)
        reward = -(
            self.config.w1 * components.d_makespan_norm + self.config.w2 * components.d_energy_norm
        )

        start, finish = earliest_start_finish(task, node, state)
        step_energy = energy(task, node)
        node.free_at_time = finish
        state.task_finish[task_id] = finish
        state.task_node[task_id] = node_id
        state.sim_time = max(state.sim_time, finish)
        self.schedule.add(Assignment(task_id, node_id, start, finish), energy=step_energy)
        self.scheduled.add(task_id)

        done = len(self.scheduled) == state.dag.n_tasks
        makespan = self.schedule.makespan()
        info: dict = {
            "m_ref": state.m_ref,
            "e_ref": state.e_ref,
            "makespan": makespan,
            "energy": self.schedule.total_energy,
        }
        if done:
            n_alive = sum(1 for n in state.nodes if n.alive)
            balance = self.schedule.load_balance_index(n_alive)
            reward += self.config.w3 * balance
            info["balance"] = balance

        obs = build_observation(state, self.scheduled, current_makespan=makespan)
        return obs, reward, done, info
