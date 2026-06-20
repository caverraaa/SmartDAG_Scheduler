"""Gymnasium-style decision-point environment (TZ §5.3, §6.4, Appendix A)."""

import zlib

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.core.schedule import Assignment, Schedule
from src.dag_factory.factory import DAGFactory
from src.env.cluster_factory import make_cluster
from src.env.cost_model import energy, exec_time
from src.env.observation import Observation, build_observation
from src.env.placement import ClusterState, earliest_start_finish, weighted_cost
from src.utils.config import Config
from src.utils.seeding import derive_rng, make_rng


class ClusterEnv:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._rng = make_rng(config.seed)
        self.state: ClusterState | None = None
        self.schedule: Schedule | None = None
        self.scheduled: set[int] = set()

    @staticmethod
    def _instance_signature(dag: TaskDAG, nodes: list[ComputeNode]) -> int:
        """Deterministic crc32 over the instance, so the calendar is keyed to it."""
        parts: list[str] = []
        for tid in range(dag.n_tasks):
            t = dag.task(tid)
            parts.append(f"t{tid}:{t.base_cost!r}:{t.mem_required!r}:{t.task_class.value}")
        for u, v in dag.edge_index():
            parts.append(f"e{u}-{v}:{dag.edge_data(u, v)!r}")
        for n in nodes:
            speeds = ",".join(
                f"{c.value}={n.speed_by_class[c]!r}"
                for c in sorted(n.speed_by_class, key=lambda c: c.value)
            )
            parts.append(f"n{n.node_id}:{n.node_type.value}:{n.power_w!r}:{n.bandwidth!r}:{speeds}")
        return zlib.crc32("|".join(parts).encode("utf-8"))

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
        if m_ref <= 0.0 or e_ref <= 0.0:
            raise ValueError(
                f"Degenerate instance: m_ref={m_ref}, e_ref={e_ref} must both be > 0 "
                f"(zero would make the normalized reward divide by zero)."
            )
        sig = self._instance_signature(dag, nodes)
        noise_rng = derive_rng(self.config.seed, f"noise|{sig}")
        fail_rng = derive_rng(self.config.seed, f"failure|{sig}")
        if self.config.noise_std > 0.0:
            noise_eps = {
                tid: float(noise_rng.normal(0.0, self.config.noise_std))
                for tid in range(dag.n_tasks)
            }
        else:
            noise_eps = {tid: 0.0 for tid in range(dag.n_tasks)}
        if self.config.failure_rate > 0.0:
            scale = 1.0 / self.config.failure_rate
            failure_times = {n.node_id: float(fail_rng.exponential(scale)) for n in nodes}
        else:
            failure_times = {n.node_id: float("inf") for n in nodes}

        self.state = ClusterState(
            nodes=nodes,
            dag=dag,
            task_finish={},
            task_node={},
            m_ref=m_ref,
            e_ref=e_ref,
            failure_times=failure_times,
            noise_eps=noise_eps,
        )
        self.schedule = Schedule(n_nodes=len(nodes))
        self.scheduled = set()
        obs = build_observation(self.state, self.scheduled, current_makespan=0.0)
        info = {"m_ref": m_ref, "e_ref": e_ref}
        return obs, info

    def step(self, action: tuple[int, int]) -> tuple[Observation, float, bool, dict]:
        """One assignment. Returns the 4-tuple (obs, reward, done, info) per spec §5.3.

        Planning (reward, weighted_cost, observation) is on NOMINAL costs; the actual
        noisy duration is revealed only at commit. A placement fails iff its actual
        finish exceeds the node's exogenous failure time t_f (the node then dies and the
        task is requeued, reward 0). done=True at full completion or deadlock.
        """
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
        components = weighted_cost(task, node, state)  # NOMINAL planning
        start, nominal_finish = earliest_start_finish(task, node, state)
        eps = state.noise_eps.get(task_id, 0.0)
        actual_exec = max(0.0, (nominal_finish - start) * (1.0 + eps))
        actual_finish = start + actual_exec

        t_f = state.failure_times.get(node_id, float("inf"))
        if actual_finish > t_f:
            # Node dies before this task could finish: task lost (requeued), nothing committed.
            node.alive = False
            remaining = state.dag.n_tasks - len(self.scheduled)
            deadlocked = remaining > 0 and not any(n.alive for n in state.nodes)
            makespan = self.schedule.makespan()
            info: dict = {
                "m_ref": state.m_ref,
                "e_ref": state.e_ref,
                "makespan": makespan,
                "energy": self.schedule.total_energy,
                "failed_node": node_id,
                "deadlocked": deadlocked,
            }
            if deadlocked:
                alive_ids = [n.node_id for n in state.nodes if n.alive]
                info["balance"] = self.schedule.load_balance_index(alive_ids)
            obs = build_observation(state, self.scheduled, current_makespan=makespan)
            return obs, 0.0, deadlocked, info

        # Success: reward is nominal; commit uses the actual (noisy) finish/energy.
        reward = -(
            self.config.w1 * components.d_makespan_norm + self.config.w2 * components.d_energy_norm
        )
        actual_energy = node.power_w * actual_exec
        node.free_at_time = actual_finish
        state.task_finish[task_id] = actual_finish
        state.task_node[task_id] = node_id
        state.sim_time = max(state.sim_time, actual_finish)
        self.schedule.add(Assignment(task_id, node_id, start, actual_finish), energy=actual_energy)
        self.scheduled.add(task_id)

        done = len(self.scheduled) == state.dag.n_tasks
        makespan = self.schedule.makespan()
        info = {
            "m_ref": state.m_ref,
            "e_ref": state.e_ref,
            "makespan": makespan,
            "energy": self.schedule.total_energy,
        }
        if done:
            alive_ids = [n.node_id for n in state.nodes if n.alive]
            balance = self.schedule.load_balance_index(alive_ids)
            reward += self.config.w3 * balance
            info["balance"] = balance

        obs = build_observation(state, self.scheduled, current_makespan=makespan)
        return obs, reward, done, info
