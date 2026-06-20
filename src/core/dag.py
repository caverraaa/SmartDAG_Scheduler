"""TaskDAG: a typed wrapper over networkx.DiGraph (TZ §4)."""

from collections.abc import Callable

import networkx as nx

from src.core.task import Task


class TaskDAG:
    def __init__(self, tasks: list[Task], edges: list[tuple[int, int, float]]) -> None:
        ids = [t.id for t in tasks]
        if sorted(ids) != list(range(len(tasks))):
            raise ValueError("Task ids must be exactly 0..N-1 (node_id == index).")
        self._g: nx.DiGraph = nx.DiGraph()
        for t in tasks:
            self._g.add_node(t.id, task=t)
        for src, dst, data in edges:
            self._g.add_edge(src, dst, data=float(data))
        if not nx.is_directed_acyclic_graph(self._g):
            raise ValueError("TaskDAG must be acyclic.")
        self._b_level: dict[int, float] = {}
        self._t_level: dict[int, float] = {}
        self._compute_levels()

    @property
    def n_tasks(self) -> int:
        return self._g.number_of_nodes()

    def task(self, tid: int) -> Task:
        return self._g.nodes[tid]["task"]

    def predecessors(self, tid: int) -> list[int]:
        return sorted(self._g.predecessors(tid))

    def successors(self, tid: int) -> list[int]:
        return sorted(self._g.successors(tid))

    def out_degree(self, tid: int) -> int:
        return self._g.out_degree(tid)

    def out_data(self, tid: int) -> float:
        return float(sum(self._g.edges[tid, s]["data"] for s in self._g.successors(tid)))

    def edge_data(self, src: int, dst: int) -> float:
        return float(self._g.edges[src, dst]["data"])

    def ready_set(self, scheduled: set[int]) -> list[int]:
        return [
            n
            for n in sorted(self._g.nodes)
            if n not in scheduled and all(p in scheduled for p in self._g.predecessors(n))
        ]

    def longest_path_length(
        self,
        node_weight: Callable[[int], float],
        edge_weight: Callable[[int, int], float],
    ) -> float:
        dist: dict[int, float] = {}
        for n in nx.topological_sort(self._g):
            best_pred = 0.0
            for p in self._g.predecessors(n):
                best_pred = max(best_pred, dist[p] + edge_weight(p, n))
            dist[n] = best_pred + node_weight(n)
        return max(dist.values()) if dist else 0.0

    def _compute_levels(self) -> None:
        topo = list(nx.topological_sort(self._g))

        # t_level: longest path from a source to (excluding) the node, base_cost weighted.
        for n in topo:
            preds = list(self._g.predecessors(n))
            self._t_level[n] = (
                0.0 if not preds else max(self._t_level[p] + self.task(p).base_cost for p in preds)
            )

        # b_level: longest path from the node (inclusive) to a sink, base_cost weighted.
        for n in reversed(topo):
            succ = list(self._g.successors(n))
            downstream = 0.0 if not succ else max(self._b_level[s] for s in succ)
            self._b_level[n] = self.task(n).base_cost + downstream

    def b_level(self, tid: int) -> float:
        return self._b_level[tid]

    def t_level(self, tid: int) -> float:
        return self._t_level[tid]

    def critical_path_length(self) -> float:
        return max(self._b_level.values()) if self._b_level else 0.0

    def edge_index(self) -> list[tuple[int, int]]:
        return [(int(u), int(v)) for u, v in self._g.edges]
