"""Assignment and Schedule result objects + integral metrics (TZ §4, §5.1)."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Assignment:
    task_id: int
    node_id: int
    start: float
    finish: float


@dataclass
class Schedule:
    n_nodes: int
    assignments: list[Assignment] = field(default_factory=list)
    total_energy: float = 0.0

    def add(self, assignment: Assignment, energy: float) -> None:
        self.assignments.append(assignment)
        self.total_energy += energy

    def makespan(self) -> float:
        return max((a.finish for a in self.assignments), default=0.0)

    def busy_time_by_node(self) -> dict[int, float]:
        busy: dict[int, float] = {}
        for a in self.assignments:
            busy[a.node_id] = busy.get(a.node_id, 0.0) + (a.finish - a.start)
        return busy

    def load_balance_index(self, n_alive_nodes: int) -> float:
        """1 - CV(busy time) over all alive nodes; idle nodes count as 0."""
        if n_alive_nodes <= 0:
            return 0.0
        busy = self.busy_time_by_node()
        times = [busy.get(i, 0.0) for i in range(n_alive_nodes)]
        mean = sum(times) / n_alive_nodes
        if mean == 0.0:
            return 1.0  # nothing scheduled yet: treat as perfectly even
        variance = sum((t - mean) ** 2 for t in times) / n_alive_nodes
        cv = (variance**0.5) / mean
        return max(0.0, min(1.0, 1.0 - cv))
