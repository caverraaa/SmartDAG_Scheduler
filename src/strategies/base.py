"""Strategy interface shared by every scheduler (heuristic + RL) (TZ §3, §8)."""

from abc import ABC, abstractmethod

from src.env.placement import ClusterState


class BaseSchedulingStrategy(ABC):
    """A scheduling policy: choose one (task, node) at a decision point."""

    @abstractmethod
    def predict(self, ready: list[int], state: ClusterState) -> tuple[int, int]:
        """Return (task_id, node_id) for the next assignment.

        `ready` is the list of ready (unscheduled, all-predecessors-done) task
        ids; `state` is the live ClusterState. The returned node_id must be the
        index of an alive node.
        """
        raise NotImplementedError
