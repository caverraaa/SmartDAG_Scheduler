"""SystemMonitor: the Observer that detects failure events (TZ §3, §8).

check() reports the nodes that have newly died since the previous call, so every
strategy reacts through the same uniform trigger (the fairness invariant).
"""

from collections.abc import Callable

from src.env.placement import ClusterState


class SystemMonitor:
    def __init__(self) -> None:
        self._subscribers: list[Callable[[ClusterState], None]] = []
        self._seen_dead: set[int] = set()

    def subscribe(self, callback: Callable[[ClusterState], None]) -> None:
        self._subscribers.append(callback)

    def check(self, state: ClusterState) -> list[int]:
        """Return the node ids that newly died since the last check; notify subscribers."""
        dead = {n.node_id for n in state.nodes if not n.alive}
        new_failures = sorted(dead - self._seen_dead)
        self._seen_dead = dead
        for _nid in new_failures:
            for callback in self._subscribers:
                callback(state)
        return new_failures
