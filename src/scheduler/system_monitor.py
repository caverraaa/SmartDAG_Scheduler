"""SystemMonitor: the Observer that detects failure/overload events (TZ §3, §8).

In deterministic M2 mode it is idle (no events). M4 will make `check` emit
failure events here, so that all strategies react through the SAME trigger
(the fairness invariant). It is wired into the scheduler loop now to establish
that single trigger point.
"""

from collections.abc import Callable

from src.env.placement import ClusterState


class SystemMonitor:
    def __init__(self) -> None:
        self._subscribers: list[Callable[[ClusterState], None]] = []

    def subscribe(self, callback: Callable[[ClusterState], None]) -> None:
        self._subscribers.append(callback)

    def check(self, state: ClusterState) -> list:
        """Return the events fired at this decision point. Empty in M2."""
        return []
