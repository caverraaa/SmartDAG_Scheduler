"""Compute node definitions (TZ §4)."""

from dataclasses import dataclass
from enum import Enum

from src.core.task import TaskClass


class NodeType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    FPGA = "fpga"
    TPU = "tpu"


@dataclass
class ComputeNode:
    node_id: int
    node_type: NodeType
    speed_by_class: dict[TaskClass, float]
    power_w: float
    bandwidth: float
    free_at_time: float = 0.0
    alive: bool = True

    def speed(self, task_class: TaskClass) -> float:
        return self.speed_by_class[task_class]

    def reset(self) -> None:
        """Return the node to its initial idle, alive state."""
        self.free_at_time = 0.0
        self.alive = True
