"""Task and task-class definitions (TZ §4)."""

from dataclasses import dataclass
from enum import Enum


class TaskClass(Enum):
    DATA_PARALLEL = "data_parallel"
    SEQUENTIAL = "sequential"
    STREAMING = "streaming"


@dataclass(frozen=True)
class Task:
    id: int
    base_cost: float
    mem_required: float
    task_class: TaskClass
