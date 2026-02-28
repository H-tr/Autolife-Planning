from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PlanningStatus(Enum):
    """Status of a motion planning attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"


@dataclass
class PlannerConfig:
    """Configuration parameters for the motion planner."""

    planner_name: str = "rrtc"
    max_iterations: int = 1_000_000
    point_radius: float = 0.01
    simplify: bool = True
    interpolate: bool = True

    def __post_init__(self):
        valid_planners = ("rrtc", "prm", "fcit", "aorrtc")
        if self.planner_name not in valid_planners:
            raise ValueError(
                f"Unknown planner '{self.planner_name}'. "
                f"Supported: {', '.join(valid_planners)}"
            )
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.point_radius <= 0:
            raise ValueError("point_radius must be > 0")


@dataclass
class PlanningResult:
    """Result of a motion planning attempt."""

    status: PlanningStatus
    path: np.ndarray | None
    planning_time_ns: int
    iterations: int
    path_cost: float

    @property
    def success(self) -> bool:
        return self.status == PlanningStatus.SUCCESS
