from dataclasses import dataclass
from typing import Any


@dataclass
class PlanningContext:
    vamp_module: Any
    planner_func: Any
    plan_settings: Any
    simp_settings: Any
    env: Any
    sampler: Any
    robot_name: str
    planner_name: str
