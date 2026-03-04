# Examples

## Inverse Kinematics

A minimal IK example without visualization:

```python
import numpy as np

from autolife_planning.config.robot_config import HOME_JOINTS, JOINT_GROUPS
from autolife_planning.kinematics.trac_ik_solver import create_ik_solver
from autolife_planning.types import IKConfig, SE3Pose, SolveType

# Home configuration for the left arm
G = JOINT_GROUPS
HOME_LEFT_ARM = HOME_JOINTS[G["left_arm"]]

# Create solver with custom config
config = IKConfig(
    timeout=0.2,
    epsilon=1e-5,
    solve_type=SolveType.SPEED,
    max_attempts=10,
    position_tolerance=1e-4,
    orientation_tolerance=1e-4,
)

solver = create_ik_solver("left_arm", config=config)

# Forward kinematics at home position
home_pose = solver.fk(HOME_LEFT_ARM)
print(f"Home EE position: {home_pose.position}")

# Solve IK for a target offset from home
target = SE3Pose(
    position=home_pose.position + np.array([0.05, 0.0, -0.05]),
    rotation=home_pose.rotation,
)

result = solver.solve(target, seed=HOME_LEFT_ARM)
if result.success:
    print(f"Solution: {np.round(result.joint_positions, 4)}")
```

### Available Chains

| Chain | DOF | Description |
|-------|-----|-------------|
| `left_arm` | 7 | Shoulder to left wrist |
| `right_arm` | 7 | Shoulder to right wrist |
| `whole_body_left` | 11 | Ground vehicle to left wrist |
| `whole_body_right` | 11 | Ground vehicle to right wrist |
| `whole_body_base_left` | 14 | Zero point to left wrist (includes base) |
| `whole_body_base_right` | 14 | Zero point to right wrist (includes base) |

!!! tip "Shorthand"
    Use `create_ik_solver("whole_body", side="left")` instead of `create_ik_solver("whole_body_left")`.

### Joint Groups

Indices into the full 24-DOF configuration:

| Group | Indices | Joints |
|-------|---------|--------|
| `base` | 0–2 | Virtual_X, Virtual_Y, Virtual_Theta |
| `legs` | 3–4 | Ankle, Knee |
| `waist` | 5–6 | Waist Pitch, Yaw |
| `left_arm` | 7–13 | Shoulder → Wrist (7 DOF) |
| `neck` | 14–16 | Roll, Pitch, Yaw |
| `right_arm` | 17–23 | Shoulder → Wrist (7 DOF) |

## Motion Planning

```python
from autolife_planning.planning.motion_planner import create_planner
from autolife_planning.types import PlannerConfig

config = PlannerConfig()
planner = create_planner("autolife", config=config)

# Plan from start to goal configuration
result = planner.plan(start_config, goal_config)
if result.success:
    print(f"Path with {len(result.path)} waypoints")
```
