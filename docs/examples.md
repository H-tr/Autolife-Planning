# Examples

## IK Solver (Minimal)

Solve inverse kinematics without visualization:

```python
import numpy as np

from autolife_planning.config.robot_config import HOME_JOINTS, JOINT_GROUPS
from autolife_planning.kinematics.trac_ik_solver import create_ik_solver
from autolife_planning.types import IKConfig, SE3Pose, SolveType

G = JOINT_GROUPS
HOME_LEFT_ARM = HOME_JOINTS[G["left_arm"]]

config = IKConfig(
    timeout=0.2,              # seconds per TRAC-IK attempt
    epsilon=1e-5,             # convergence tolerance
    solve_type=SolveType.SPEED,  # SPEED | DISTANCE | MANIP1 | MANIP2
    max_attempts=10,          # random restart attempts
    position_tolerance=1e-4,  # post-solve check (meters)
    orientation_tolerance=1e-4,  # post-solve check (radians)
)

solver = create_ik_solver("left_arm", config=config)

# Forward kinematics at home position
home_pose = solver.fk(HOME_LEFT_ARM)

# Solve IK for a target offset from home
target = SE3Pose(
    position=home_pose.position + np.array([0.05, 0.0, -0.05]),
    rotation=home_pose.rotation,
)

result = solver.solve(target, seed=HOME_LEFT_ARM)
if result.success:
    print(f"Solution: {np.round(result.joint_positions, 4)}")

    # Verify with FK
    achieved = solver.fk(result.joint_positions)
    print(f"Achieved position: {np.round(achieved.position, 4)}")
```

## IK Solver with PyBullet Visualization

Interactive IK example that visualizes results in PyBullet. Tests multiple chains (left arm, right arm, whole body) with coordinate frame overlays.

```python
from autolife_planning.config.robot_config import (
    CHAIN_CONFIGS, HOME_JOINTS, JOINT_GROUPS, autolife_robot_config,
)
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.kinematics.trac_ik_solver import create_ik_solver
from autolife_planning.types import IKConfig, SE3Pose, SolveType

env = PyBulletEnv(autolife_robot_config, visualize=True)

config = IKConfig(solve_type=SolveType.DISTANCE)
solver = create_ik_solver("left_arm", config=config)

# Solve and apply to simulation
result = solver.solve(target_pose, seed=HOME_JOINTS[JOINT_GROUPS["left_arm"]])
if result.success:
    env.set_joint_states(result.joint_positions)
```

!!! note
    Requires the `dev` environment: `pixi run -e dev python examples/ik_example_vis.py`

## Motion Planning (Minimal)

Plan a collision-free path between configurations:

```python
import numpy as np

from autolife_planning.config.robot_config import HOME_JOINTS
from autolife_planning.planning import create_planner

# Create a planner with an empty environment (no obstacles)
planner = create_planner("autolife")

start = HOME_JOINTS.copy()
goal = planner.sample_valid()

result = planner.plan(start, goal)
if result.success:
    print(f"Path: {result.path.shape[0]} waypoints, "
          f"{result.planning_time_ns / 1e6:.1f}ms, "
          f"cost: {result.path_cost:.4f}")
```

## Motion Planning with Obstacles

Plan around a point cloud obstacle (e.g., a table) with PyBullet visualization and interactive configuration sampling:

```python
import numpy as np
import trimesh

from autolife_planning.config.robot_config import autolife_robot_config
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import create_planner
from autolife_planning.types import PlannerConfig

# Load point cloud obstacle
table_pcd = trimesh.load("table.ply")
points = np.array(table_pcd.vertices)

# Create planner with obstacle
planner = create_planner(
    "autolife",
    config=PlannerConfig(planner_name="rrtc", point_radius=0.01),
    pointcloud=points,
)

# Visualize
env = PyBulletEnv(autolife_robot_config, visualize=True)
env.add_pointcloud(points)

# Sample valid configs and plan
start = planner.sample_valid()
goal = planner.sample_valid()

result = planner.plan(start, goal)
if result.success:
    for waypoint in result.path:
        env.set_configuration(waypoint)
```

---

## Reference

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

### IK Solve Types

| Type | Description |
|------|-------------|
| `SolveType.SPEED` | Return first valid solution (fastest) |
| `SolveType.DISTANCE` | Minimize joint displacement from seed |
| `SolveType.MANIP1` | Maximize manipulability (product of singular values) |
| `SolveType.MANIP2` | Maximize isotropy (min/max singular value ratio) |
