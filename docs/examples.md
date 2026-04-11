# Examples

The library exposes a small number of factories — `create_ik_solver`,
`create_planner`, `PyBulletEnv` — that cover every workflow below.
Each section pairs a short code snippet with a rendered clip of the
example in action.  All clips come from the offscreen renderer in
`scripts/render_videos/render_docs_videos.py`, so they are exactly
reproducible from a clean checkout.

---

## Inverse Kinematics (TRAC-IK)

TRAC-IK is a fast numerical IK solver — pass a 6-DOF target pose and
a joint seed, get either a closed-form-quality solution or a
`failure` status in under a millisecond.  The factory picks the right
kinematic chain and solve mode from the arguments alone.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/trac_ik.mp4" type="video/mp4">
</video>

```python
import numpy as np

from autolife_planning.config.robot_config import HOME_JOINTS, JOINT_GROUPS
from autolife_planning.kinematics import create_ik_solver
from autolife_planning.types import IKConfig, SE3Pose, SolveType

G = JOINT_GROUPS
HOME_LEFT_ARM = HOME_JOINTS[G["left_arm"]]

config = IKConfig(
    timeout=0.2,                  # seconds per TRAC-IK attempt
    epsilon=1e-5,                 # convergence tolerance
    solve_type=SolveType.DISTANCE,  # SPEED | DISTANCE | MANIP1 | MANIP2
    max_attempts=10,              # random restart attempts
    position_tolerance=1e-4,      # post-solve check (meters)
    orientation_tolerance=1e-4,   # post-solve check (radians)
)

solver = create_ik_solver("left_arm", config=config)

home_pose = solver.fk(HOME_LEFT_ARM)
target = SE3Pose(
    position=home_pose.position + np.array([0.22, 0.15, 0.05]),
    rotation=home_pose.rotation,
)

result = solver.solve(target, seed=HOME_LEFT_ARM)
if result.success:
    print(f"solution: {np.round(result.joint_positions, 4)}")
```

The interactive visualization lives in `examples/ik_example_vis.py`:

```bash
pixi run -e dev python examples/ik_example_vis.py
```

It sweeps the left arm, right arm, and both whole-body chains,
drawing RGB end-effector frames on every solution.  `n` advances to
the next target; `q` quits.

---

## Constrained IK (Pink QP)

The Pink backend solves IK as a weighted QP instead of a closed-form
Jacobian step — which means you can stack secondary objectives like
centre-of-mass stability, camera-frame stabilization, and (optionally)
self-collision avoidance on top of the primary task.  Same
`create_ik_solver` factory, different backend.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/constrained_ik.mp4" type="video/mp4">
</video>

```python
from autolife_planning.config.robot_config import HOME_JOINTS, JOINT_GROUPS
from autolife_planning.kinematics import create_ik_solver
from autolife_planning.types import PinkIKConfig, SE3Pose

G = JOINT_GROUPS
SEED = np.concatenate([
    HOME_JOINTS[G["legs"]],
    HOME_JOINTS[G["waist"]],
    HOME_JOINTS[G["left_arm"]],
])

config = PinkIKConfig(
    lm_damping=1e-3,
    com_cost=0.1,                                    # CoM stability
    camera_frame="Link_Waist_Yaw_to_Shoulder_Inner", # chest camera frame
    camera_cost=0.1,                                 # camera stabilization
    max_iterations=200,
)

solver = create_ik_solver("whole_body", side="left", backend="pink", config=config)

home_pose = solver.fk(SEED)
target = SE3Pose(home_pose.position + np.array([0.30, 0.0, 0.0]), home_pose.rotation)

result = solver.solve_constrained(target, seed=SEED)
if result.success:
    print(f"Solved in {result.iterations} iterations")
    print(f"Position error: {result.position_error * 1000:.2f} mm")
```

### With self-collision avoidance

```python
from autolife_planning.kinematics.collision_model import build_collision_model

collision_ctx = build_collision_model(
    "path/to/autolife_simple.urdf",
    srdf_path="path/to/autolife.srdf",
)

config = PinkIKConfig(
    lm_damping=1e-3,
    com_cost=0.1,
    self_collision=True,
    collision_pairs=5,
    solver="proxqp",
)
solver = create_ik_solver("whole_body", side="left", backend="pink", config=config)
solver.set_collision_context(collision_ctx)
```

```bash
pixi run -e dev python examples/constrained_ik_example_vis.py
```

The clip above sweeps four reach targets — front, high, side, and
low-cross-body.  Throughout, the chest camera link and the robot's
centre of mass are both held at their home values by the secondary
QP tasks, so the whole body coordinates the reach instead of just
the arm.

---

## Motion Planning around Obstacles

The planner factory returns an OMPL-backed geometric planner with a
SIMD collision backend.  Pass a point cloud (or any supported
obstacle primitive) at construction time and the planner will treat
every point as a sphere of ``point_radius`` during edge validation.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/motion_planning.mp4" type="video/mp4">
</video>

```python
import numpy as np
import trimesh

from autolife_planning.config.robot_config import HOME_JOINTS, autolife_robot_config
from autolife_planning.envs.pybullet_env import PyBulletEnv
from autolife_planning.planning import create_planner
from autolife_planning.types import PlannerConfig

cloud = np.asarray(trimesh.load("table.ply").vertices, dtype=np.float32)

planner = create_planner(
    "autolife_left_arm",
    config=PlannerConfig(
        planner_name="rrtc",
        time_limit=2.0,
        point_radius=0.012,
    ),
    base_config=HOME_JOINTS.copy(),
    pointcloud=cloud,
)

start = planner.extract_config(HOME_JOINTS)
goal = planner.sample_valid()

result = planner.plan(start, goal)
if result.success:
    env = PyBulletEnv(autolife_robot_config, visualize=True)
    env.add_pointcloud(cloud)
    env.animate_path(planner.embed_path(result.path))
```

The runnable version lives at `examples/motion_planning_example.py`
and uses the bundled `table.ply` — no downloads needed.

```bash
pixi run -e dev python examples/motion_planning_example.py
pixi run -e dev python examples/motion_planning_example.py --planner_name bitstar --time_limit 3
```

---

## Subgroup Planning

Plan for just the joints you care about — a single arm, both arms,
torso + arm, the whole body, or the 3-DOF virtual base — while the
rest of the robot stays pinned to whatever 24-DOF configuration you
pass in as `base_config`.  The C++ collision checker injects the
inactive joints on every state and edge query, so any plan you get
back is valid for the *whole* robot, not just the subgroup.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/subgroup_planning.mp4" type="video/mp4">
</video>

### Available subgroups

| Category | Subgroup | DOF |
|---|---|---|
| Mobile base | `autolife_base` | 3 |
| Height chain | `autolife_height` | 3 |
| Single arm | `autolife_left_arm`, `autolife_right_arm` | 7 |
| Torso + arm | `autolife_torso_left_arm`, `autolife_torso_right_arm` | 9 |
| Dual arm | `autolife_dual_arm` | 14 |
| Whole body (no base) | `autolife_body` | 21 |

The full-body planner `"autolife"` (24 DOF including the virtual base)
is also available.

### Basic usage

```python
from autolife_planning.config.robot_config import HOME_JOINTS
from autolife_planning.planning import create_planner
from autolife_planning.types import PlannerConfig

# Pin the rest of the body to whatever pose you like — HOME_JOINTS,
# a live robot state, or any custom 24-DOF array.
base_cfg = HOME_JOINTS.copy()

planner = create_planner(
    "autolife_left_arm",
    config=PlannerConfig(planner_name="rrtc"),
    base_config=base_cfg,
)

start = planner.extract_config(base_cfg)
goal = planner.sample_valid()

result = planner.plan(start, goal)
if result.success:
    # Map the subgroup path back to full-body configurations.
    full_path = planner.embed_path(result.path)
```

```bash
pixi run -e dev python examples/subgroup_planning_example.py
pixi run -e dev python examples/subgroup_planning_example.py --planner_name prm
```

The example above iterates through several kinematic subgroups at
three stances (high / mid / low).  Press `space` to play/pause each
clip, `←/→` to step through waypoints, `n` to advance to the next
plan, and close the window to quit.

---

## Constrained Planning (Task-space Manifolds)

A constrained planner treats a user-supplied CasADi residual
`h(q) = 0` as a hard equality and runs OMPL's
`ProjectedStateSpace` — every sampled state is projected onto the
manifold before validation, so the returned path is guaranteed to
satisfy the constraint.  The residual can encode any task-space
equation you can express symbolically: a plane, a rail, an
orientation lock, a contact constraint, a handover coupling.

All five demos live in `examples/constrained_planning/` and share a
tiny scaffold (`_shared.py`) so each file only contains the one
interesting line: the residual itself.  Run any of them with:

```bash
pixi run -e dev python examples/constrained_planning/plane.py
pixi run -e dev python examples/constrained_planning/plane_with_obstacle.py
pixi run -e dev python examples/constrained_planning/line_horizontal.py
pixi run -e dev python examples/constrained_planning/line_vertical.py
pixi run -e dev python examples/constrained_planning/orientation_lock.py
```

### Plane constraint

One holonomic equation: the left gripper's world `z` coordinate is
pinned to its home value.  The 7-DOF arm has a 6-dimensional null
space the planner can exploit; the gripper trajectory is
*guaranteed* flat by construction.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/constrained_plane.mp4" type="video/mp4">
</video>

```python
from autolife_planning.planning import Constraint, SymbolicContext, create_planner

ctx = SymbolicContext("autolife_left_arm")
p0 = ctx.evaluate_link_pose("Link_Left_Gripper", start)[:3, 3]

plane = Constraint(
    residual=ctx.link_translation("Link_Left_Gripper")[2] - float(p0[2]),
    q_sym=ctx.q,
    name="plane_z",
)

planner = create_planner("autolife_left_arm", constraints=[plane])
```

### Plane constraint with obstacle

Same plane manifold as above, plus a red sphere planted on the swept
arc — the classic end-effector-on-a-surface problem.  The planner
curves the arm around the obstacle while the gripper keeps sliding
flat on the plane.  Collision avoidance and equality constraint are
handled by the same OMPL planner.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/constrained_plane_obstacle.mp4" type="video/mp4">
</video>

### Line constraint (horizontal rail)

Stacked residual: two translation equations pin `y` and `z` to their
home values, and six rotation-matrix equations lock the gripper's
orientation.  That leaves `x` as the only free end-effector DOF, and
the gripper slides along the rail without rolling or flipping.  On
the 7-DOF arm the residual has rank 5, so the planner still has a
2-D null space to curl the elbow around the arm's reach limits.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/constrained_line_horizontal.mp4" type="video/mp4">
</video>

```python
import casadi as ca

left_pos = ctx.link_translation("Link_Left_Gripper")
left_rot = ctx.link_rotation("Link_Left_Gripper")

residual = ca.vertcat(
    left_pos[1] - float(p0[1]),
    left_pos[2] - float(p0[2]),
    left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
    left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
)
line = Constraint(residual=residual, q_sym=ctx.q, name="line_h")
```

### Line constraint (vertical rail)

Same idea with the free axis swapped — `x` and `y` are pinned, the
rotation is locked, and only `z` is free.  The gripper slides
cleanly up and down the yellow rail with a frozen orientation.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/constrained_line_vertical.mp4" type="video/mp4">
</video>

### Orientation lock

Six holonomic equations: the first two columns of the gripper's
rotation matrix are pinned to their home values.  (The third column
follows from orthonormality of SO(3), so the entire 3x3 rotation is
locked.)  The residual is rank 3, leaving a 4-DOF null space — enough
for the gripper to translate freely in `x`, `y`, `z` while its
orientation stays frozen.

<video controls autoplay loop muted playsinline width="100%">
  <source src="../assets/constrained_orientation_lock.mp4" type="video/mp4">
</video>

```python
R0 = ctx.evaluate_link_pose("Link_Left_Gripper", start)[:3, :3]
left_rot = ctx.link_rotation("Link_Left_Gripper")

residual = ca.vertcat(
    left_rot[:, 0] - ca.DM(R0[:, 0].tolist()),
    left_rot[:, 1] - ca.DM(R0[:, 1].tolist()),
)
orient = Constraint(residual=residual, q_sym=ctx.q, name="orient_lock")
```

---

## Rendering the videos

Every clip on this page is rendered by one driver script that boots
a headless ``PyBulletEnv`` (``DIRECT`` mode), runs each demo
end-to-end, and pipes offscreen-rendered frames into ``ffmpeg`` via
:class:`autolife_planning.utils.video_recorder.VideoRecorder`.

```bash
# Render everything (≈10 minutes at 1280x720, CPU rendering):
pixi run python scripts/render_videos/render_docs_videos.py

# Render just one clip:
pixi run python scripts/render_videos/render_docs_videos.py --only motion_planning

# Render a subset (note the quotes):
pixi run python scripts/render_videos/render_docs_videos.py --only "plane,line_h,line_v"
```

Output lands in ``docs/assets/*.mp4`` — idempotent, no external
download or GPU required.

---

## Reference

### Kinematic chains

| Chain | DOF | Description |
|-------|-----|-------------|
| `left_arm` | 7 | Shoulder to left wrist |
| `right_arm` | 7 | Shoulder to right wrist |
| `whole_body_left` | 11 | Ground vehicle to left wrist |
| `whole_body_right` | 11 | Ground vehicle to right wrist |
| `whole_body_base_left` | 14 | Zero point to left wrist (includes base) |
| `whole_body_base_right` | 14 | Zero point to right wrist (includes base) |

!!! tip "Shorthand"
    Use `create_ik_solver("whole_body", side="left")` instead of
    `create_ik_solver("whole_body_left")`.

### Joint groups

Indices into the full 24-DOF configuration:

| Group | Indices | Joints |
|-------|---------|--------|
| `base` | 0–2 | Virtual_X, Virtual_Y, Virtual_Theta |
| `legs` | 3–4 | Ankle, Knee |
| `waist` | 5–6 | Waist Pitch, Yaw |
| `left_arm` | 7–13 | Shoulder → Wrist (7 DOF) |
| `neck` | 14–16 | Roll, Pitch, Yaw |
| `right_arm` | 17–23 | Shoulder → Wrist (7 DOF) |

### IK solve types

| Type | Description |
|------|-------------|
| `SolveType.SPEED` | Return first valid solution (fastest) |
| `SolveType.DISTANCE` | Minimize joint displacement from seed |
| `SolveType.MANIP1` | Maximize manipulability (product of singular values) |
| `SolveType.MANIP2` | Maximize isotropy (min/max singular value ratio) |
