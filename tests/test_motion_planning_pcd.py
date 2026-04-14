"""Motion planning around the bundled table point cloud.

Mirrors ``examples/motion_planning_example.py`` but headless and
under a tight time budget so it fits CI.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("autolife_planning._ompl_vamp")


def test_plan_around_table(table_pointcloud, home_joints):
    from autolife_planning.config.robot_config import HOME_JOINTS
    from autolife_planning.planning import create_planner
    from autolife_planning.types import PlannerConfig

    planner = create_planner(
        "autolife_left_arm",
        config=PlannerConfig(
            planner_name="rrtc",
            time_limit=2.0,
            point_radius=0.012,
        ),
        base_config=HOME_JOINTS.copy(),
        pointcloud=table_pointcloud,
    )

    start = planner.extract_config(home_joints)
    assert planner.validate(start), "HOME should be collision-free with the table"

    np.random.seed(0)
    goal = planner.sample_valid()
    result = planner.plan(start, goal)

    assert result.status.value in {"success", "failed"}
    if result.success:
        assert result.path is not None
        np.testing.assert_allclose(result.path[0], start, atol=1e-6)
        np.testing.assert_allclose(result.path[-1], goal, atol=1e-6)
        # Every waypoint must remain collision-free.
        for q in result.path:
            assert planner.validate(q)
