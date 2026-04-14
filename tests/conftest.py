"""Shared pytest fixtures.

The native ``_ompl_vamp`` extension and the URDFs it depends on are
built by the project's CMake / scikit-build pipeline.  Tests that need
the extension import it lazily through ``planner_factory`` so missing
artefacts surface as a clean *skip* rather than a collection error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def home_joints() -> np.ndarray:
    from autolife_planning.autolife import HOME_JOINTS

    return HOME_JOINTS.copy()


@pytest.fixture(scope="session")
def left_arm_planner():
    """Left-arm planner with a fast RRT-Connect config — shared across tests."""
    pytest.importorskip("autolife_planning._ompl_vamp")
    from autolife_planning.planning import create_planner
    from autolife_planning.types import PlannerConfig

    return create_planner(
        "autolife_left_arm",
        config=PlannerConfig(planner_name="rrtc", time_limit=2.0),
    )


@pytest.fixture(scope="session")
def left_arm_start(left_arm_planner, home_joints) -> np.ndarray:
    return left_arm_planner.extract_config(home_joints)


@pytest.fixture(scope="session")
def table_pointcloud(repo_root) -> np.ndarray:
    """Bundled table.ply rotated and shifted in front of the robot.

    Mirrors ``examples/motion_planning_example.py::load_table`` so the
    motion-planning tests exercise the same geometry the demo uses.
    """
    trimesh = pytest.importorskip("trimesh")
    import autolife_planning

    pkg_root = Path(autolife_planning.__file__).parent
    pcd = trimesh.load(str(pkg_root / "resources" / "envs" / "pcd" / "table.ply"))
    pts = np.asarray(pcd.vertices, dtype=np.float32)
    pts = pts - pts.mean(axis=0)
    rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    pts = pts @ rot.T
    pts[:, 0] += 0.85
    pts[:, 2] += 0.35
    return pts
