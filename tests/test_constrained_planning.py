"""Constraint-based (manifold) planning — mirrors ``examples/constrained_planning``.

Builds the same horizontal-line residual the gallery uses, hands it
to the planner, and verifies the planner accepts the manifold and
keeps validating the seed configuration.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("autolife_planning._ompl_vamp")
ca = pytest.importorskip("casadi")
pytest.importorskip("pinocchio")


@pytest.fixture(scope="module", autouse=True)
def _isolated_constraint_cache(tmp_path_factory):
    """Use a fresh on-disk cache per test session.

    The default cache lives under ``~/.cache/autolife_planning/`` and
    pollutes the user's machine; CI runners get a clean dir.
    """
    cache = tmp_path_factory.mktemp("constraint_cache")
    old = os.environ.get("AUTOLIFE_CONSTRAINT_CACHE_DIR")
    os.environ["AUTOLIFE_CONSTRAINT_CACHE_DIR"] = str(cache)
    yield
    if old is None:
        os.environ.pop("AUTOLIFE_CONSTRAINT_CACHE_DIR", None)
    else:
        os.environ["AUTOLIFE_CONSTRAINT_CACHE_DIR"] = old


SUBGROUP = "autolife_left_arm"
EE_LINK = "Link_Left_Gripper"
LEFT_FINGER_LINK = "Link_Left_Gripper_Left_Finger"
RIGHT_FINGER_LINK = "Link_Left_Gripper_Right_Finger"


def _build_horizontal_line_constraint():
    from autolife_planning.config.robot_config import HOME_JOINTS
    from autolife_planning.planning import Constraint, SymbolicContext

    ctx = SymbolicContext(SUBGROUP)
    start = HOME_JOINTS[ctx.active_indices].copy()

    tcp = 0.5 * (
        ctx.link_translation(LEFT_FINGER_LINK) + ctx.link_translation(RIGHT_FINGER_LINK)
    )
    R = ctx.evaluate_link_pose(EE_LINK, start)[:3, :3]
    p0 = 0.5 * (
        np.asarray(ctx.evaluate_link_pose(LEFT_FINGER_LINK, start))[:3, 3]
        + np.asarray(ctx.evaluate_link_pose(RIGHT_FINGER_LINK, start))[:3, 3]
    )
    left_rot = ctx.link_rotation(EE_LINK)

    residual = ca.vertcat(
        tcp[1] - float(p0[1]),
        tcp[2] - float(p0[2]),
        left_rot[:, 0] - ca.DM(R[:, 0].tolist()),
        left_rot[:, 1] - ca.DM(R[:, 1].tolist()),
    )
    constraint = Constraint(residual=residual, q_sym=ctx.q, name="line_h_test")
    return ctx, start, constraint


def test_symbolic_context_dimensions():
    from autolife_planning.planning import SymbolicContext

    ctx = SymbolicContext(SUBGROUP)
    assert len(ctx.active_indices) == 7
    assert ctx.q.numel() == 7


def test_constraint_compiles():
    _ctx, _start, c = _build_horizontal_line_constraint()
    assert c.so_path.exists(), "Constraint .so should be JIT-compiled and cached"
    assert c.ambient_dim == 7
    assert c.co_dim == 8  # 2 scalar + 2 columns of 3 = 8


def test_planner_accepts_constraint_and_validates_start():
    from autolife_planning.planning import create_planner
    from autolife_planning.types import PlannerConfig

    _ctx, start, constraint = _build_horizontal_line_constraint()
    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(planner_name="rrtc", time_limit=1.0),
        constraints=[constraint],
    )
    # The HOME pose is on the manifold by construction (the residual was
    # built from it), so validate must agree.
    assert planner.validate(start)


def test_constraint_rejects_non_sx_q_sym():
    from autolife_planning.planning import Constraint

    with pytest.raises(TypeError):
        Constraint(residual=ca.DM(0.0), q_sym="not a SX")
