"""Soft-cost planning — mirrors ``examples/cost_planning``.

Only checks that the cost JIT-compiles, the planner accepts it, and
the resulting RRT* run completes (success or clean failure — both are
fine; we're not benchmarking).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("autolife_planning._ompl_vamp")
ca = pytest.importorskip("casadi")
pytest.importorskip("pinocchio")


@pytest.fixture(scope="module", autouse=True)
def _isolated_cost_cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cost_cache")
    old = os.environ.get("AUTOLIFE_COST_CACHE_DIR")
    os.environ["AUTOLIFE_COST_CACHE_DIR"] = str(cache)
    yield
    if old is None:
        os.environ.pop("AUTOLIFE_COST_CACHE_DIR", None)
    else:
        os.environ["AUTOLIFE_COST_CACHE_DIR"] = old


SUBGROUP = "autolife_left_arm"
LEFT_FINGER_LINK = "Link_Left_Gripper_Left_Finger"
RIGHT_FINGER_LINK = "Link_Left_Gripper_Right_Finger"


def _build_height_cost():
    from autolife_planning.config.robot_config import HOME_JOINTS
    from autolife_planning.planning import Cost, SymbolicContext

    ctx = SymbolicContext(SUBGROUP)
    start = HOME_JOINTS[ctx.active_indices].copy()
    tcp = 0.5 * (
        ctx.link_translation(LEFT_FINGER_LINK) + ctx.link_translation(RIGHT_FINGER_LINK)
    )
    p0 = 0.5 * (
        np.asarray(ctx.evaluate_link_pose(LEFT_FINGER_LINK, start))[:3, 3]
        + np.asarray(ctx.evaluate_link_pose(RIGHT_FINGER_LINK, start))[:3, 3]
    )
    residual = tcp[2] - float(p0[2])
    cost = Cost(
        expression=ca.sumsqr(residual),
        q_sym=ctx.q,
        name="height_test",
        weight=10.0,
    )
    return ctx, start, cost


def test_cost_compiles():
    _ctx, _start, cost = _build_height_cost()
    assert cost.so_path.exists()
    assert cost.ambient_dim == 7


def test_cost_rejects_non_scalar_expression():
    from autolife_planning.planning import Cost

    q = ca.SX.sym("q", 7)
    with pytest.raises(ValueError):
        Cost(expression=ca.vertcat(q[0], q[1]), q_sym=q, name="bad")


def test_cost_rejects_negative_weight():
    from autolife_planning.planning import Cost

    q = ca.SX.sym("q", 7)
    with pytest.raises(ValueError):
        Cost(expression=q[0] * q[0], q_sym=q, name="bad", weight=-1.0)


def test_planner_accepts_cost_and_runs():
    from autolife_planning.planning import create_planner
    from autolife_planning.types import PlannerConfig

    _ctx, start, cost = _build_height_cost()
    planner = create_planner(
        SUBGROUP,
        config=PlannerConfig(
            planner_name="rrtstar",
            time_limit=1.5,
            simplify=False,
        ),
        costs=[cost],
    )
    assert planner.validate(start)
    # Trivial start==goal plan must succeed regardless of the cost.
    result = planner.plan(start, start)
    assert result.success
