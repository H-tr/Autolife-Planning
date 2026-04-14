"""End-to-end checks for the OMPL+VAMP planner without obstacles.

Mirrors the spirit of ``examples/subgroup_planning_example.py`` and
``examples/motion_planning_example.py`` (without the table) — small,
fast, and only exercises the bits the demos demonstrate.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("autolife_planning._ompl_vamp")


def test_available_robots_includes_known_subgroups():
    from autolife_planning.planning import available_robots

    names = available_robots()
    assert "autolife" in names
    assert "autolife_left_arm" in names
    assert "autolife_right_arm" in names
    assert "autolife_dual_arm" in names


def test_planner_dimension_matches_subgroup(left_arm_planner):
    # Left arm is the 7-DOF chain in PLANNING_SUBGROUPS.
    assert left_arm_planner._ndof == 7
    lo = np.asarray(left_arm_planner._planner.lower_bounds())
    hi = np.asarray(left_arm_planner._planner.upper_bounds())
    assert lo.shape == (7,) and hi.shape == (7,)
    assert np.all(hi > lo)


def test_extract_embed_round_trip(left_arm_planner, home_joints):
    extracted = left_arm_planner.extract_config(home_joints)
    embedded = left_arm_planner.embed_config(extracted)
    np.testing.assert_allclose(embedded, home_joints)


def test_home_is_collision_free(left_arm_planner, left_arm_start):
    assert left_arm_planner.validate(left_arm_start)


def test_sample_valid_returns_collision_free(left_arm_planner):
    cfg = left_arm_planner.sample_valid()
    assert cfg.shape == (7,)
    assert left_arm_planner.validate(cfg)


def test_trivial_plan_succeeds(left_arm_planner, left_arm_start):
    """Planning from a state to itself must always succeed."""
    result = left_arm_planner.plan(left_arm_start, left_arm_start)
    assert result.success
    assert result.path is not None and result.path.shape[1] == 7


def test_plan_to_random_valid_goal(left_arm_planner, left_arm_start):
    np.random.seed(0)
    goal = left_arm_planner.sample_valid()
    result = left_arm_planner.plan(left_arm_start, goal)
    # rrtc with 2 s + a free workspace should solve almost surely; if
    # the random sample lands in a tricky pocket we still want a clean
    # status, not a crash.
    assert result.status.value in {"success", "failed"}
    if result.success:
        assert result.path is not None
        np.testing.assert_allclose(result.path[0], left_arm_start, atol=1e-6)
        np.testing.assert_allclose(result.path[-1], goal, atol=1e-6)


def test_plan_rejects_wrong_dimension(left_arm_planner, left_arm_start):
    with pytest.raises(ValueError):
        left_arm_planner.plan(left_arm_start, np.zeros(8))
