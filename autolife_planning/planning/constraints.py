"""Holonomic constraint factories for the OMPL + VAMP planner.

Two parameterised constraint kinds, both consumed by
:class:`autolife_planning.planning.MotionPlanner` via its ``constraints``
keyword.  Each one is a small dataclass that the planner translates into
a C++ ``ompl::base::Constraint`` and feeds to ``ProjectedStateSpace``.

Both ``start`` and ``goal`` passed to ``plan(start, goal)`` must already
lie on the constraint manifold — the planner does not run an IK pass on
them.  If either endpoint is off the manifold the planner raises
``ValueError`` with a clear message.

Example::

    from autolife_planning.planning import create_planner
    from autolife_planning.planning.constraints import (
        LinearCoupling, PoseLock,
    )
    import numpy as np

    knee = LinearCoupling(
        master="Joint_Ankle",
        slave="Joint_Knee",
        multiplier=2.0,
    )
    lock = PoseLock(
        link="Link_Left_Gripper",
        target=target_4x4,                       # SE(3) matrix
        frame="ee",                              # or "world"
        weight=[True, True, True, False, False, False],  # rx ry rz x y z
    )

    planner = create_planner(
        "autolife_body",
        constraints=[knee, lock],
        base_config=base,
    )
    result = planner.plan(start, goal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Union

import numpy as np


@dataclass
class LinearCoupling:
    """Linear joint coupling constraint: ``q[slave] = m * q[master] + b``.

    Both joints must be in the planner's *active* subspace — pick a
    subgroup that contains both, e.g. ``autolife_height`` or
    ``autolife_body`` for the knee/ankle case.

    Attributes:
        master: Name of the driving joint.
        slave: Name of the driven joint.
        multiplier: ``m`` in the equation above.
        offset: ``b`` in the equation above (default 0).
    """

    master: str
    slave: str
    multiplier: float
    offset: float = 0.0


@dataclass
class PoseLock:
    """Lock the pose of one URDF link to a target SE(3).

    Internally backed by pinocchio FK + Jacobian.  The 6-element
    ``weight`` mask follows cuRobo's convention ``[rx, ry, rz, x, y, z]``
    — ``True`` (or any truthy value) means "this axis is locked",
    ``False`` means "free".  The ``frame`` parameter picks whether the
    pose error is expressed in the link's ``"ee"`` (LOCAL) frame or in
    ``"world"`` frame, which lets you compose multiple ``PoseLock``
    instances to express things like "free along the gripper's local x
    but stay upright in world z".

    Attributes:
        link: URDF frame/link name to constrain.
        target: ``(4, 4)`` SE(3) matrix the link is locked to.  Compute
            it via pinocchio FK on whichever configuration you want
            ``plan(start, goal)`` to start from — both ``start`` and
            ``goal`` must satisfy this constraint.
        frame: ``"ee"`` (LOCAL) or ``"world"``.
        weight: 6-element ``[rx, ry, rz, x, y, z]`` mask, default all
            ``True`` (full pose lock).
        urdf_path: Override the URDF used for FK.  Defaults to the
            project's ``autolife_robot_config.urdf_path``.
    """

    link: str
    target: np.ndarray
    frame: str = "ee"
    weight: Sequence[bool] = field(
        default_factory=lambda: [True, True, True, True, True, True]
    )
    urdf_path: Union[str, None] = None

    def __post_init__(self):
        target = np.asarray(self.target, dtype=np.float64)
        if target.shape != (4, 4):
            raise ValueError(
                f"PoseLock.target must be a (4, 4) SE(3) matrix, "
                f"got shape {target.shape}"
            )
        self.target = target

        if len(self.weight) != 6:
            raise ValueError(
                f"PoseLock.weight must have 6 elements [rx, ry, rz, x, y, z], "
                f"got {len(self.weight)}"
            )
        self.weight = [bool(w) for w in self.weight]
        if not any(self.weight):
            raise ValueError(
                "PoseLock.weight must lock at least one axis (got all-False)"
            )

        if self.frame not in ("ee", "world", "local", "LOCAL", "WORLD"):
            raise ValueError(
                f"PoseLock.frame must be 'ee' or 'world', got {self.frame!r}"
            )


# Public re-export
Constraint = Union[LinearCoupling, PoseLock]


__all__ = ["LinearCoupling", "PoseLock", "Constraint"]
