"""User-defined manifold constraints, CasADi-backed.

Users write the constraint equation as a CasADi symbolic expression
in their own script.  The wrapper handles symbolic Jacobian via
autodiff, C codegen, compilation, caching, and hand-off to the
native C++ ``CompiledConstraint`` adapter.

No prebuilt constraint primitives are shipped.  Every constraint is
defined inline by the caller as a function of the planner's active
joint vector.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from autolife_planning.config.robot_config import (
    HOME_JOINTS,
    PLANNING_SUBGROUPS,
    autolife_robot_config,
)

# ── cache location ──────────────────────────────────────────────────


def _cache_root() -> Path:
    """Return the constraint cache directory.

    Honours ``AUTOLIFE_CONSTRAINT_CACHE_DIR`` if set (useful for CI).
    Otherwise falls back to ``~/.cache/autolife_planning/constraints``.
    """
    override = os.environ.get("AUTOLIFE_CONSTRAINT_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "autolife_planning" / "constraints").resolve()


@contextmanager
def _cwd(path: Path):
    """Temporarily chdir — CasADi's generate() always writes to cwd."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ── SymbolicContext ────────────────────────────────────────────────


class SymbolicContext:
    """CasADi-friendly view of the planner's active subgroup.

    Holds the CasADi symbolic joint vector ``q`` matching the subgroup's
    active dimension, plus a ``pinocchio.casadi`` model for symbolic FK.

    The context hides the planar-root encoding (the 24-DOF body vector
    uses ``[x, y, theta, j0..j20]``; pinocchio uses
    ``[x, y, cos(theta), sin(theta), j0..j20]``) and the mapping from
    the active subspace back to the full 24-DOF body via ``base_config``.
    """

    def __init__(
        self,
        subgroup: str,
        base_config: np.ndarray | None = None,
    ) -> None:
        if base_config is None:
            base_config = HOME_JOINTS
        self.base_config = np.asarray(base_config, dtype=np.float64).copy()
        if self.base_config.shape != HOME_JOINTS.shape:
            raise ValueError(
                f"base_config must have shape {HOME_JOINTS.shape}, "
                f"got {self.base_config.shape}"
            )

        self.subgroup_name = subgroup
        full_names = list(autolife_robot_config.joint_names)
        if subgroup == "autolife":
            self.active_indices = list(range(24))
            self.active_names = full_names
        else:
            sg = PLANNING_SUBGROUPS.get(subgroup)
            if sg is None:
                raise ValueError(f"Unknown subgroup: {subgroup!r}")
            self.active_names = list(sg["joints"])
            self.active_indices = [full_names.index(j) for j in self.active_names]

        self.q = ca.SX.sym("q", len(self.active_indices))

        # Numeric + symbolic pinocchio models with planar root.
        self.pin_model = pin.buildModelFromUrdf(
            autolife_robot_config.urdf_path, pin.JointModelPlanar()
        )
        self.pin_data = self.pin_model.createData()
        self.cmodel = cpin.Model(self.pin_model)
        self.cdata = self.cmodel.createData()

        # Build the symbolic mapping q_active -> pinocchio q (nq=25).
        self.q_pin = self._build_pinocchio_q(self.q)

        # Pre-run symbolic FK so users can access frame transforms.
        cpin.forwardKinematics(self.cmodel, self.cdata, self.q_pin)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

    def _build_pinocchio_q(self, q_active: ca.SX) -> ca.SX:
        """Map active-dim symbol to pinocchio q (nq=25)."""
        full: list[ca.SX | ca.DM] = [ca.DM(float(v)) for v in self.base_config]
        for i, idx in enumerate(self.active_indices):
            full[idx] = q_active[i]
        # full is 24 entries: [x, y, theta, j0..j20]
        pin_q = ca.vertcat(
            full[0],
            full[1],
            ca.cos(full[2]),
            ca.sin(full[2]),
            *full[3:],
        )
        return pin_q

    def link_pose(self, link_name: str, q_active: ca.SX | None = None):
        """Return the symbolic pinocchio SE3 of a URDF link.

        Pass ``q_active=self.q`` (or omit) to get an expression that
        depends on the active joints symbolically.
        """
        if q_active is None or q_active is self.q:
            frame_id = self.cmodel.getFrameId(link_name)
            return self.cdata.oMf[frame_id]
        # Rebuild with a different symbol.
        cdata = self.cmodel.createData()
        q_pin = self._build_pinocchio_q(q_active)
        cpin.forwardKinematics(self.cmodel, cdata, q_pin)
        cpin.updateFramePlacement(self.cmodel, cdata, self.cmodel.getFrameId(link_name))
        return cdata.oMf[self.cmodel.getFrameId(link_name)]

    def link_translation(self, link_name: str, q_active: ca.SX | None = None) -> ca.SX:
        """Symbolic 3-vector: link position in world frame."""
        return self.link_pose(link_name, q_active).translation

    def link_rotation(self, link_name: str, q_active: ca.SX | None = None) -> ca.SX:
        """Symbolic 3x3 rotation matrix of the link."""
        return self.link_pose(link_name, q_active).rotation

    def evaluate_link_pose(
        self, link_name: str, q_active_numeric: np.ndarray
    ) -> np.ndarray:
        """Compute a NUMERIC 4x4 link pose (handy for building targets).

        Uses the numeric pinocchio model, not the symbolic one, so it is
        fast and has no dependence on CasADi expressions.
        """
        full = self.base_config.copy()
        for i, idx in enumerate(self.active_indices):
            full[idx] = q_active_numeric[i]
        q = np.empty(int(self.pin_model.nq))
        q[0] = full[0]
        q[1] = full[1]
        q[2] = np.cos(full[2])
        q[3] = np.sin(full[2])
        q[4:] = full[3:]
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacement(
            self.pin_model,
            self.pin_data,
            self.pin_model.getFrameId(link_name),
        )
        pose = self.pin_data.oMf[self.pin_model.getFrameId(link_name)]
        M = np.eye(4)
        M[:3, :3] = pose.rotation
        M[:3, 3] = pose.translation
        return M

    def project(
        self,
        q_init: np.ndarray,
        residual: ca.SX,
        tol: float = 1e-8,
        max_iters: int = 100,
    ) -> np.ndarray:
        """Project a joint configuration onto the manifold ``residual(q) = 0``.

        Runs damped Gauss-Newton on the CasADi Jacobian — the same
        iteration OMPL's ``ProjectedStateSpace`` runs internally — so
        the returned config will pass the planner's tolerance check
        and can be used directly as a start or goal state.
        """
        res_fn = ca.Function("proj_res", [self.q], [ca.reshape(residual, -1, 1)])
        jac_fn = ca.Function("proj_jac", [self.q], [ca.jacobian(residual, self.q)])
        q = np.asarray(q_init, dtype=np.float64).copy()
        for _ in range(max_iters):
            r = np.asarray(res_fn(q)).flatten()
            if np.linalg.norm(r) < tol:
                return q
            J = np.asarray(jac_fn(q))
            JJt = J @ J.T + 1e-10 * np.eye(J.shape[0])
            q -= J.T @ np.linalg.solve(JJt, r)
        raise RuntimeError(
            f"SymbolicContext.project failed to converge: "
            f"|residual|={np.linalg.norm(r):.2e} after {max_iters} iters"
        )


# ── Constraint ─────────────────────────────────────────────────────


@dataclass
class Constraint:
    """A user-defined holonomic constraint, JIT-compiled via CasADi.

    Constructing this class triggers (on cold cache):
        1. symbolic Jacobian via ``ca.jacobian(residual, q_sym)``
        2. C code generation via CasADi
        3. compilation to a ``.so`` with ``c++ -O3 -shared -fPIC``
        4. caching under ``~/.cache/autolife_planning/constraints/<sha>/``

    On a cache hit the whole thing is a single ``stat`` + string compare.
    """

    residual: ca.SX
    q_sym: ca.SX
    name: str = "constraint"

    _so_path: Path = field(init=False)
    _ambient_dim: int = field(init=False)
    _co_dim: int = field(init=False)
    _symbol_name: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.q_sym, ca.SX):
            raise TypeError("Constraint.q_sym must be a CasADi SX symbol")

        res = ca.reshape(self.residual, -1, 1)

        self._ambient_dim = int(self.q_sym.numel())
        self._co_dim = int(res.numel())

        jac = ca.densify(ca.jacobian(res, self.q_sym))

        f = ca.Function(self.name, [self.q_sym], [res, jac]).expand()

        raw = f.serialize()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        sha = hashlib.sha256(raw).hexdigest()

        cache_dir = _cache_root() / sha[:2] / sha[2:]
        cache_dir.mkdir(parents=True, exist_ok=True)

        c_path = cache_dir / "constraint.c"
        so_path = cache_dir / "constraint.so"

        if not so_path.exists():
            sys.stderr.write(f"[autolife] compiling constraint {sha[:8]}... ")
            sys.stderr.flush()
            t0 = time.perf_counter()
            with _cwd(cache_dir):
                f.generate("constraint.c")
            compiler = os.environ.get("AUTOLIFE_CONSTRAINT_CC", "c++")
            subprocess.run(
                [
                    compiler,
                    "-O3",
                    "-shared",
                    "-fPIC",
                    str(c_path),
                    "-o",
                    str(so_path),
                ],
                check=True,
            )
            dt = time.perf_counter() - t0
            sys.stderr.write(f"done ({dt * 1000:.0f} ms)\n")
            sys.stderr.flush()

        self._so_path = so_path
        self._symbol_name = self.name

    @property
    def so_path(self) -> Path:
        return self._so_path

    @property
    def ambient_dim(self) -> int:
        return self._ambient_dim

    @property
    def co_dim(self) -> int:
        return self._co_dim

    @property
    def symbol_name(self) -> str:
        return self._symbol_name


__all__ = ["Constraint", "SymbolicContext"]
