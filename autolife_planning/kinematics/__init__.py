# IK solver base + unified factory (always available)
from .ik_solver_base import IKSolverBase, create_ik_solver

# TRAC-IK backend (always available)
from .trac_ik_solver import TracIKSolver

__all__ = [
    "IKSolverBase",
    "create_ik_solver",
    "TracIKSolver",
]

# Pinocchio-dependent modules (pinocchio comes from conda, not pip,
# so it may be absent in pure-pip installs).
try:
    from .collision_model import (
        CollisionContext,
        add_pointcloud_obstacles,
        build_collision_model,
    )
    from .pink_ik_solver import PinkIKSolver
    from .pinocchio_fk import (
        PinocchioContext,
        compute_forward_kinematics,
        compute_jacobian,
        create_pinocchio_context,
    )

    __all__ += [
        "PinocchioContext",
        "create_pinocchio_context",
        "compute_forward_kinematics",
        "compute_jacobian",
        "CollisionContext",
        "build_collision_model",
        "add_pointcloud_obstacles",
        "PinkIKSolver",
    ]
except ModuleNotFoundError:
    # pinocchio (or pink) not installed — pinocchio FK, collision model,
    # and PinkIKSolver are unavailable.  TracIKSolver and create_ik_solver
    # (with backend="trac_ik") still work.
    pass
