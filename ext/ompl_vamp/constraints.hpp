/**
 * Holonomic constraint primitives for the OMPL + VAMP planner.
 *
 * Two parameterised primitives, both subclasses of
 * ``ompl::base::Constraint``:
 *
 *  - ``LinearCouplingConstraint``  — joint-space, no FK.  Enforces
 *    ``q[slave] = multiplier * q[master] + offset``.  Constant Jacobian.
 *
 *  - ``PoseLockConstraint``  — link pose lock via pinocchio FK +
 *    Jacobian.  6-element axis mask ([rx, ry, rz, x, y, z], cuRobo
 *    convention) selects which axes are constrained, and a ``Frame``
 *    flag picks whether the residual is expressed in the link's
 *    ``LOCAL`` frame or the ``WORLD`` frame.  Combine multiple
 *    instances (in different frames if you like) via OMPL's
 *    ``ConstraintIntersection`` to express things like "free along
 *    EE-x but stay upright in world-z".
 *
 * The active joint subset is handled the same way the rest of the
 * planner handles it: the OMPL state is the reduced ``active_dim``
 * vector, and we expand it to the full 24-DOF body via
 * ``active_indices_`` + ``frozen_config_`` before calling pinocchio.
 * Pinocchio is loaded with a ``JointModelPlanar`` root, so its ``nv``
 * = 3 (planar base) + 21 (URDF joints) = 24, matching our DOF order
 * exactly — Jacobian columns map 1:1.
 */

#pragma once

#include <ompl/base/Constraint.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <utility>
#include <vector>

namespace autolife {

namespace ob = ompl::base;

// ─── Linear joint coupling ──────────────────────────────────────────
//
// Enforces q[slave] = multiplier * q[master] + offset.  Both indices
// are positions in the planner's *active* subspace, so the user must
// pick a subgroup that contains both joints.

class LinearCouplingConstraint : public ob::Constraint {
 public:
  LinearCouplingConstraint(unsigned int ambient_dim, int master_idx,
                           int slave_idx, double multiplier, double offset)
      : ob::Constraint(ambient_dim, /*coDim=*/1),
        master_(master_idx),
        slave_(slave_idx),
        multiplier_(multiplier),
        offset_(offset) {}

  void function(const Eigen::Ref<const Eigen::VectorXd> &q,
                Eigen::Ref<Eigen::VectorXd> out) const override {
    out[0] = q[slave_] - (multiplier_ * q[master_] + offset_);
  }

  void jacobian(const Eigen::Ref<const Eigen::VectorXd> & /*q*/,
                Eigen::Ref<Eigen::MatrixXd> out) const override {
    out.setZero();
    out(0, master_) = -multiplier_;
    out(0, slave_) = 1.0;
  }

 private:
  int master_;
  int slave_;
  double multiplier_;
  double offset_;
};

// ─── Pose lock (parameterised, EE/world frame, 6-axis mask) ─────────

class PoseLockConstraint : public ob::Constraint {
 public:
  enum class Frame { LOCAL, WORLD };

  PoseLockConstraint(std::shared_ptr<pinocchio::Model> model,
                     pinocchio::FrameIndex frame_id,
                     std::vector<int> active_indices,
                     std::array<float, 24> frozen_config, Frame frame,
                     std::array<bool, 6> mask, pinocchio::SE3 target)
      : ob::Constraint(static_cast<unsigned int>(active_indices.size()),
                       count_kept(mask)),
        model_(std::move(model)),
        data_(*model_),
        frame_id_(frame_id),
        active_indices_(std::move(active_indices)),
        frozen_(frozen_config),
        frame_(frame),
        mask_(mask),
        target_(std::move(target)) {
    for (int i = 0; i < 6; ++i)
      if (mask_[i]) kept_axes_.push_back(i);
  }

  pinocchio::FrameIndex frame_id() const { return frame_id_; }
  const std::shared_ptr<pinocchio::Model> &model() const { return model_; }
  const std::vector<int> &active_indices() const { return active_indices_; }
  const std::array<float, 24> &frozen_config() const { return frozen_; }

  void function(const Eigen::Ref<const Eigen::VectorXd> &q_active,
                Eigen::Ref<Eigen::VectorXd> out) const override {
    Eigen::VectorXd q_full = expand_to_pinocchio_q(q_active);
    pinocchio::forwardKinematics(*model_, data_, q_full);
    pinocchio::updateFramePlacement(*model_, data_, frame_id_);
    const auto &T = data_.oMf[frame_id_];

    // Right (LOCAL) vs left (WORLD) trivialised SE(3) error.
    pinocchio::SE3 err = (frame_ == Frame::LOCAL) ? target_.inverse() * T
                                                  : T * target_.inverse();

    auto m = pinocchio::log6(err);
    // Pinocchio log6 returns Motion(linear, angular).  Re-order into
    // [rx, ry, rz, x, y, z] (cuRobo convention).
    Eigen::Matrix<double, 6, 1> e;
    e[0] = m.angular()[0];
    e[1] = m.angular()[1];
    e[2] = m.angular()[2];
    e[3] = m.linear()[0];
    e[4] = m.linear()[1];
    e[5] = m.linear()[2];

    for (std::size_t i = 0; i < kept_axes_.size(); ++i)
      out[i] = e[kept_axes_[i]];
  }

  void jacobian(const Eigen::Ref<const Eigen::VectorXd> &q_active,
                Eigen::Ref<Eigen::MatrixXd> out) const override {
    Eigen::VectorXd q_full = expand_to_pinocchio_q(q_active);

    pinocchio::computeJointJacobians(*model_, data_, q_full);
    pinocchio::updateFramePlacements(*model_, data_);

    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, model_->nv);
    J.setZero();
    pinocchio::getFrameJacobian(
        *model_, data_, frame_id_,
        (frame_ == Frame::LOCAL) ? pinocchio::LOCAL : pinocchio::WORLD, J);

    // Pinocchio Jacobian rows are [linear; angular].  Re-order to
    // [angular; linear] so they line up with the function() output.
    Eigen::Matrix<double, 6, Eigen::Dynamic> J_reord(6, model_->nv);
    J_reord.topRows(3) = J.bottomRows(3);
    J_reord.bottomRows(3) = J.topRows(3);

    out.setZero();
    for (std::size_t r = 0; r < kept_axes_.size(); ++r) {
      const int axis = kept_axes_[r];
      for (std::size_t c = 0; c < active_indices_.size(); ++c) {
        out(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) =
            J_reord(axis, active_indices_[c]);
      }
    }
  }

  // Build a full 24-DOF body config from active values + frozen base,
  // then convert to pinocchio's nq=25 layout (planar root uses
  // [x, y, cos(theta), sin(theta)] for the first four entries).
  Eigen::VectorXd expand_to_pinocchio_q(
      const Eigen::Ref<const Eigen::VectorXd> &q_active) const {
    std::array<double, 24> full;
    for (int i = 0; i < 24; ++i) full[i] = static_cast<double>(frozen_[i]);
    for (std::size_t i = 0; i < active_indices_.size(); ++i)
      full[active_indices_[i]] = q_active[i];

    Eigen::VectorXd pin_q(model_->nq);
    pin_q[0] = full[0];
    pin_q[1] = full[1];
    pin_q[2] = std::cos(full[2]);
    pin_q[3] = std::sin(full[2]);
    for (int i = 0; i < 21; ++i) pin_q[4 + i] = full[3 + i];
    return pin_q;
  }

 private:
  static unsigned int count_kept(const std::array<bool, 6> &mask) {
    unsigned int n = 0;
    for (bool m : mask)
      if (m) ++n;
    return n;
  }

  std::shared_ptr<pinocchio::Model> model_;
  mutable pinocchio::Data data_;  // Newton projection is single-threaded.
  pinocchio::FrameIndex frame_id_;
  std::vector<int> active_indices_;
  std::array<float, 24> frozen_;
  Frame frame_;
  std::array<bool, 6> mask_;
  std::vector<int> kept_axes_;
  pinocchio::SE3 target_;
};

}  // namespace autolife
