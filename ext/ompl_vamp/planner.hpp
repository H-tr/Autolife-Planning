/**
 * Main planner class — OMPL frontend with VAMP collision backend.
 *
 * Two construction modes:
 *
 *  - ``OmplVampPlanner()`` — full body, 24 DOF (3 base + 21 joints).
 *  - ``OmplVampPlanner(active_indices, frozen_config)`` — subgroup
 *    planning over the listed joint indices, with the rest of the
 *    body pinned to ``frozen_config`` for every collision check.
 *
 * The planner exposes a uniform Python-friendly API:
 * ``add_pointcloud`` / ``add_sphere`` / ``clear_environment`` build
 * the obstacle environment, ``plan(start, goal, ...)`` runs OMPL,
 * ``validate(...)``, ``dimension()``, ``lower_bounds()``,
 * ``upper_bounds()`` and ``min_max_radii()`` round out the surface.
 */

#pragma once

#include <ompl/base/ConstrainedSpaceInformation.h>
#include <ompl/base/Constraint.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/constraint/ConstrainedStateSpace.h>
#include <ompl/base/spaces/constraint/ProjectedStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>

#include "compiled_constraint.hpp"
#include "validity.hpp"
// OMPL — informed trees
#include <ompl/geometric/planners/informedtrees/ABITstar.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/informedtrees/EITstar.h>
#include <ompl/geometric/planners/lazyinformedtrees/BLITstar.h>
// OMPL — FMT
#include <ompl/geometric/planners/fmt/BFMT.h>
#include <ompl/geometric/planners/fmt/FMT.h>
// OMPL — KPIECE
#include <ompl/geometric/planners/kpiece/BKPIECE1.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/planners/kpiece/LBKPIECE1.h>
// OMPL — PRM
#include <ompl/geometric/planners/prm/LazyPRM.h>
#include <ompl/geometric/planners/prm/LazyPRMstar.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/prm/SPARS.h>
#include <ompl/geometric/planners/prm/SPARStwo.h>
// OMPL — RRT family
#include <ompl/geometric/planners/rrt/BiTRRT.h>
#include <ompl/geometric/planners/rrt/InformedRRTstar.h>
#include <ompl/geometric/planners/rrt/LBTRRT.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTXstatic.h>
#include <ompl/geometric/planners/rrt/RRTsharp.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/STRRTstar.h>
#include <ompl/geometric/planners/rrt/TRRT.h>
// OMPL — exploration-based
#include <ompl/geometric/planners/est/BiEST.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/pdst/PDST.h>
#include <ompl/geometric/planners/sbl/SBL.h>
#include <ompl/geometric/planners/stride/STRIDE.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vamp/collision/shapes.hh>
#include <vector>

namespace autolife {

namespace og = ompl::geometric;

struct PlanResult {
  bool solved;
  std::vector<std::vector<double>> path;
  int64_t planning_time_ns;
  double path_cost;
};

class OmplVampPlanner {
 public:
  /// Full-body constructor (24 DOF).
  OmplVampPlanner() : active_dim_(Robot::dimension), is_subgroup_(false) {
    Robot::Configuration lo, hi;
    std::array<float, Robot::dimension> zeros{}, ones{};
    ones.fill(1.0f);
    lo = Robot::Configuration(zeros.data());
    hi = Robot::Configuration(ones.data());
    Robot::scale_configuration(lo);
    Robot::scale_configuration(hi);

    auto lo_arr = lo.to_array();
    auto hi_arr = hi.to_array();

    auto space = std::make_shared<ob::RealVectorStateSpace>(Robot::dimension);
    ob::RealVectorBounds bounds(Robot::dimension);
    for (std::size_t i = 0; i < Robot::dimension; ++i) {
      bounds.setLow(i, std::min(lo_arr[i], hi_arr[i]));
      bounds.setHigh(i, std::max(lo_arr[i], hi_arr[i]));
    }
    space->setBounds(bounds);
    space_ = space;
  }

  /// Subgroup constructor (reduced DOF).
  OmplVampPlanner(std::vector<int> active_indices,
                  std::vector<double> frozen_config)
      : active_dim_(active_indices.size()),
        is_subgroup_(true),
        active_indices_(std::move(active_indices)) {
    frozen_config_.resize(frozen_config.size());
    for (std::size_t i = 0; i < frozen_config.size(); ++i)
      frozen_config_[i] = static_cast<float>(frozen_config[i]);

    Robot::Configuration lo, hi;
    std::array<float, Robot::dimension> zeros{}, ones{};
    ones.fill(1.0f);
    lo = Robot::Configuration(zeros.data());
    hi = Robot::Configuration(ones.data());
    Robot::scale_configuration(lo);
    Robot::scale_configuration(hi);
    auto lo_arr = lo.to_array();
    auto hi_arr = hi.to_array();

    auto space = std::make_shared<ob::RealVectorStateSpace>(active_dim_);
    ob::RealVectorBounds bounds(active_dim_);
    for (std::size_t i = 0; i < active_indices_.size(); ++i) {
      auto idx = active_indices_[i];
      bounds.setLow(i, std::min(lo_arr[idx], hi_arr[idx]));
      bounds.setHigh(i, std::max(lo_arr[idx], hi_arr[idx]));
    }
    space->setBounds(bounds);
    space_ = space;
  }

  void add_pointcloud(const std::vector<std::array<float, 3>> &points,
                      float r_min, float r_max, float point_radius) {
    std::vector<vamp::collision::Point> pts;
    pts.reserve(points.size());
    for (const auto &p : points) pts.push_back({p[0], p[1], p[2]});
    float_env_.pointclouds.emplace_back(pts, r_min, r_max, point_radius);
    sync_env();
  }

  void add_sphere(const std::array<float, 3> &center, float radius) {
    float_env_.spheres.emplace_back(vamp::collision::Sphere<float>{
        center[0], center[1], center[2], radius});
    float_env_.sort();
    sync_env();
  }

  void clear_environment() {
    float_env_ = FloatEnv{};
    env_ = VampEnv{};
  }

  // ── Constraint API ────────────────────────────────────────────────
  //
  // Constraints are accumulated by repeated add_*() calls and consumed
  // by the next plan() call.  Use clear_constraints() to reset.

  void add_compiled_constraint(const std::string &so_path,
                               const std::string &symbol_name,
                               unsigned int ambient_dim, unsigned int co_dim) {
    if (static_cast<int>(ambient_dim) != active_dim_) {
      throw std::invalid_argument(
          "add_compiled_constraint: ambient_dim (" +
          std::to_string(ambient_dim) +
          ") does not match planner active dimension (" +
          std::to_string(active_dim_) + ")");
    }
    constraints_.push_back(std::make_shared<CompiledConstraint>(
        ambient_dim, co_dim, so_path, symbol_name));
  }

  void clear_constraints() { constraints_.clear(); }

  std::size_t num_constraints() const { return constraints_.size(); }

  auto plan(std::vector<double> start, std::vector<double> goal,
            const std::string &planner_name, double time_limit, bool simplify,
            bool interpolate) -> PlanResult {
    const bool constrained = !constraints_.empty();
    if (constrained) {
      reject_incompatible_planner(planner_name);
      // Both endpoints must already lie on the constraint manifold —
      // we don't run a manifold IK on them.  This catches the common
      // foot-gun where the user computes a target pose from FK on a
      // different configuration than the one they're starting from.
      check_constraint_satisfaction(start, "start");
      check_constraint_satisfaction(goal, "goal");
    }

    // Pick the right state space + space information for this plan.
    ob::StateSpacePtr active_space = space_;
    ob::SpaceInformationPtr si;
    if (constrained) {
      auto intersection = std::make_shared<ob::ConstraintIntersection>(
          static_cast<unsigned int>(active_dim_), constraints_);
      auto css =
          std::make_shared<ob::ProjectedStateSpace>(space_, intersection);
      // ConstrainedSpaceInformation's constructor wires the
      // back-reference; css->setup() must come *after* that step.
      auto csi = std::make_shared<ob::ConstrainedSpaceInformation>(css);
      css->setup();
      si = csi;
      active_space = css;
    } else {
      si = std::make_shared<ob::SpaceInformation>(space_);
    }

    if (is_subgroup_) {
      si->setStateValidityChecker(std::make_shared<SubgroupValidityChecker>(
          si, env_, active_indices_, frozen_config_));
      // ConstrainedSpaceInformation provides its own motion validator
      // that wraps the projection — we only override it in the
      // unconstrained case.
      if (!constrained) {
        si->setMotionValidator(std::make_shared<SubgroupMotionValidator>(
            si, env_, active_indices_, frozen_config_));
      }
    } else {
      si->setStateValidityChecker(
          std::make_shared<AutolifeValidityChecker>(si, env_));
      if (!constrained) {
        si->setMotionValidator(
            std::make_shared<AutolifeMotionValidator>(si, env_));
      }
    }
    si->setup();

    og::SimpleSetup ss(si);
    ss.setPlanner(create_planner(si, planner_name));

    ob::ScopedState<> ompl_start(active_space);
    ob::ScopedState<> ompl_goal(active_space);
    for (int i = 0; i < active_dim_; ++i) {
      ompl_start[i] = start[i];
      ompl_goal[i] = goal[i];
    }
    ss.setStartAndGoalStates(ompl_start, ompl_goal);

    auto t0 = std::chrono::steady_clock::now();
    auto status = ss.solve(time_limit);
    auto t1 = std::chrono::steady_clock::now();
    auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    PlanResult result;
    result.planning_time_ns = elapsed_ns;
    result.solved = static_cast<bool>(status);

    if (result.solved) {
      if (simplify) ss.simplifySolution();

      auto &path = ss.getSolutionPath();
      // Interpolate after simplify so the returned path has enough
      // waypoints to animate smoothly — OMPL's default uses the
      // longest valid segment fraction of the state space.
      if (interpolate) path.interpolate();
      result.path_cost = path.length();

      for (std::size_t i = 0; i < path.getStateCount(); ++i) {
        const auto *rv = extract_real_state(path.getState(i));
        std::vector<double> config(active_dim_);
        for (int j = 0; j < active_dim_; ++j) config[j] = rv->values[j];
        result.path.push_back(std::move(config));
      }
    } else {
      result.path_cost = std::numeric_limits<double>::infinity();
    }

    return result;
  }

  auto validate(std::vector<double> config) -> bool {
    if (is_subgroup_) {
      alignas(Robot::Configuration::S::Alignment)
          std::array<float, Robot::Configuration::num_scalars>
              buf{};
      std::copy(frozen_config_.begin(), frozen_config_.end(), buf.begin());
      for (std::size_t i = 0; i < active_indices_.size(); ++i)
        buf[active_indices_[i]] = static_cast<float>(config[i]);
      auto q = Robot::Configuration(buf.data());
      return vamp::planning::validate_motion<Robot, kRake, 1>(q, q, env_);
    } else {
      alignas(Robot::Configuration::S::Alignment)
          std::array<float, Robot::Configuration::num_scalars>
              buf{};
      for (std::size_t i = 0; i < Robot::dimension; ++i)
        buf[i] = static_cast<float>(config[i]);
      auto q = Robot::Configuration(buf.data());
      return vamp::planning::validate_motion<Robot, kRake, 1>(q, q, env_);
    }
  }

  auto dimension() const -> int { return active_dim_; }

  auto lower_bounds() const -> std::vector<double> {
    auto bounds = space_->as<ob::RealVectorStateSpace>()->getBounds();
    std::vector<double> lo(active_dim_);
    for (int i = 0; i < active_dim_; ++i) lo[i] = bounds.low[i];
    return lo;
  }

  auto upper_bounds() const -> std::vector<double> {
    auto bounds = space_->as<ob::RealVectorStateSpace>()->getBounds();
    std::vector<double> hi(active_dim_);
    for (int i = 0; i < active_dim_; ++i) hi[i] = bounds.high[i];
    return hi;
  }

  auto min_max_radii() const -> std::pair<float, float> {
    return {Robot::min_radius, Robot::max_radius};
  }

  /// Switch to a different subgroup without rebuilding the environment.
  void set_subgroup(std::vector<int> active_indices,
                    std::vector<double> frozen_config) {
    active_dim_ = static_cast<int>(active_indices.size());
    is_subgroup_ = true;
    active_indices_ = std::move(active_indices);
    frozen_config_.resize(frozen_config.size());
    for (std::size_t i = 0; i < frozen_config.size(); ++i)
      frozen_config_[i] = static_cast<float>(frozen_config[i]);
    constraints_.clear();
    rebuild_space_();
  }

  /// Switch back to full-body mode without rebuilding the environment.
  void set_full_body() {
    active_dim_ = Robot::dimension;
    is_subgroup_ = false;
    active_indices_.clear();
    frozen_config_.clear();
    constraints_.clear();
    rebuild_space_();
  }

 private:
  int active_dim_;
  bool is_subgroup_;
  std::vector<int> active_indices_;
  std::vector<float> frozen_config_;
  ob::StateSpacePtr space_;
  FloatEnv float_env_;
  VampEnv env_;
  std::vector<ob::ConstraintPtr> constraints_;

  void sync_env() { env_ = VampEnv(float_env_); }

  void rebuild_space_() {
    Robot::Configuration lo, hi;
    std::array<float, Robot::dimension> zeros{}, ones{};
    ones.fill(1.0f);
    lo = Robot::Configuration(zeros.data());
    hi = Robot::Configuration(ones.data());
    Robot::scale_configuration(lo);
    Robot::scale_configuration(hi);
    auto lo_arr = lo.to_array();
    auto hi_arr = hi.to_array();

    auto space = std::make_shared<ob::RealVectorStateSpace>(active_dim_);
    ob::RealVectorBounds bounds(active_dim_);
    if (is_subgroup_) {
      for (std::size_t i = 0; i < active_indices_.size(); ++i) {
        auto idx = active_indices_[i];
        bounds.setLow(i, std::min(lo_arr[idx], hi_arr[idx]));
        bounds.setHigh(i, std::max(lo_arr[idx], hi_arr[idx]));
      }
    } else {
      for (int i = 0; i < active_dim_; ++i) {
        bounds.setLow(i, std::min(lo_arr[i], hi_arr[i]));
        bounds.setHigh(i, std::max(lo_arr[i], hi_arr[i]));
      }
    }
    space->setBounds(bounds);
    space_ = space;
  }

  // Check that *active_q* satisfies every constraint within the
  // OMPL constraint tolerance.  Throws std::invalid_argument with a
  // descriptive message naming which constraint and how badly it was
  // violated, so the user gets a clear error rather than a hang.
  void check_constraint_satisfaction(const std::vector<double> &active_q,
                                     const char *which) const {
    Eigen::VectorXd q(active_dim_);
    for (int i = 0; i < active_dim_; ++i) q[i] = active_q[i];
    for (std::size_t i = 0; i < constraints_.size(); ++i) {
      const auto &c = constraints_[i];
      Eigen::VectorXd r(c->getCoDimension());
      c->function(q, r);
      const double residual = r.norm();
      if (residual > c->getTolerance()) {
        throw std::invalid_argument(
            std::string("Constraint #") + std::to_string(i) +
            " is violated at " + which + " (residual " +
            std::to_string(residual) + " > tolerance " +
            std::to_string(c->getTolerance()) +
            ").  Both start and goal must already lie on the constraint "
            "manifold — compute target poses from FK on the start config "
            "you intend to plan from.");
      }
    }
  }

  // ProjectedStateSpace only supports single-tree planners — batch
  // / informed-tree variants don't go through manifold projection.
  static void reject_incompatible_planner(const std::string &name) {
    static const std::vector<std::string> bad = {
        "bitstar",  "abitstar",   "aitstar",  "eitstar",
        "blitstar", "fmt",        "bfmt",     "informed_rrtstar",
        "rrtsharp", "rrtxstatic", "strrtstar"};
    for (const auto &b : bad) {
      if (name == b) {
        throw std::invalid_argument(
            "Planner '" + name +
            "' is incompatible with constrained planning.  Use one of: "
            "rrtc, rrt, rrtstar, prm, prmstar, kpiece, bkpiece, lbkpiece, "
            "est, biest, sbl, stride.");
      }
    }
  }

  static auto create_planner(const ob::SpaceInformationPtr &si,
                             const std::string &name) -> ob::PlannerPtr {
    // RRT family
    if (name == "rrtc" || name == "rrtconnect")
      return std::make_shared<og::RRTConnect>(si);
    if (name == "rrt") return std::make_shared<og::RRT>(si);
    if (name == "rrtstar") return std::make_shared<og::RRTstar>(si);
    if (name == "informed_rrtstar")
      return std::make_shared<og::InformedRRTstar>(si);
    if (name == "rrtsharp") return std::make_shared<og::RRTsharp>(si);
    if (name == "rrtxstatic") return std::make_shared<og::RRTXstatic>(si);
    if (name == "strrtstar") return std::make_shared<og::STRRTstar>(si);
    if (name == "lbtrrt") return std::make_shared<og::LBTRRT>(si);
    if (name == "trrt") return std::make_shared<og::TRRT>(si);
    if (name == "bitrrt") return std::make_shared<og::BiTRRT>(si);
    // Informed trees (asymptotically optimal)
    if (name == "bitstar") return std::make_shared<og::BITstar>(si);
    if (name == "abitstar") return std::make_shared<og::ABITstar>(si);
    if (name == "aitstar") return std::make_shared<og::AITstar>(si);
    if (name == "eitstar") return std::make_shared<og::EITstar>(si);
    if (name == "blitstar") return std::make_shared<og::BLITstar>(si);
    // FMT
    if (name == "fmt") return std::make_shared<og::FMT>(si);
    if (name == "bfmt") return std::make_shared<og::BFMT>(si);
    // KPIECE
    if (name == "kpiece") return std::make_shared<og::KPIECE1>(si);
    if (name == "bkpiece") return std::make_shared<og::BKPIECE1>(si);
    if (name == "lbkpiece") return std::make_shared<og::LBKPIECE1>(si);
    // PRM family
    if (name == "prm") return std::make_shared<og::PRM>(si);
    if (name == "prmstar") return std::make_shared<og::PRMstar>(si);
    if (name == "lazyprm") return std::make_shared<og::LazyPRM>(si);
    if (name == "lazyprmstar") return std::make_shared<og::LazyPRMstar>(si);
    if (name == "spars") return std::make_shared<og::SPARS>(si);
    if (name == "spars2") return std::make_shared<og::SPARStwo>(si);
    // Exploration-based
    if (name == "est") return std::make_shared<og::EST>(si);
    if (name == "biest") return std::make_shared<og::BiEST>(si);
    if (name == "sbl") return std::make_shared<og::SBL>(si);
    if (name == "stride") return std::make_shared<og::STRIDE>(si);
    if (name == "pdst") return std::make_shared<og::PDST>(si);
    throw std::invalid_argument("Unknown planner: " + name);
  }
};

}  // namespace autolife
