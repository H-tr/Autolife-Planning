/**
 * Pinocchio model loader + simple per-process cache.
 *
 * The Autolife URDF is loaded with a ``JointModelPlanar`` root, so the
 * resulting model has ``nv = 3 + 21 = 24`` velocity DOFs in the same
 * order our planner uses ``[Virtual_X, Virtual_Y, Virtual_Theta,
 * j0..j20]``.  ``nq = 25`` because the planar joint stores its yaw as
 * ``[cos(theta), sin(theta)]``.
 *
 * Models are cached by absolute URDF path so multiple constraints
 * built against the same robot share one ``pinocchio::Model`` instance.
 * The cache lives for the lifetime of the process.
 */

#pragma once

#include <memory>
#include <pinocchio/multibody/joint/joint-planar.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace autolife {

class PinocchioRobotCache {
 public:
  static PinocchioRobotCache& instance() {
    static PinocchioRobotCache c;
    return c;
  }

  // Load (or fetch from cache) a planar-rooted Pinocchio model for
  // the URDF at the given absolute path.
  std::shared_ptr<pinocchio::Model> load(const std::string& urdf_path) {
    auto it = cache_.find(urdf_path);
    if (it != cache_.end()) return it->second;

    auto model = std::make_shared<pinocchio::Model>();
    try {
      pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelPlanar(),
                                  *model);
    } catch (const std::exception& e) {
      throw std::runtime_error("PinocchioRobotCache: failed to load URDF '" +
                               urdf_path + "': " + e.what());
    }
    cache_.emplace(urdf_path, model);
    return model;
  }

 private:
  PinocchioRobotCache() = default;
  std::unordered_map<std::string, std::shared_ptr<pinocchio::Model>> cache_;
};

}  // namespace autolife
