/**
 * OMPL + VAMP Python extension — nanobind bindings.
 *
 * The actual planner, validity checkers, constraint primitives, and
 * pinocchio robot loader live in self-contained internal headers
 * under this directory.  This file is intentionally kept thin: it
 * only imports those headers and exposes the C++ API to Python via
 * nanobind.  If you find yourself adding more than a few lines of
 * non-binding code here, that's a sign it belongs in one of the
 * internal headers instead.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "planner.hpp"

namespace nb = nanobind;
using autolife::OmplVampPlanner;
using autolife::PlanResult;

NB_MODULE(_ompl_vamp, m) {
  m.doc() = "OMPL + VAMP C++ planning extension for Autolife robot";

  nb::class_<PlanResult>(m, "PlanResult")
      .def_ro("solved", &PlanResult::solved)
      .def_ro("path", &PlanResult::path)
      .def_ro("planning_time_ns", &PlanResult::planning_time_ns)
      .def_ro("path_cost", &PlanResult::path_cost);

  nb::class_<OmplVampPlanner>(m, "OmplVampPlanner")
      .def(nb::init<>(), "Create a full-body planner (24 DOF).")
      .def(nb::init<std::vector<int>, std::vector<double>>(),
           "Create a subgroup planner.", nb::arg("active_indices"),
           nb::arg("frozen_config"))
      .def("add_pointcloud", &OmplVampPlanner::add_pointcloud,
           nb::arg("points"), nb::arg("r_min"), nb::arg("r_max"),
           nb::arg("point_radius"))
      .def("add_sphere", &OmplVampPlanner::add_sphere, nb::arg("center"),
           nb::arg("radius"))
      .def("clear_environment", &OmplVampPlanner::clear_environment)
      .def("add_linear_coupling", &OmplVampPlanner::add_linear_coupling,
           nb::arg("master_idx"), nb::arg("slave_idx"), nb::arg("multiplier"),
           nb::arg("offset") = 0.0)
      .def("add_pose_lock", &OmplVampPlanner::add_pose_lock,
           nb::arg("urdf_path"), nb::arg("link_name"), nb::arg("frame"),
           nb::arg("mask"), nb::arg("target_xform"))
      .def("clear_constraints", &OmplVampPlanner::clear_constraints)
      .def("num_constraints", &OmplVampPlanner::num_constraints)
      .def("plan", &OmplVampPlanner::plan, nb::arg("start"), nb::arg("goal"),
           nb::arg("planner_name") = "rrtc", nb::arg("time_limit") = 10.0,
           nb::arg("simplify") = true, nb::arg("interpolate") = true)
      .def("validate", &OmplVampPlanner::validate, nb::arg("config"))
      .def("dimension", &OmplVampPlanner::dimension)
      .def("lower_bounds", &OmplVampPlanner::lower_bounds)
      .def("upper_bounds", &OmplVampPlanner::upper_bounds)
      .def("min_max_radii", &OmplVampPlanner::min_max_radii);
}
