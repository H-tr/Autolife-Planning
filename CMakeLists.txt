cmake_minimum_required(VERSION 3.20)
project(autolife_planning_ext LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0167 NEW)

# ── Dependencies ──────────────────────────────────────────────────────

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization filesystem system)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(orocos_kdl QUIET)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND AND NOT orocos_kdl_FOUND)
    pkg_check_modules(orocos_kdl orocos-kdl)
endif()
find_package(NLopt QUIET)

# ── VAMP dependencies (header-only, fetched via CPM) ─────────────────

include(third_party/vamp/cmake/FetchInitCPM.cmake)
include(third_party/vamp/cmake/Dependencies.cmake)

# SIMDxorshift needs AVX2 (conda compiler defaults to -march=nocona)
if(TARGET simdxorshift)
    target_compile_options(simdxorshift PRIVATE -march=native)
endif()

# VAMP headers (our fork with Autolife robot — header-only)
set(VAMP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vamp/src/impl")

# ── OMPL (from source, no Python bindings) ────────────────────────────

set(OMPL_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_VAMP OFF CACHE BOOL "" FORCE)
set(OMPL_REGISTRATION OFF CACHE BOOL "" FORCE)
set(OMPL_BUILD_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ompl EXCLUDE_FROM_ALL)

# ── OMPL + VAMP extension ────────────────────────────────────────────

nanobind_add_module(_ompl_vamp
    ext/ompl_vamp/ompl_vamp_ext.cpp
)

target_include_directories(_ompl_vamp PRIVATE
    ${VAMP_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ompl/src
)

target_link_libraries(_ompl_vamp PRIVATE
    ompl
    Eigen3::Eigen
    nigh
    pdqsort
    Boost::serialization
    Boost::filesystem
    Boost::system
    ${CMAKE_DL_LIBS}   # for dlopen/dlsym in CompiledConstraint
)

target_compile_options(_ompl_vamp PRIVATE
    -march=native
)

install(TARGETS _ompl_vamp LIBRARY DESTINATION autolife_planning)

# ── TRAC-IK extension (pybind11) ─────────────────────────────────────

if(orocos_kdl_FOUND AND NLopt_FOUND)
    file(GLOB_RECURSE TRAC_IK_SOURCES "ext/trac_ik/*.cpp")

    pybind11_add_module(pytracik ${TRAC_IK_SOURCES})

    target_include_directories(pytracik PRIVATE
        ext/trac_ik
        ext/trac_ik/urdf
        ${orocos_kdl_INCLUDE_DIRS}
    )

    target_link_libraries(pytracik PRIVATE
        ${orocos_kdl_LIBRARIES}
        NLopt::nlopt
    )

    install(TARGETS pytracik LIBRARY DESTINATION .)
else()
    message(STATUS "Skipping pytracik: orocos-kdl or NLopt not found")
endif()
