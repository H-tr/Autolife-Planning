"""Build script for C++ extensions (TRAC-IK).

The extension is skipped when build dependencies are missing (KDL headers not found).
"""

import glob
import os

from setuptools import Extension, setup


def _find_kdl_header():
    """Find kdl/tree.hpp in conda or system include paths."""
    search_dirs = []

    # Conda prefix
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        search_dirs.append(os.path.join(conda_prefix, "include"))

    # System paths
    search_dirs.extend(["/usr/include", "/usr/local/include"])

    for d in search_dirs:
        candidate = os.path.join(d, "kdl", "tree.hpp")
        if os.path.isfile(candidate):
            return d
    return None


kdl_include = _find_kdl_header()
ext_modules = []

if kdl_include:
    trac_ik_sources = sorted(glob.glob("ext/trac_ik/**/*.cpp", recursive=True))

    try:
        import pybind11

        pybind11_include = pybind11.get_include()
    except ImportError:
        pybind11_include = ""

    include_dirs = [
        "ext/trac_ik",
        "ext/trac_ik/urdf",
        pybind11_include,
        kdl_include,
        os.path.join(kdl_include, "eigen3"),
    ]

    # Library dirs: conda lib if available, else system
    library_dirs = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        library_dirs.append(os.path.join(conda_prefix, "lib"))

    ext_modules.append(
        Extension(
            "pytracik",
            sources=trac_ik_sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["orocos-kdl", "nlopt"],
            language="c++",
            extra_compile_args=["-std=c++17"],
        )
    )

setup(ext_modules=ext_modules)
