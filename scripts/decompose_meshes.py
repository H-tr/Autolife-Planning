#!/usr/bin/env python3
"""Convex-decompose each collision mesh in a URDF and emit an "exploded" URDF
where every link has one ``<collision>`` block per convex piece.

Foam spherizes whole meshes; a concave torso mesh leaves foam's medial axis
passing through empty space, giving a giant sphere inside the concavity.
Decomposing first means every piece foam sees is (approximately) convex, so
its fitted spheres actually hug surface geometry.
"""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import coacd
import numpy as np
import trimesh


def decompose_mesh(
    mesh: trimesh.Trimesh,
    *,
    threshold: float,
    resolution: int,
    mcts_nodes: int,
    mcts_iterations: int,
    mcts_max_depth: int,
    preprocess_resolution: int,
    max_convex_hull: int,
    seed: int,
) -> list[trimesh.Trimesh]:
    """Run CoACD on ``mesh`` and return a list of convex-piece trimeshes."""
    cm = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    parts = coacd.run_coacd(
        cm,
        threshold=threshold,
        # "on" forces manifold-preserving remeshing of the input, which
        # throws away micro-artefacts (internal bolt holes, wire paths) that
        # would otherwise be decomposed as spurious concavities.
        preprocess_mode="on",
        preprocess_resolution=preprocess_resolution,
        resolution=resolution,
        mcts_nodes=mcts_nodes,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        merge=True,
        max_convex_hull=max_convex_hull,
        max_ch_vertex=256,
        seed=seed,
    )

    # Clean each piece: trimesh ``process=True`` merges duplicate vertices,
    # drops zero-area faces, and recomputes normals. Then take the convex
    # hull explicitly so the STL we hand to foam's sphere-tree tool is a
    # guaranteed-manifold closed convex polytope (foam's makeTreeMedial
    # segfaults on raw CoACD sliver output otherwise).
    cleaned: list[trimesh.Trimesh] = []
    for v, f in parts:
        piece = trimesh.Trimesh(vertices=v, faces=f, process=True)
        if len(piece.vertices) < 4 or piece.volume <= 0:
            continue
        try:
            piece = piece.convex_hull
        except Exception:
            continue
        # Drop slivers below 1 mm³ — too small to meaningfully contribute
        # to collision coverage and the most common source of degenerate
        # sphere-tree failures.
        if piece.volume < 1e-9:
            continue
        cleaned.append(piece)
    return cleaned


def _mesh_rel(urdf_dir: Path, filename: str) -> Path:
    """Resolve a URDF mesh filename (may be ``package://`` or a relative path)."""
    cleaned = filename.replace("package://", "")
    return (urdf_dir / cleaned).resolve()


def _package_rel(urdf_dir: Path, abs_path: Path) -> str:
    """Render an absolute path as the original URDF-relative form."""
    return abs_path.relative_to(urdf_dir).as_posix()


def _resolve_mesh(urdf_dir: Path, raw_mesh_dir: Path | None, filename: str) -> Path:
    mesh_abs = _mesh_rel(urdf_dir, filename)
    if raw_mesh_dir is not None:
        raw_candidate = raw_mesh_dir / mesh_abs.name
        if raw_candidate.exists():
            return raw_candidate
    return mesh_abs


def _replace_collisions_with_convex_hulls(
    link: ET.Element,
    link_name: str,
    collisions: list,
    urdf_dir: Path,
    decomposed_mesh_dir: Path,
    raw_mesh_dir: Path | None,
) -> None:
    """Rewrite a link's <collision> blocks to point at a convex-hull STL."""
    new_collisions: list[ET.Element] = []
    for idx, collision in enumerate(collisions):
        geom = collision.find("geometry")
        if geom is None:
            new_collisions.append(collision)
            continue
        mesh_elem = geom.find("mesh")
        if mesh_elem is None:
            new_collisions.append(collision)
            continue

        mesh_abs = _resolve_mesh(urdf_dir, raw_mesh_dir, mesh_elem.get("filename"))
        if not mesh_abs.exists():
            new_collisions.append(collision)
            continue

        mesh = trimesh.load(mesh_abs, force="mesh")
        try:
            hull = mesh.convex_hull
        except Exception:
            new_collisions.append(collision)
            continue

        hull_path = decomposed_mesh_dir / f"{link_name}_hull{idx:02d}.stl"
        hull.export(hull_path)

        new_coll = ET.Element("collision", {"name": f"{link_name}_hull{idx:02d}"})
        origin_elem = collision.find("origin")
        if origin_elem is not None:
            new_coll.append(copy.deepcopy(origin_elem))
        new_geom = ET.SubElement(new_coll, "geometry")
        ET.SubElement(
            new_geom,
            "mesh",
            {"filename": _package_rel(urdf_dir, hull_path)},
        )
        new_collisions.append(new_coll)

    for old in collisions:
        link.remove(old)
    for new in new_collisions:
        link.append(new)


def decompose_urdf(
    urdf_path: Path,
    out_urdf: Path,
    decomposed_mesh_dir: Path,
    raw_mesh_dir: Path | None,
    include_patterns: list[re.Pattern] | None,
    **coacd_kwargs,
) -> None:
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    urdf_dir = urdf_path.parent

    if decomposed_mesh_dir.exists():
        shutil.rmtree(decomposed_mesh_dir)
    decomposed_mesh_dir.mkdir(parents=True)

    total_links_decomposed = 0
    total_pieces = 0
    total_links_passthrough = 0

    for link in root.findall("link"):
        link_name = link.get("name")
        collisions = link.findall("collision")
        if not collisions:
            continue

        # Only decompose links whose name matches one of the include patterns.
        # For the rest, replace each collision mesh with its convex hull:
        # (1) foam's makeTreeSpawn/Medial segfault on non-manifold raw STLs,
        # (2) on non-critical links we don't care about tight fit, so a
        # convex-hull approximation is fine (gives a handful of spheres).
        if include_patterns is not None and not any(
            pat.search(link_name) for pat in include_patterns
        ):
            total_links_passthrough += 1
            _replace_collisions_with_convex_hulls(
                link,
                link_name,
                collisions,
                urdf_dir,
                decomposed_mesh_dir,
                raw_mesh_dir,
            )
            continue

        # One link may have multiple collision blocks; handle each independently.
        new_collisions: list[ET.Element] = []

        for collision in collisions:
            geom = collision.find("geometry")
            if geom is None:
                new_collisions.append(collision)
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is None:
                new_collisions.append(collision)
                continue

            mesh_filename = mesh_elem.get("filename")
            mesh_abs = _mesh_rel(urdf_dir, mesh_filename)

            # Prefer the raw (un-repaired) mesh as CoACD input. The repair step
            # can replace meshes with convex hulls, which would make CoACD a
            # no-op. Match by basename in raw_mesh_dir.
            if raw_mesh_dir is not None:
                raw_candidate = raw_mesh_dir / mesh_abs.name
                if raw_candidate.exists():
                    mesh_abs = raw_candidate
                else:
                    print(
                        f"[warn] {link_name}: no raw mesh at {raw_candidate}; "
                        f"falling back to {mesh_abs}"
                    )

            if not mesh_abs.exists():
                print(f"[skip] {link_name}: mesh not found: {mesh_abs}")
                new_collisions.append(collision)
                continue

            mesh = trimesh.load(mesh_abs, force="mesh")
            print(
                f"[decompose] {link_name}  <- {mesh_abs.name}  "
                f"({len(mesh.vertices)} verts, vol={mesh.volume:.4g})",
                flush=True,
            )
            parts = decompose_mesh(mesh, **coacd_kwargs)
            print(f"            -> {len(parts)} convex part(s)", flush=True)
            total_links_decomposed += 1
            total_pieces += len(parts)

            origin_elem = collision.find("origin")

            for idx, part in enumerate(parts):
                part_path = decomposed_mesh_dir / f"{link_name}_part{idx:02d}.stl"
                part.export(part_path)

                new_coll = ET.Element(
                    "collision", {"name": f"{link_name}_part{idx:02d}"}
                )
                if origin_elem is not None:
                    new_coll.append(copy.deepcopy(origin_elem))
                new_geom = ET.SubElement(new_coll, "geometry")
                ET.SubElement(
                    new_geom,
                    "mesh",
                    {"filename": _package_rel(urdf_dir, part_path)},
                )
                new_collisions.append(new_coll)

        # Replace old collisions with the new list
        for old in collisions:
            link.remove(old)
        for new in new_collisions:
            link.append(new)

    ET.indent(tree, space="  ")
    tree.write(out_urdf, encoding="utf-8", xml_declaration=True)
    print()
    print(
        f"Wrote {out_urdf}  "
        f"(decomposed {total_links_decomposed} link(s) into {total_pieces} convex pieces; "
        f"{total_links_passthrough} link(s) passed through with raw meshes)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Source URDF")
    p.add_argument("--output", type=Path, required=True, help="Exploded URDF to write")
    p.add_argument(
        "--parts-dir",
        type=Path,
        required=True,
        help="Directory to write per-piece STLs into (wiped on each run)",
    )
    p.add_argument(
        "--raw-mesh-dir",
        type=Path,
        default=None,
        help="Directory holding the original (un-repaired) STL meshes. "
        "Each URDF mesh is matched by basename and the raw version is "
        "fed to CoACD so we decompose the true geometry, not its "
        "convex-hull repair.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="CoACD concavity threshold. Larger = fewer pieces. 0.05 is the "
        "CoACD default but shatters raw SolidWorks meshes on internal "
        "artefacts; 0.15 is a practical sweet spot for robot STLs.",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=2000,
        help="Voxel resolution for CoACD (higher = finer analysis, slower).",
    )
    p.add_argument("--preprocess-resolution", type=int, default=50)
    p.add_argument("--mcts-nodes", type=int, default=20)
    p.add_argument("--mcts-iterations", type=int, default=200)
    p.add_argument("--mcts-max-depth", type=int, default=3)
    p.add_argument(
        "--max-convex-hull",
        type=int,
        default=20,
        help="Hard cap on convex pieces per mesh. -1 for unlimited.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--include",
        action="append",
        default=None,
        help="Regex pattern for link names to decompose. May be given multiple "
        "times. Links whose name does not match any pattern keep their "
        "original collision geometry untouched (raw mesh → foam gets "
        "just a few coarse spheres). If omitted, every link is decomposed.",
    )
    args = p.parse_args()

    coacd.set_log_level("warn")

    include_patterns = [re.compile(p) for p in args.include] if args.include else None

    decompose_urdf(
        urdf_path=args.input.resolve(),
        out_urdf=args.output.resolve(),
        decomposed_mesh_dir=args.parts_dir.resolve(),
        raw_mesh_dir=args.raw_mesh_dir.resolve() if args.raw_mesh_dir else None,
        include_patterns=include_patterns,
        threshold=args.threshold,
        resolution=args.resolution,
        preprocess_resolution=args.preprocess_resolution,
        mcts_nodes=args.mcts_nodes,
        mcts_iterations=args.mcts_iterations,
        mcts_max_depth=args.mcts_max_depth,
        max_convex_hull=args.max_convex_hull,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
