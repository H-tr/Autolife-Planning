"""Render the scene as a voxelised pointcloud instead of raw meshes.

The heavy mesh assets (``rls_2.obj``, ``open_kitchen.obj``) don't render
cleanly under PyBullet's TinyRenderer — walls moire after decimation
and the sofa sometimes vanishes at oblique angles.  The demo already
ships with a clean pointcloud under ``assets/envs/rls_env/pcd/`` that
the planner uses for collision checking — we voxelise that cloud and
export a single OBJ of tiny gray cubes, then load *that* as the only
scene body in the render script.  Result: a uniform "voxel diorama"
look that reads as a clean schematic of the real apartment.

Usage::

    pixi run python scripts/render_videos/prepare_pointcloud_scene.py

Output: ``tmp/rls_render/pc_scene/scene.obj`` (+ matching .mtl for a flat grey).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples" / "demos"))

OUT_DIR = REPO_ROOT / "tmp/rls_render/pc_scene"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Voxel pitch (metres).  8 cm keeps walls readable while keeping the
# total tri count under 200 k on the 150 k-point scene cloud.
PITCH = 0.06
POINT_STRIDE = 1
# Drop voxels above this Z so the iso / hero panes can look INTO the
# room.  The robot is ~1.8 m tall, but viewers care about the floor
# plan and waist-level furniture; ceilings/upper walls only occlude.
Z_CUTOFF = 1.55

# Hex colour per PCD source so the voxel diorama reads like a real apartment
# (walls are light grey, furniture has distinctive tones).
_PCD_COLOUR: dict[str, tuple[int, int, int]] = {
    "rls_2": (210, 214, 222),
    "open_kitchen": (225, 200, 170),
    "wall": (210, 214, 222),
    "workstation": (120, 140, 170),
    "table": (170, 130, 90),
    "sofa": (215, 195, 160),
    "coffee_table": (95, 70, 55),
}


def main() -> None:
    import rls_pick_place as demo  # noqa: E402

    print(f"── loading scene pointcloud from {demo.PCD_DIR} ──")
    # Load every pcd with its source tag so each voxel can be coloured
    # by origin (wall / kitchen / sofa / table / etc).
    chunks: list[np.ndarray] = []
    source_tags: list[np.ndarray] = []
    for src_idx, (_, pcd_name) in enumerate(demo.SCENE_PROPS):
        path = os.path.abspath(f"{demo.PCD_DIR}/{pcd_name}.ply")
        pc = trimesh.load(path)
        v = np.asarray(pc.vertices, dtype=np.float32)  # type: ignore[union-attr]
        if POINT_STRIDE > 1:
            v = v[::POINT_STRIDE]
        chunks.append(v)
        source_tags.append(np.full(len(v), src_idx, dtype=np.int32))
        print(f"  {pcd_name:<15s}  {len(v):>7d} pts  from {Path(path).name}")
    cloud = np.concatenate(chunks, axis=0)
    cloud_tags = np.concatenate(source_tags)
    print(f"  combined   → {len(cloud):>7d} pts (stride={POINT_STRIDE})")
    keep = cloud[:, 2] <= Z_CUTOFF
    cloud = cloud[keep]
    cloud_tags = cloud_tags[keep]
    print(f"  z<={Z_CUTOFF}m → {len(cloud):>7d} pts (cap above chest height)")

    # Voxelise by snapping points to a grid, then keep only SURFACE
    # voxels (those with at least one empty 6-neighbour).  Solid walls
    # become 1-voxel-thick shells, cutting the mesh size ~10×.
    t0 = time.perf_counter()
    ijk = np.floor(cloud / PITCH).astype(np.int64)

    # Dedup voxel coords, but for each unique voxel keep the DOMINANT
    # source tag (the one with the most points in that voxel).
    vox_key = ijk[:, 0] * (1 << 40) + ijk[:, 1] * (1 << 20) + ijk[:, 2]
    order = np.argsort(vox_key, kind="stable")
    sorted_keys = vox_key[order]
    sorted_tags = cloud_tags[order]
    sorted_ijk = ijk[order]
    uniq_mask = np.empty_like(sorted_keys, dtype=bool)
    uniq_mask[0] = True
    uniq_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    grp_starts = np.flatnonzero(uniq_mask)
    grp_ends = np.append(grp_starts[1:], len(sorted_keys))
    unique_ijk = np.empty((len(grp_starts), 3), dtype=np.int64)
    voxel_tag = np.empty(len(grp_starts), dtype=np.int32)
    for gi, (s, e) in enumerate(zip(grp_starts, grp_ends)):
        unique_ijk[gi] = sorted_ijk[s]
        # dominant tag = most common in this cell
        vals, cnts = np.unique(sorted_tags[s:e], return_counts=True)
        voxel_tag[gi] = vals[int(np.argmax(cnts))]

    occupied = set(map(tuple, unique_ijk))
    surface_mask = np.zeros(len(unique_ijk), dtype=bool)
    for i, (x, y, z) in enumerate(unique_ijk):
        for dx, dy, dz in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            if (x + dx, y + dy, z + dz) not in occupied:
                surface_mask[i] = True
                break
    surface_ijk = unique_ijk[surface_mask]
    surface_tag = voxel_tag[surface_mask]
    centres = (surface_ijk.astype(np.float32) + 0.5) * PITCH
    print(
        f"  surface voxels : {len(surface_ijk):>7d} / {len(unique_ijk):>7d} "
        f"({100*len(surface_ijk)/max(1, len(unique_ijk)):.1f}%)"
    )

    # Build one cube per surface voxel as a merged mesh.
    cube = trimesh.creation.box(extents=(PITCH, PITCH, PITCH))
    verts_per = cube.vertices.copy()
    faces_per = cube.faces.copy()

    n = len(centres)
    all_verts = np.empty((n * len(verts_per), 3), dtype=np.float32)
    all_faces = np.empty((n * len(faces_per), 3), dtype=np.int64)
    for i, c in enumerate(centres):
        all_verts[i * 8 : (i + 1) * 8] = verts_per + c
        all_faces[i * 12 : (i + 1) * 12] = faces_per + i * 8
    box_mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    elapsed = time.perf_counter() - t0
    print(
        f"  voxelised  @ pitch={PITCH}m → "
        f"{len(box_mesh.vertices)} verts / {len(box_mesh.faces)} faces "
        f"in {elapsed:.1f}s"
    )

    # Split into one OBJ per source tag so pybullet can colour each
    # independently via changeVisualShape (OBJ vertex-colour / .mtl
    # loading is unreliable across pybullet builds).
    prop_names = [name for name, _ in demo.SCENE_PROPS]
    pcd_names = [pcd for _, pcd in demo.SCENE_PROPS]

    # Clear any previous split outputs
    for p in OUT_DIR.glob("scene_*.obj"):
        p.unlink()
    for p in OUT_DIR.glob("scene_*.mtl"):
        p.unlink()

    print("  per-source counts:")
    for tag_idx, (prop, pcd) in enumerate(zip(prop_names, pcd_names)):
        mask = surface_tag == tag_idx
        if not mask.any():
            continue
        sub_centres = centres[mask]
        # Build one mesh for this source tag
        n = len(sub_centres)
        v = np.empty((n * 8, 3), dtype=np.float32)
        f = np.empty((n * 12, 3), dtype=np.int64)
        for i, c in enumerate(sub_centres):
            v[i * 8 : (i + 1) * 8] = verts_per + c
            f[i * 12 : (i + 1) * 12] = faces_per + i * 8
        sub_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        out = OUT_DIR / f"scene_{prop}.obj"
        sub_mesh.export(out, file_type="obj", include_normals=True)
        r, g, b = _PCD_COLOUR.get(pcd, (210, 214, 224))
        print(f"    {prop:<12s}  {n:>6d} voxels  ({len(f)} tris)  colour=({r},{g},{b})")

    # For backwards compat keep a 'scene.obj' that's the full merged mesh
    # coloured flat gray; the render script prefers the split files.
    sub_mesh_all = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    obj_path = OUT_DIR / "scene.obj"
    sub_mesh_all.export(obj_path, file_type="obj", include_normals=True)
    print(f"  wrote split OBJs + merged {obj_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
