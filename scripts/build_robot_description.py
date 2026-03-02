#!/usr/bin/env python3
"""Build pipeline: convert a raw URDF into planning-ready robot description.

All paths are provided via CLI arguments — no hardcoded project paths.

Stages:
  1. Preprocess URDF  (rename robot, fix mesh paths, freeze gripper joints)
  2. Generate simple URDF  (strip sensor/decorative links)
  3. Generate base URDFs  (add 3-DOF virtual base joints)
  4. Generate SRDF  (groups, end-effectors, collision disables)
  5. Copy meshes
  6. Repair collision meshes  (optional, requires foam -- makes meshes valid for sphere-tree)
  7. Distribute to third-party directories
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROBOT_NAME = "autolife"

# Gripper joint prefixes – these revolute joints become fixed for planning.
GRIPPER_JOINT_PREFIXES = ("Joint_Left_Gripper", "Joint_Right_Gripper")

# Links to strip for the simplified URDF (sensors, decorative covers, gripper sub-parts).
# These are all attached via fixed joints and not relevant for motion planning collision
# checking – the parent link's collision mesh already covers the relevant volume.
STRIP_LINK_PREFIXES = (
    "Link_Camera_",
    "Link_Lidar_",
    "Link_IMU",
    "Link_Knee_Black_Cover",
    "Link_Ground_Vehicle_Visual_Cover",
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Add pretty-print indentation (for Python < 3.9 compat)."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    _indent(tree.getroot())
    tree.write(str(path), encoding="utf-8", xml_declaration=True)
    print(f"  wrote {path}")


def _is_gripper_joint(name: str) -> bool:
    return any(name.startswith(p) for p in GRIPPER_JOINT_PREFIXES)


# ── Stage 1: Preprocess URDF ────────────────────────────────────────────────


def preprocess_urdf(urdf_path: Path) -> ET.ElementTree:
    """Parse raw URDF, rename robot, rewrite mesh paths, freeze gripper joints."""
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    # Rename robot
    root.set("name", ROBOT_NAME)

    # Rewrite mesh paths: package://autolife_description/meshes/robot_v2_0/X.STL → package://meshes/X.STL
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        # Strip any package-relative prefix down to just the basename
        if "meshes/" in fn:
            basename = fn.split("/")[-1]
            mesh.set("filename", f"package://meshes/{basename}")

    # Convert gripper sub-joints from revolute → fixed
    for joint in root.findall("joint"):
        if _is_gripper_joint(joint.get("name", "")) and joint.get("type") == "revolute":
            joint.set("type", "fixed")
            for tag in ("limit", "axis"):
                el = joint.find(tag)
                if el is not None:
                    joint.remove(el)

    return tree


# ── Stage 2: Generate simple URDF ───────────────────────────────────────────


def _should_strip(name: str) -> bool:
    return any(name.startswith(p) for p in STRIP_LINK_PREFIXES)


def generate_simple_urdf(preprocessed: ET.ElementTree) -> ET.ElementTree:
    """Strip sensor, decorative, and gripper sub-links for faster collision checking."""
    tree = copy.deepcopy(preprocessed)
    root = tree.getroot()

    links_to_remove = {
        link.get("name")
        for link in root.findall("link")
        if _should_strip(link.get("name", ""))
    }

    for link in list(root.findall("link")):
        if link.get("name") in links_to_remove:
            root.remove(link)

    for joint in list(root.findall("joint")):
        child_el = joint.find("child")
        if child_el is not None and child_el.get("link") in links_to_remove:
            root.remove(joint)

    print(f"  stripped {len(links_to_remove)} non-planning links")
    return tree


# ── Stage 3: Generate base URDF ─────────────────────────────────────────────

_VIRTUAL_LINK_TEMPLATE = """
<link name="{name}">
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="0.001"/>
    <inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>
  </inertial>
</link>
"""


def generate_base_urdf(preprocessed: ET.ElementTree) -> ET.ElementTree:
    """Insert 3-DOF virtual base (X, Y, Theta) before Link_Ground_Vehicle."""
    tree = copy.deepcopy(preprocessed)
    root = tree.getroot()

    # Find and remove Joint_Ground_Vehicle_Base (fixed joint Zero_Point → Ground_Vehicle)
    base_joint = None
    for j in root.findall("joint"):
        if j.get("name") == "Joint_Ground_Vehicle_Base":
            base_joint = j
            break

    if base_joint is None:
        raise RuntimeError("Joint_Ground_Vehicle_Base not found in URDF")

    origin_el = base_joint.find("origin")
    base_xyz = origin_el.get("xyz", "0 0 0") if origin_el is not None else "0 0 0"
    base_rpy = origin_el.get("rpy", "0 0 0") if origin_el is not None else "0 0 0"

    root.remove(base_joint)

    # Insert virtual links
    for name in ["Link_Virtual_X", "Link_Virtual_Y", "Link_Virtual_Theta"]:
        root.append(ET.fromstring(_VIRTUAL_LINK_TEMPLATE.format(name=name)))

    pi = math.pi
    root.append(
        _make_joint(
            "Joint_Virtual_X",
            "prismatic",
            "Link_Zero_Point",
            "Link_Virtual_X",
            axis="1 0 0",
            lower=-10,
            upper=10,
        )
    )
    root.append(
        _make_joint(
            "Joint_Virtual_Y",
            "prismatic",
            "Link_Virtual_X",
            "Link_Virtual_Y",
            axis="0 1 0",
            lower=-10,
            upper=10,
        )
    )
    root.append(
        _make_joint(
            "Joint_Virtual_Theta",
            "revolute",
            "Link_Virtual_Y",
            "Link_Virtual_Theta",
            axis="0 0 1",
            lower=-pi,
            upper=pi,
        )
    )
    root.append(
        _make_joint(
            "Joint_Ground_Vehicle_Fixed",
            "fixed",
            "Link_Virtual_Theta",
            "Link_Ground_Vehicle",
            origin_xyz=base_xyz,
            origin_rpy=base_rpy,
        )
    )

    return tree


def _make_joint(
    name: str,
    jtype: str,
    parent: str,
    child: str,
    *,
    axis: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
    origin_xyz: str = "0 0 0",
    origin_rpy: str = "0 0 0",
) -> ET.Element:
    joint = ET.Element("joint", name=name, type=jtype)
    ET.SubElement(joint, "origin", xyz=origin_xyz, rpy=origin_rpy)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    if axis is not None:
        ET.SubElement(joint, "axis", xyz=axis)
    if lower is not None and upper is not None:
        ET.SubElement(
            joint,
            "limit",
            lower=str(lower),
            upper=str(upper),
            effort="100",
            velocity="1.0",
        )
    return joint


# ── Stage 4: Generate SRDF ──────────────────────────────────────────────────


def generate_srdf(preprocessed: ET.ElementTree) -> ET.ElementTree:
    """Generate SRDF with groups, end-effectors, and collision disables."""
    root_urdf = preprocessed.getroot()
    all_links = {link.get("name") for link in root_urdf.findall("link")}

    robot = ET.Element("robot", name=ROBOT_NAME)
    tree = ET.ElementTree(robot)

    # Groups
    _add_group(
        robot, "Left_Arm", "Link_Waist_Yaw_to_Shoulder_Inner", "Link_Left_Gripper"
    )
    _add_group(
        robot, "Right_Arm", "Link_Waist_Yaw_to_Shoulder_Inner", "Link_Right_Gripper"
    )
    _add_group(robot, "Lower_Body", "Link_Ground_Vehicle", "Link_Knee_to_Waist_Pitch")
    _add_group(robot, "Head", "Link_Waist_Yaw_to_Shoulder_Inner", "Link_Head")

    # Group states
    _add_group_state(
        robot,
        "Home",
        "Left_Arm",
        [
            "Joint_Left_Shoulder_Inner",
            "Joint_Left_Shoulder_Outer",
            "Joint_Left_UpperArm",
            "Joint_Left_Elbow",
            "Joint_Left_Forearm",
            "Joint_Left_Wrist_Upper",
            "Joint_Left_Wrist_Lower",
            "Joint_Waist_Pitch",
            "Joint_Waist_Yaw",
        ],
    )
    _add_group_state(
        robot,
        "Home",
        "Right_Arm",
        [
            "Joint_Right_Shoulder_Inner",
            "Joint_Right_Shoulder_Outer",
            "Joint_Right_UpperArm",
            "Joint_Right_Elbow",
            "Joint_Right_Forearm",
            "Joint_Right_Wrist_Upper",
            "Joint_Right_Wrist_Lower",
            "Joint_Waist_Pitch",
            "Joint_Waist_Yaw",
        ],
    )
    _add_group_state(robot, "Home", "Lower_Body", ["Joint_Ankle", "Joint_Knee"])
    _add_group_state(
        robot, "Home", "Head", ["Joint_Neck_Roll", "Joint_Neck_Pitch", "Joint_Neck_Yaw"]
    )

    # End effectors
    ET.SubElement(
        robot,
        "end_effector",
        name="Left_Gripper",
        parent_link="Link_Left_Gripper",
        group="Left_Arm",
    )
    ET.SubElement(
        robot,
        "end_effector",
        name="Right_Gripper",
        parent_link="Link_Right_Gripper",
        group="Right_Arm",
    )

    # Disable collisions
    _generate_collision_disables(robot, all_links)

    return tree


def _add_group(robot: ET.Element, name: str, base_link: str, tip_link: str) -> None:
    group = ET.SubElement(robot, "group", name=name)
    ET.SubElement(group, "chain", base_link=base_link, tip_link=tip_link)


def _add_group_state(
    robot: ET.Element, state_name: str, group_name: str, joints: list[str]
) -> None:
    gs = ET.SubElement(robot, "group_state", name=state_name, group=group_name)
    for j in joints:
        ET.SubElement(gs, "joint", name=j, value="0")


def _generate_collision_disables(robot: ET.Element, all_links: set[str]) -> None:
    """Generate disable_collisions pairs adapted from v1.0 patterns."""
    pairs: set[tuple[str, str]] = set()

    def add(a: str, b: str) -> None:
        if a in all_links and b in all_links:
            pairs.add((min(a, b), max(a, b)))

    main_chain = [
        "Link_Zero_Point",
        "Link_Ground_Vehicle",
        "Link_Ankle_to_Knee",
        "Link_Knee_to_Waist_Pitch",
        "Link_Waist_Pitch_to_Waist_Yaw",
        "Link_Waist_Yaw_to_Shoulder_Inner",
    ]
    left_arm_chain = [
        "Link_Left_Shoulder_Inner_to_Shoulder_Outer",
        "Link_Left_Shoulder_Outer_to_UpperArm",
        "Link_Left_UpperArm_to_Elbow",
        "Link_Left_Elbow_to_Forearm",
        "Link_Left_Forearm_to_Wrist_Upper",
        "Link_Left_Wrist_Upper_to_Wrist_Lower",
        "Link_Left_Wrist_Lower_to_Gripper",
        "Link_Left_Gripper",
    ]
    right_arm_chain = [
        "Link_Right_Shoulder_Inner_to_Shoulder_Outer",
        "Link_Right_Shoulder_Outer_to_UpperArm",
        "Link_Right_UpperArm_to_Elbow",
        "Link_Right_Elbow_to_Forearm",
        "Link_Right_Forearm_to_Wrist_Upper",
        "Link_Right_Wrist_Upper_to_Wrist_Lower",
        "Link_Right_Wrist_Lower_to_Gripper",
        "Link_Right_Gripper",
    ]
    neck_chain = [
        "Link_Neck_Roll_to_Neck_Pitch",
        "Link_Neck_Pitch_to_Neck_Yaw",
        "Link_Neck_Yaw_to_Head",
        "Link_Head",
    ]
    gripper_links = [
        "Link_Left_Gripper_Left_Inner_Knuckle",
        "Link_Left_Gripper_Left_Finger",
        "Link_Left_Gripper_Right_Outer_Knuckle",
        "Link_Left_Gripper_Right_Inner_Knuckle",
        "Link_Left_Gripper_Right_Finger",
        "Link_Right_Gripper_Left_Inner_Knuckle",
        "Link_Right_Gripper_Left_Finger",
        "Link_Right_Gripper_Right_Outer_Knuckle",
        "Link_Right_Gripper_Right_Inner_Knuckle",
        "Link_Right_Gripper_Right_Finger",
    ]
    decorative_links = [
        "Link_Knee_Black_Cover",
        "Link_Ground_Vehicle_Visual_Cover",
        "Link_Camera_Gripper_Left",
        "Link_Camera_Gripper_Right",
        "Link_Camera_Head_Back",
        "Link_Camera_Head_Right_Eye",
        "Link_Camera_Head_Left_Eye",
        "Link_Camera_Head_Forehead",
        "Link_IMU",
        "Link_Lidar_Back",
        "Link_Lidar_Front",
    ]

    body_links = main_chain + left_arm_chain + right_arm_chain + neck_chain

    # Adjacent links along main chain (2-hop)
    for i in range(len(main_chain)):
        for j in range(i + 1, min(i + 3, len(main_chain))):
            add(main_chain[i], main_chain[j])

    # All intra-chain pairs for arms and neck (spherized model causes false positives)
    for chain in [left_arm_chain, right_arm_chain, neck_chain]:
        for i in range(len(chain)):
            for j in range(i + 1, len(chain)):
                add(chain[i], chain[j])

    # Substrings for distal arm links that should collide with the chest.
    _distal_arm = ("Finger", "Gripper", "Elbow", "Forearm", "Wrist")

    # Main chain vs both arm chains
    # Keep collision checking for Waist_Yaw_to_Shoulder_Inner vs distal arm links.
    for lower in main_chain:
        for arm_link in left_arm_chain + right_arm_chain:
            if lower == "Link_Waist_Yaw_to_Shoulder_Inner" and any(
                s in arm_link for s in _distal_arm
            ):
                continue
            add(lower, arm_link)

    # Left arm vs right arm (opposite sides)
    for ll in left_arm_chain:
        for rl in right_arm_chain:
            add(ll, rl)

    # Main chain vs neck/head
    for torso in main_chain:
        for neck in neck_chain:
            add(torso, neck)

    # Neck/head vs both arms
    for neck in neck_chain:
        for arm_link in left_arm_chain + right_arm_chain:
            add(neck, arm_link)

    # Gripper sub-links vs all body links and vs each other
    # Keep collision checking for gripper sub-links vs Waist_Yaw_to_Shoulder_Inner.
    for gl in gripper_links:
        for bl in body_links:
            if bl == "Link_Waist_Yaw_to_Shoulder_Inner":
                continue
            add(gl, bl)
        for gl2 in gripper_links:
            if gl != gl2:
                add(gl, gl2)

    # Decorative/sensor links vs all links
    for dl in decorative_links:
        for bl in body_links + gripper_links:
            add(dl, bl)
        for dl2 in decorative_links:
            if dl != dl2:
                add(dl, dl2)

    for a, b in sorted(pairs):
        ET.SubElement(robot, "disable_collisions", link1=a, link2=b, reason="Default")


# ── Stage 5: Copy meshes ────────────────────────────────────────────────────


def copy_meshes(urdf_tree: ET.ElementTree, mesh_src_dir: Path, out_dir: Path) -> None:
    """Copy referenced mesh STL files from source to output meshes directory."""
    mesh_out = out_dir / "meshes"
    mesh_out.mkdir(parents=True, exist_ok=True)

    referenced = set()
    for mesh in urdf_tree.getroot().iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://meshes/"):
            referenced.add(fn.split("/")[-1])

    copied = 0
    for basename in sorted(referenced):
        src = mesh_src_dir / basename
        dst = mesh_out / basename
        if src.exists():
            shutil.copy2(str(src), str(dst))
            copied += 1
        else:
            print(f"  WARNING: mesh not found: {src}")

    print(f"  copied {copied}/{len(referenced)} mesh files")


# ── Stage 6: Repair collision meshes ────────────────────────────────────────


def repair_collision_meshes(simple_urdf_path: Path, method: str = "medial") -> None:
    """Repair meshes so they pass sphere-tree validation (requires foam)."""
    import trimesh
    from foam import smooth_manifold
    from foam.external import check_valid_for_spherization

    tree = ET.parse(str(simple_urdf_path))
    urdf_dir = simple_urdf_path.parent

    mesh_paths: set[Path] = set()
    for mesh_el in tree.getroot().iter("mesh"):
        fn = mesh_el.get("filename", "")
        if fn:
            mesh_paths.add(urdf_dir / fn.replace("package://", ""))

    repaired_count = 0
    for mesh_path in sorted(mesh_paths):
        if not mesh_path.exists():
            continue

        mesh = trimesh.load(str(mesh_path), process=False)

        if check_valid_for_spherization(method, mesh):
            print(f"  OK    {mesh_path.name}")
            continue

        # Stage 1: manifold repair (preserves concavity)
        fixed = None
        strategy = ""
        try:
            repaired = smooth_manifold(mesh)
            if check_valid_for_spherization(method, repaired):
                fixed = repaired
                strategy = "smooth manifold"
        except Exception:
            pass

        # Stage 2: convex hull fallback (always valid)
        if fixed is None:
            fixed = mesh.convex_hull
            strategy = "convex hull"

        fixed.export(str(mesh_path))
        repaired_count += 1
        print(
            f"  REPAIRED  {mesh_path.name}  ({strategy}, {len(fixed.vertices)} verts)"
        )

    print(f"  {repaired_count} mesh(es) repaired out of {len(mesh_paths)} total")


# ── Stage 7: Distribute ─────────────────────────────────────────────────────


def distribute(out_dir: Path, dest_dirs: list[Path]) -> None:
    """Copy generated files to third-party directories."""
    files = [
        "autolife.urdf",
        "autolife_simple.urdf",
        "autolife_base.urdf",
        "autolife_base_simple.urdf",
        "autolife.srdf",
    ]
    for dest_dir in dest_dirs:
        if not dest_dir.exists():
            print(f"  skip {dest_dir} (not found)")
            continue
        for f in files:
            src = out_dir / f
            if src.exists():
                shutil.copy2(str(src), str(dest_dir / f))
        src_meshes = out_dir / "meshes"
        dst_meshes = dest_dir / "meshes"
        if src_meshes.exists():
            if dst_meshes.exists():
                shutil.rmtree(str(dst_meshes))
            shutil.copytree(str(src_meshes), str(dst_meshes))
        print(f"  distributed to {dest_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--urdf", type=Path, required=True, help="Path to raw input URDF"
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        required=True,
        help="Directory containing source mesh STL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--distribute-to",
        type=Path,
        nargs="*",
        default=[],
        help="Third-party directories to copy results into",
    )
    parser.add_argument(
        "--repair-meshes",
        action="store_true",
        help="Repair collision meshes for sphere-tree construction (requires foam)",
    )
    args = parser.parse_args()

    urdf_path: Path = args.urdf
    mesh_src_dir: Path = args.mesh_dir
    out_dir: Path = args.output_dir
    dist_dirs: list[Path] = args.distribute_to

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not mesh_src_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_src_dir}")

    print(f"Input URDF:  {urdf_path}")
    print(f"Mesh source: {mesh_src_dir}")
    print(f"Output dir:  {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_stages = 7 if args.repair_meshes else 6

    print(f"\n[1/{total_stages}] Preprocessing URDF...")
    preprocessed = preprocess_urdf(urdf_path)
    _write_xml(preprocessed, out_dir / "autolife.urdf")

    print(f"\n[2/{total_stages}] Generating simple URDF...")
    simple_tree = generate_simple_urdf(preprocessed)
    _write_xml(simple_tree, out_dir / "autolife_simple.urdf")

    print(f"\n[3/{total_stages}] Generating base URDFs...")
    base_tree = generate_base_urdf(preprocessed)
    _write_xml(base_tree, out_dir / "autolife_base.urdf")
    base_simple_tree = generate_base_urdf(simple_tree)
    _write_xml(base_simple_tree, out_dir / "autolife_base_simple.urdf")

    srdf_path = out_dir / "autolife.srdf"
    if srdf_path.exists():
        print(
            f"\n[4/{total_stages}] SRDF already exists, keeping existing (hand-edited) version."
        )
    else:
        print(f"\n[4/{total_stages}] Generating SRDF...")
        srdf_tree = generate_srdf(base_simple_tree)
        _write_xml(srdf_tree, srdf_path)

    print(f"\n[5/{total_stages}] Copying meshes...")
    copy_meshes(preprocessed, mesh_src_dir, out_dir)

    if args.repair_meshes:
        print(f"\n[6/{total_stages}] Repairing collision meshes...")
        repair_collision_meshes(out_dir / "autolife_base_simple.urdf")

    last = total_stages
    if dist_dirs:
        print(f"\n[{last}/{total_stages}] Distributing to third_party...")
        distribute(out_dir, dist_dirs)
    else:
        print(f"\n[{last}/{total_stages}] No distribute targets specified, skipping.")

    # Verify joint count
    rev_joints = [
        j.get("name")
        for j in preprocessed.getroot().findall("joint")
        if j.get("type") in ("revolute", "prismatic")
        and not _is_gripper_joint(j.get("name", ""))
    ]
    print(f"\nPlanning joints ({len(rev_joints)} DOF): {rev_joints}")
    assert len(rev_joints) == 21, f"Expected 21 planning DOF, got {len(rev_joints)}"
    print("\nDone!")


if __name__ == "__main__":
    main()
