import argparse
import glob
import logging
import os
import shutil

import pymeshlab

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MeshSimplifier")


def simplify_meshes(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Find all OBJ files
    obj_files = glob.glob(os.path.join(src_dir, "**/*.obj"), recursive=True)
    logger.info(f"Found {len(obj_files)} OBJ files in {src_dir}")

    for file_path in obj_files:
        rel_path = os.path.relpath(file_path, src_dir)
        out_path = os.path.join(dst_dir, rel_path)
        out_folder = os.path.dirname(out_path)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        logger.info(f"Processing: {rel_path}...")

        # Initialize MeshSet
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(file_path)

        # Check properties
        num_faces = ms.current_mesh().face_number()
        logger.info(f"  - Original triangles: {num_faces}")

        if num_faces / 4 < 50000:
            logger.info(f"  - Keeping original (faces {num_faces} < 200000)")
        else:
            target_triangles = int(num_faces / 4)
            # Decimate with texture preservation
            # Use specific filter for textured meshes
            ms.apply_filter(
                "meshing_decimation_quadric_edge_collapse_with_texture",
                targetfacenum=target_triangles,
                preserveboundary=True,
                planarquadric=True,
            )
            new_faces = ms.current_mesh().face_number()
            logger.info(f"  - Simplified triangles: {new_faces}")

        # Save simplified mesh with texture
        ms.save_current_mesh(out_path)

        # Copy texture and mtl files manually
        # We overwrite any existing files to ensure we use the originals
        src_folder = os.path.dirname(file_path)
        for other_file in os.listdir(src_folder):
            if other_file.endswith((".png", ".jpg", ".jpeg", ".tga", ".mtl")):
                src_other = os.path.join(src_folder, other_file)
                dst_other = os.path.join(out_folder, other_file)
                shutil.copy2(src_other, dst_other)
                logger.info(f"  - Copied {other_file}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Simplify meshes for PyBullet")
    parser.add_argument(
        "--src",
        type=str,
        default=os.path.join(project_root, "assets", "envs", "rls_env", "meshes"),
        help="Source directory",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=os.path.join(
            project_root, "assets", "envs", "rls_env", "simplified_meshes"
        ),
        help="Destination directory",
    )

    args = parser.parse_args()

    simplify_meshes(args.src, args.dst)
