import subprocess
import os
import shutil
import struct
import numpy as np
from pathlib import Path
import sys

# ── scikit-learn: DBSCAN (density-based floater removal)
from sklearn.cluster import DBSCAN

# ── scipy: RANSAC plane fitting (geometry-inconsistent point removal)
from scipy.spatial import cKDTree

# NOTE: Open3D removed entirely. No more sys.modules hack needed.


# ───────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────

def run_step(command, cwd):
    print(f"\n[RUNNING]: {' '.join(command)}")
    
    # 1. Copy the current environment variables
    custom_env = os.environ.copy()
    
    # 2. Add the fix for the MKL/OpenMP conflict
    custom_env["MKL_THREADING_LAYER"] = "GNU"
    
    result = subprocess.run(
        command, 
        cwd=cwd,
        env=custom_env,  # 3. Pass the modified environment to the subprocess
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    if result.stdout:
        print(result.stdout[-3000:])
        
    if result.returncode != 0:
        error_details = result.stdout[-1500:] if result.stdout else "No output captured."
        raise RuntimeError(
            f"Step failed: {command[0]} (Exit code: {result.returncode})\n\n"
            f"--- FASTGS ERROR DETAILS ---\n{error_details}"
        )


def copy_sparse_txt(dense_dir: Path):
    sparse_src = dense_dir / "sparse"
    sparse_dst = dense_dir / "sparse" / "0"
    os.makedirs(sparse_dst, exist_ok=True)
    for txt_file in sparse_src.glob("*.txt"):
        shutil.copy(txt_file, sparse_dst / txt_file.name)

    for required in ["cameras.txt", "images.txt", "points3D.txt"]:
        p = sparse_dst / required
        if not p.exists():
            raise RuntimeError(f"FastGS prerequisite missing: {required}")


# ───────────────────────────────────────────────────────────
# PLY I/O  (reused by clean_splat)
# ───────────────────────────────────────────────────────────

def _read_ply(input_path: Path):
    """
    Returns (header_lines, data np.ndarray, properties list, num_vertices).
    All floats, binary_little_endian only (matches FastGS output).
    """
    with open(input_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break

        num_vertices = 0
        properties = []
        for line in header_lines:
            s = line.decode("utf-8").strip()
            if s.startswith("element vertex"):
                num_vertices = int(s.split()[-1])
            if s.startswith("property float"):
                properties.append(s.split()[-1])

        num_props = len(properties)
        data = []
        for _ in range(num_vertices):
            raw = f.read(4 * num_props)
            vals = struct.unpack_from(f"{num_props}f", raw)
            data.append(vals)

    return header_lines, np.array(data), properties, num_vertices


def _write_ply(output_path: Path, header_lines: list, final_data: np.ndarray, num_props: int):
    with open(output_path, "wb") as f:
        for line in header_lines:
            s = line.decode("utf-8").strip()
            if s.startswith("element vertex"):
                f.write(f"element vertex {len(final_data)}\n".encode("utf-8"))
            else:
                f.write(line)
        for row in final_data:
            f.write(struct.pack(f"{num_props}f", *row))


# ───────────────────────────────────────────────────────────
# RANSAC PLANE FITTER
# Fits up to `max_planes` dominant planes from the point cloud.
# Returns a boolean mask — True = consistent with at least one plane.
# ───────────────────────────────────────────────────────────

def _ransac_plane_inlier_mask(
    xyz: np.ndarray,
    distance_threshold: float = 0.05,
    num_iterations: int = 200,
    max_planes: int = 3,
    min_inlier_ratio: float = 0.03,
) -> np.ndarray:
    """
    Iteratively fits dominant planes (RANSAC) and marks all inliers.
    Points that belong to at least one plane are considered geometrically
    consistent with the scene structure.

    Parameters
    ----------
    distance_threshold  : max point-to-plane distance to count as inlier (scene units)
    num_iterations      : RANSAC iterations per plane fit
    max_planes          : how many dominant planes to extract
    min_inlier_ratio    : stop early if best plane explains < this fraction of remaining pts

    Returns
    -------
    consistent_mask : bool array, True = belongs to a fitted plane
    """
    remaining = np.arange(len(xyz))
    consistent_mask = np.zeros(len(xyz), dtype=bool)

    for plane_idx in range(max_planes):
        if len(remaining) < 50:
            break

        pts = xyz[remaining]
        best_inliers_local = np.array([], dtype=int)
        best_count = 0

        for _ in range(num_iterations):
            # Sample 3 points to define a plane
            idx = np.random.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]

            # Plane normal via cross product
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8:
                continue
            normal = normal / norm_len

            # Signed distance of every remaining point to this plane
            dists = np.abs((pts - p0) @ normal)
            inliers_local = np.where(dists < distance_threshold)[0]

            if len(inliers_local) > best_count:
                best_count = len(inliers_local)
                best_inliers_local = inliers_local

        ratio = best_count / len(pts)
        if ratio < min_inlier_ratio:
            print(f"    -> RANSAC plane {plane_idx + 1}: only {ratio:.1%} inliers — stopping early.")
            break

        # Mark these global indices as consistent
        global_inliers = remaining[best_inliers_local]
        consistent_mask[global_inliers] = True

        # Remove inliers from remaining set for next plane search
        remaining = np.setdiff1d(remaining, global_inliers)
        print(f"    -> RANSAC plane {plane_idx + 1}: {best_count} inliers ({ratio:.1%} of remaining).")

    return consistent_mask


# ───────────────────────────────────────────────────────────
# MAIN CLEANER
# ───────────────────────────────────────────────────────────

def clean_splat(input_path: Path, output_path: Path):
    print(f"\n[CLEANING SPLAT]: {input_path}")

    header_lines, data, properties, num_vertices = _read_ply(input_path)
    num_props = len(properties)

    x_i = properties.index("x")
    y_i = properties.index("y")
    z_i = properties.index("z")
    o_i = properties.index("opacity")

    # ── Resolve scale properties (FastGS stores them as scale_0/1/2)
    scale_props = [p for p in properties if p.startswith("scale_")]
    has_scales = len(scale_props) >= 3

    # ── Compute a rough scene-scale reference for adaptive thresholds
    xyz = data[:, [x_i, y_i, z_i]]
    scene_extent = np.percentile(
        np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1), 90
    )
    print(f"  Scene extent (90th-pct radius): {scene_extent:.4f} units")

    # ════════════════════════════════════════════════════════
    # STAGE 1 — SCALE FILTER  (property-level, cheapest)
    # Kill Gaussians with degenerate / bloated scales.
    # Runs before spatial operations to shrink the working set.
    # ════════════════════════════════════════════════════════
    if has_scales:
        print("  [Stage 1] Scale + Anisotropy Filter...")
        s_indices = [properties.index(p) for p in scale_props[:3]]
        scales = data[:, s_indices]  # shape (N, 3)

        # FastGS stores log-scales — exponentiate to get actual axis lengths
        scales_exp = np.exp(scales)
        max_scale = scales_exp.max(axis=1)
        min_scale = scales_exp.min(axis=1)

        # Bloated Gaussians: max axis > 5% of scene extent  →  floater
        bloat_threshold = 0.08 * scene_extent
        bloat_mask = max_scale < bloat_threshold

        # Degenerate Gaussians: aspect ratio > 30:1  →  needle artifact
        aspect_ratio = max_scale / (min_scale + 1e-8)
        aspect_mask = aspect_ratio < 50.0

        # Tiny near-zero Gaussians: max axis < 0.01% of scene  →  noise dust
        dust_threshold = 0.0001 * scene_extent
        dust_mask = max_scale > dust_threshold

        scale_mask = bloat_mask & aspect_mask & dust_mask
        cleaned_data = data[scale_mask]
        print(f"    -> Removed {np.sum(~scale_mask)} scale-degenerate Gaussians "
              f"({np.sum(~bloat_mask)} bloated, {np.sum(~aspect_mask)} needle, "
              f"{np.sum(~dust_mask)} dust).")
    else:
        print("  [Stage 1] Scale properties not found — skipping scale filter.")
        cleaned_data = data.copy()

    # ════════════════════════════════════════════════════════
    # STAGE 2 — OPACITY FILTER
    # Remove near-transparent fog / unoptimized Gaussians.
    # ════════════════════════════════════════════════════════
    print("  [Stage 2] Opacity Filter...")
    raw_opacity = cleaned_data[:, o_i]

    # Sigmoid of raw opacity gives the actual [0,1] opacity
    opacity_sigmoid = 1.0 / (1.0 + np.exp(-raw_opacity))

    # Keep only Gaussians with opacity > 5th percentile of the scene
    # (more conservative than the old 15th — Stage 1 has already trimmed bloat)
    op_threshold = np.percentile(opacity_sigmoid, 5)
    op_mask = opacity_sigmoid > op_threshold
    cleaned_data = cleaned_data[op_mask]
    print(f"    -> Removed {np.sum(~op_mask)} transparent Gaussians (threshold={op_threshold:.4f}).")

    # ════════════════════════════════════════════════════════
    # STAGE 3 — RADIAL CROP
    # Trim the outer explosion. Kept after opacity so the
    # percentile is computed on a cleaner distribution.
    # ════════════════════════════════════════════════════════
    print("  [Stage 3] Radial Crop...")
    xyz_clean = cleaned_data[:, [x_i, y_i, z_i]]
    center = np.median(xyz_clean, axis=0)
    distances = np.linalg.norm(xyz_clean - center, axis=1)

    # 80th percentile: slightly looser than old 75th because earlier stages
    # have already removed most floaters — safer to keep real scene edges.
    radius_limit = np.percentile(distances, 80)
    radial_mask = distances < radius_limit
    cleaned_data = cleaned_data[radial_mask]
    print(f"    -> Removed {np.sum(~radial_mask)} outer-explosion points "
          f"(radius limit={radius_limit:.4f}).")

    # ════════════════════════════════════════════════════════
    # STAGE 4 — DBSCAN  (density-based cluster isolation)
    # Kills floating debris clusters that survived the above.
    # Keeps only the largest connected spatial component(s).
    # ════════════════════════════════════════════════════════
    print("  [Stage 4] DBSCAN Cluster Isolation...")
    xyz_s4 = cleaned_data[:, [x_i, y_i, z_i]]

    # eps: neighbourhood radius. We use 1% of scene extent as a reasonable
    # adaptive default. Tune upward if real geometry gets fragmented.
    eps = 0.02 * scene_extent
    min_samples = 15  # minimum neighbours to be a core point

    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree", n_jobs=-1).fit(xyz_s4)
    labels = db.labels_  # -1 = noise

    # Count population per cluster, sort descending
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(counts) == 0:
        print("    -> DBSCAN found no clusters — skipping. Consider increasing eps.")
        dbscan_mask = labels >= 0  # at least remove noise points
    else:
        sorted_idx = np.argsort(-counts)
        unique_labels = unique_labels[sorted_idx]
        counts = counts[sorted_idx]

        # Keep clusters that are > 1% of the largest cluster's population
        # This retains secondary objects (second body, weapon, etc.) while
        # killing tiny floating debris clusters.
        size_threshold = 0.005 * counts[0]
        kept_labels = set(unique_labels[counts >= size_threshold].tolist())

        dbscan_mask = np.array([lbl in kept_labels for lbl in labels])
        noise_removed = np.sum(labels == -1)
        cluster_removed = np.sum(labels >= 0) - np.sum(dbscan_mask)
        print(f"    -> DBSCAN: kept {len(kept_labels)} cluster(s), "
              f"removed {noise_removed} noise points + {cluster_removed} small-cluster floaters.")

    cleaned_data = cleaned_data[dbscan_mask]

    # ════════════════════════════════════════════════════════
    # STAGE 5 — RANSAC PLANE CONSISTENCY CHECK  (optional)
    # Flags Gaussians that are inconsistent with dominant scene
    # planes (floor, wall, table). Most useful for indoor scenes.
    # Set USE_RANSAC = False if your scene is organic / non-planar.
    # ════════════════════════════════════════════════════════
    USE_RANSAC = False

    if USE_RANSAC:
        print("  [Stage 5] RANSAC Plane Consistency...")
        xyz_s5 = cleaned_data[:, [x_i, y_i, z_i]]

        # distance_threshold: 2% of scene extent — tighter means more aggressive
        plane_mask = _ransac_plane_inlier_mask(
            xyz_s5,
            distance_threshold=0.04 * scene_extent,
            num_iterations=300,
            max_planes=4,
            min_inlier_ratio=0.02,
        )

        # RANSAC is a soft signal — only remove points that are BOTH plane-
        # inconsistent AND spatially isolated (low local density). This avoids
        # over-removing genuine curved/organic geometry.
        tree = cKDTree(xyz_s5)
        neighbour_counts = np.array([
            len(tree.query_ball_point(p, r=0.02 * scene_extent)) - 1
            for p in xyz_s5
        ])
        # A point is "safe to remove" only if it fails RANSAC AND has few neighbours
        isolated = neighbour_counts < 10
        ransac_remove = (~plane_mask) & isolated

        cleaned_data = cleaned_data[~ransac_remove]
        print(f"    -> RANSAC: removed {np.sum(ransac_remove)} geometry-inconsistent "
              f"isolated points.")
    else:
        print("  [Stage 5] RANSAC skipped (USE_RANSAC=False).")

    # ════════════════════════════════════════════════════════
    # WRITE OUTPUT
    # ════════════════════════════════════════════════════════
    removed_total = num_vertices - len(cleaned_data)
    kept_pct = 100 * len(cleaned_data) / num_vertices
    print(f"\n  [CLEAN COMPLETE] {len(cleaned_data)} / {num_vertices} Gaussians kept "
          f"({kept_pct:.1f}%) — removed {removed_total} artifacts.")

    _write_ply(output_path, header_lines, cleaned_data, num_props)
    print(f"  Saved cleaned splat → {output_path}")


# ───────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ───────────────────────────────────────────────────────────

def run_pipeline(job_id: str, video_path: Path, session_dir: Path, jobs: dict):
    try:
        images_dir = session_dir / "images"
        db_path    = session_dir / "database.db"
        sparse_dir = session_dir / "sparse"
        dense_dir  = session_dir / "dense"
        output_dir = session_dir / "output"

        run_step([
            "ffmpeg", "-i", str(video_path),
            "-qscale:v", "1",
            "-vf", "fps=3,scale=800:800:force_original_aspect_ratio=decrease",
            str(images_dir / "%04d.jpg")
        ], session_dir)

        run_step(["colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", "OPENCV",
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.max_num_features", "16384"
        ], session_dir)

        run_step(["colmap", "sequential_matcher",
            "--database_path", str(db_path)
        ], session_dir)

        os.makedirs(sparse_dir / "0", exist_ok=True)
        run_step(["colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.init_min_num_inliers", "10"
        ], session_dir)

        run_step(["colmap", "model_converter",
            "--input_path", str(sparse_dir / "0"),
            "--output_path", str(sparse_dir / "0"),
            "--output_type", "TXT"
        ], session_dir)

        os.makedirs(dense_dir, exist_ok=True)
        run_step(["colmap", "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(sparse_dir / "0"),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP"
        ], session_dir)

        run_step(["colmap", "model_converter",
            "--input_path", str(dense_dir / "sparse"),
            "--output_path", str(dense_dir / "sparse"),
            "--output_type", "TXT"
        ], session_dir)

        copy_sparse_txt(dense_dir)

        fastgs_dir = Path("/home/cave/3dapp/FastGS")
        os.makedirs(output_dir, exist_ok=True)

        run_step([
            "/home/cave/miniconda3/envs/fastgs/bin/python", "train.py",
            "-s", str(dense_dir),
            "-m", str(output_dir),
            "--iterations", "10000"
        ], fastgs_dir)
        
        raw_ply     = output_dir / "point_cloud" / "iteration_10000" / "point_cloud.ply"
        cleaned_ply = session_dir / "point_cloud.ply"
        
        # 1. Clean the raw output
        clean_splat(raw_ply, cleaned_ply)

        # 2. Compress the cleaned output to .sog using SplatTransform
        sog_path = session_dir / "optimized_scene.sog"
        print("Starting SplatTransform compression...")
        try:
            subprocess.run([
                "splat-transform", 
                str(cleaned_ply), 
                str(sog_path)
            ], check=True)
            print(f"Optimization complete. Saved to: {sog_path}")
        except subprocess.CalledProcessError as e:
            print(f"Optimization failed: {e}")

        jobs[job_id] = "done"
        return cleaned_ply

    except Exception as e:
        jobs[job_id] = "failed"
        raise


def run_pipeline_from_images(job_id: str, session_dir: Path, jobs: dict):
    try:
        images_dir = session_dir / "images"
        db_path    = session_dir / "database.db"
        sparse_dir = session_dir / "sparse"
        dense_dir  = session_dir / "dense"
        output_dir = session_dir / "output"

        run_step(["colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", "OPENCV",
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.max_num_features", "16384"
        ], session_dir)

        run_step(["colmap", "sequential_matcher",
            "--database_path", str(db_path)
        ], session_dir)

        os.makedirs(sparse_dir / "0", exist_ok=True)
        run_step(["colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.init_min_num_inliers", "10"
        ], session_dir)

        run_step(["colmap", "model_converter",
            "--input_path", str(sparse_dir / "0"),
            "--output_path", str(sparse_dir / "0"),
            "--output_type", "TXT"
        ], session_dir)

        os.makedirs(dense_dir, exist_ok=True)
        run_step(["colmap", "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(sparse_dir / "0"),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP"
        ], session_dir)

        run_step(["colmap", "model_converter",
            "--input_path", str(dense_dir / "sparse"),
            "--output_path", str(dense_dir / "sparse"),
            "--output_type", "TXT"
        ], session_dir)

        copy_sparse_txt(dense_dir)

        fastgs_dir = Path("/home/cave/3dapp/FastGS")
        os.makedirs(output_dir, exist_ok=True)

        run_step([
            "/home/cave/miniconda3/envs/fastgs/bin/python", "train.py",
            "-s", str(dense_dir),
            "-m", str(output_dir),
            "--iterations", "10000"
        ], fastgs_dir)

        raw_ply     = output_dir / "point_cloud" / "iteration_10000" / "point_cloud.ply"
        cleaned_ply = session_dir / "point_cloud.ply"
        
        # 1. Clean the raw output
        clean_splat(raw_ply, cleaned_ply)

        # 2. Compress the cleaned output to .sog using SplatTransform
        sog_path = session_dir / "optimized_scene.sog"
        print("Starting SplatTransform compression...")
        try:
            subprocess.run([
                "splat-transform", 
                str(cleaned_ply), 
                str(sog_path)
            ], check=True)
            print(f"Optimization complete. Saved to: {sog_path}")
        except subprocess.CalledProcessError as e:
            print(f"Optimization failed: {e}")

        jobs[job_id] = "done"
        return cleaned_ply

    except Exception as e:
        jobs[job_id] = "failed"
        raise
