import subprocess
import os
import shutil
import struct
import numpy as np
from pathlib import Path
import sys
from backend.persistence import save_job
from backend import metrics as m
from backend.cleanup import audit_log
from backend.cleanup.statistical_filters import filter_opacity_mad
from backend.segmentation import segment_object
from backend.colmap_utils import get_reliable_colmap_points, filter_gaussians_by_reliability, get_camera_centers


# ── scikit-learn: DBSCAN (density-based floater removal)
from sklearn.cluster import DBSCAN

# ── scipy: RANSAC plane fitting (geometry-inconsistent point removal)
from scipy.spatial import cKDTree
SEGMENT_OBJECT = True

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


    
    # 2. Fix threading and display conflicts
    custom_env["MKL_THREADING_LAYER"] = "GNU"
    custom_env["QT_QPA_PLATFORM"] = "offscreen"  # Prevents headless COLMAP crash
    

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
# JOB DICT HELPER
# jobs[job_id] is always a dict ({id, title, status, progress, modelUrl}),
# created in api_routes.py. EVERY pipeline stage MUST go through this
# helper to mutate it — never do `jobs[job_id] = "some_string"` directly,
# or you will clobber the dict and break every route that reads jobs[].
#
# This helper is also self-healing: if jobs[job_id] is ever found to be
# something other than a dict (e.g. a leftover bad assignment), it will
# reset it to a fresh dict rather than crashing with
# "'str' object does not support item assignment".
# ───────────────────────────────────────────────────────────

def _set(jobs: dict, job_id: str, status: str = None, progress: int = None, **extra):
    existing = jobs.get(job_id)
    if not isinstance(existing, dict):
        jobs[job_id] = {
            "id": job_id,
            "title": None,
            "status": None,
            "progress": 0,
            "modelUrl": None,
        }
    if status is not None:
        jobs[job_id]["status"] = status
    if progress is not None:
        jobs[job_id]["progress"] = progress
    if extra:
        jobs[job_id].update(extra)

    save_job(jobs, job_id)   # <-- now takes jobs dict + job_id

def _gather_quality_metrics(session_dir: Path, dense_dir: Path, output_dir: Path):
    metrics = {}

    render_metrics = m.compute_render_quality_metrics(output_dir, iteration=10000)
    if render_metrics:
        metrics.update(render_metrics)

    points3d_txt = dense_dir / "sparse" / "points3D.txt"
    colmap_metrics = m.colmap_reprojection_and_reliability(points3d_txt)
    if colmap_metrics:
        metrics["colmap"] = colmap_metrics

    return metrics or None
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
            idx = np.random.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]

            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8:
                continue
            normal = normal / norm_len

            dists = np.abs((pts - p0) @ normal)
            inliers_local = np.where(dists < distance_threshold)[0]

            if len(inliers_local) > best_count:
                best_count = len(inliers_local)
                best_inliers_local = inliers_local

        ratio = best_count / len(pts)
        if ratio < min_inlier_ratio:
            print(f"    -> RANSAC plane {plane_idx + 1}: only {ratio:.1%} inliers — stopping early.")
            break

        global_inliers = remaining[best_inliers_local]
        consistent_mask[global_inliers] = True

        remaining = np.setdiff1d(remaining, global_inliers)
        print(f"    -> RANSAC plane {plane_idx + 1}: {best_count} inliers ({ratio:.1%} of remaining).")

    return consistent_mask

# ───────────────────────────────────────────────────────────
# ADAPTIVE EPS  (k-distance elbow method for DBSCAN)
# ───────────────────────────────────────────────────────────

def _estimate_eps_via_kdistance(xyz: np.ndarray, k: int = 15) -> float:
    """
    Estimates a good DBSCAN eps by finding the 'elbow' in the sorted
    k-th nearest neighbour distance curve, instead of a fixed fraction
    of scene extent (which breaks across differing point densities).
    """
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k + 1)  # +1 because the point itself is included
    k_dists = np.sort(dists[:, -1])

    # Elbow = point of maximum curvature on the sorted k-distance curve
    n = len(k_dists)
    x = np.arange(n)
    # normalize both axes to [0,1] so curvature isn't scale-dependent
    x_norm = x / (n - 1)
    y_norm = (k_dists - k_dists.min()) / (k_dists.max() - k_dists.min() + 1e-12)

    # distance from each point to the line connecting first and last point
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    pts = np.stack([x_norm, y_norm], axis=1) - p1
    proj_len = pts @ line_vec_norm
    proj_pts = np.outer(proj_len, line_vec_norm)
    perp_dist = np.linalg.norm(pts - proj_pts, axis=1)

    elbow_idx = np.argmax(perp_dist)
    return float(k_dists[elbow_idx])
   
  
# ───────────────────────────────────────────────────────────
# MAIN CLEANER
# ───────────────────────────────────────────────────────────

def clean_splat(input_path: Path, output_path: Path):
    print(f"\n[CLEANING SPLAT]: {input_path}")
    audit_log.start_audit_log(output_path.parent, input_path)
            
    header_lines, data, properties, num_vertices = _read_ply(input_path)
    num_props = len(properties)

    x_i = properties.index("x")
    y_i = properties.index("y")
    z_i = properties.index("z")
    o_i = properties.index("opacity")

    scale_props = [p for p in properties if p.startswith("scale_")]
    has_scales = len(scale_props) >= 3

    xyz = data[:, [x_i, y_i, z_i]]
    scene_extent = np.percentile(
        np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1), 90
    )
    print(f"  Scene extent (90th-pct radius): {scene_extent:.4f} units")

    # STAGE 1 — SCALE FILTER
    if has_scales:
        print("  [Stage 1] Scale + Anisotropy Filter...")
        s_indices = [properties.index(p) for p in scale_props[:3]]
        scales = data[:, s_indices]

        scales_exp = np.exp(scales)
        max_scale = scales_exp.max(axis=1)
        min_scale = scales_exp.min(axis=1)

        bloat_threshold = 0.08 * scene_extent
        bloat_mask = max_scale < bloat_threshold

        aspect_ratio = max_scale / (min_scale + 1e-8)
        aspect_mask = aspect_ratio < 50.0

        dust_threshold = 0.0001 * scene_extent
        dust_mask = max_scale > dust_threshold

        scale_mask = bloat_mask & aspect_mask & dust_mask
        cleaned_data = data[scale_mask]
        print(f"    -> Removed {np.sum(~scale_mask)} scale-degenerate Gaussians "
              f"({np.sum(~bloat_mask)} bloated, {np.sum(~aspect_mask)} needle, "
              f"{np.sum(~dust_mask)} dust).")
        audit_log.log_removal(output_path.parent, "Stage 1 - Scale Filter", "bloated (max axis over threshold)", int(np.sum(~bloat_mask)), threshold=float(bloat_threshold))
        audit_log.log_removal(output_path.parent, "Stage 1 - Scale Filter", "needle (aspect ratio over threshold)", int(np.sum(~aspect_mask)), threshold=50.0)
        audit_log.log_removal(output_path.parent, "Stage 1 - Scale Filter", "dust (max axis under threshold)", int(np.sum(~dust_mask)), threshold=float(dust_threshold))
    else:
        print("  [Stage 1] Scale properties not found — skipping scale filter.")
        cleaned_data = data.copy()


    # STAGE 2 — OPACITY FILTER
    print("  [Stage 2] Opacity Filter...")
    raw_opacity = cleaned_data[:, o_i]
    opacity_sigmoid = 1.0 / (1.0 + np.exp(-raw_opacity))
    op_threshold = np.percentile(opacity_sigmoid, 5)
    op_mask = opacity_sigmoid > op_threshold
    cleaned_data = cleaned_data[op_mask]
    print(f"    -> Removed {np.sum(~op_mask)} transparent Gaussians (threshold={op_threshold:.4f}).")

    # ════════════════════════════════════════════════════════
    # STAGE 2 — OPACITY FILTER (MAD-based adaptive threshold)
    # Remove near-transparent fog / unoptimized Gaussians using a
    # robust Median Absolute Deviation cut instead of a fixed
    # percentile (which drops a constant 5% regardless of how much
    # of the scene is actually contaminated).
    # ════════════════════════════════════════════════════════
    cleaned_data = filter_opacity_mad(cleaned_data, o_i, output_path)


    # STAGE 3 — RADIAL CROP
    print("  [Stage 3] Radial Crop...")
    xyz_clean = cleaned_data[:, [x_i, y_i, z_i]]
    center = np.median(xyz_clean, axis=0)
    distances = np.linalg.norm(xyz_clean - center, axis=1)
    radius_limit = np.percentile(distances, 80)
    radial_mask = distances < radius_limit
    cleaned_data = cleaned_data[radial_mask]
    print(f"    -> Removed {np.sum(~radial_mask)} outer-explosion points "
          f"(radius limit={radius_limit:.4f}).")
    audit_log.log_removal(output_path.parent, "Stage 3 - Radial Crop", "distance beyond 80th percentile radius", int(np.sum(~radial_mask)), threshold=float(radius_limit))

    # STAGE 4 — DBSCAN
    print("  [Stage 4] DBSCAN Cluster Isolation...")
    xyz_s4 = cleaned_data[:, [x_i, y_i, z_i]]

    eps = 0.02 * scene_extent
    min_samples = 15


    # eps: neighbourhood radius. We use 1% of scene extent as a reasonable
    # adaptive default. Tune upward if real geometry gets fragmented.
    min_samples = 15  # minimum neighbours to be a core point
    eps = _estimate_eps_via_kdistance(xyz_s4, k=min_samples)


    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree", n_jobs=-1).fit(xyz_s4)
    labels = db.labels_

    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(counts) == 0:
        print("    -> DBSCAN found no clusters — skipping. Consider increasing eps.")
        dbscan_mask = labels >= 0
    else:
        sorted_idx = np.argsort(-counts)
        unique_labels = unique_labels[sorted_idx]
        counts = counts[sorted_idx]


        size_threshold = 0.005 * counts[0]

        # Keep clusters that are > 1% of the largest cluster's population
        size_threshold = 0.0015 * counts[0]

        kept_labels = set(unique_labels[counts >= size_threshold].tolist())

        dbscan_mask = np.array([lbl in kept_labels for lbl in labels])
        noise_removed = np.sum(labels == -1)
        cluster_removed = np.sum(labels >= 0) - np.sum(dbscan_mask)
        print(f"    -> DBSCAN: kept {len(kept_labels)} cluster(s), "
              f"removed {noise_removed} noise points + {cluster_removed} small-cluster floaters.")

    cleaned_data = cleaned_data[dbscan_mask]


    # STAGE 5 — RANSAC (optional)

    if len(counts) > 0:
        audit_log.log_removal(output_path.parent, "Stage 4 - DBSCAN", "noise points (no cluster)", int(noise_removed), threshold=float(eps))
        audit_log.log_removal(output_path.parent, "Stage 4 - DBSCAN", "small floater clusters below size threshold", int(cluster_removed), threshold=float(size_threshold))
        
    # ════════════════════════════════════════════════════════
    # STAGE 5 — RANSAC PLANE CONSISTENCY CHECK  (optional)
    # ════════════════════════════════════════════════════════

    USE_RANSAC = False
    USE_COLOR_FILTER = False

    if USE_RANSAC:
        print("  [Stage 5] RANSAC Plane Consistency...")
        xyz_s5 = cleaned_data[:, [x_i, y_i, z_i]]

        plane_mask = _ransac_plane_inlier_mask(
            xyz_s5,
            distance_threshold=0.04 * scene_extent,
            num_iterations=300,
            max_planes=4,
            min_inlier_ratio=0.02,
        )

        tree = cKDTree(xyz_s5)
        neighbour_counts = np.array([
            len(tree.query_ball_point(p, r=0.02 * scene_extent)) - 1
            for p in xyz_s5
        ])
        isolated = neighbour_counts < 10
        ransac_remove = (~plane_mask) & isolated

        cleaned_data = cleaned_data[~ransac_remove]
        print(f"    -> RANSAC: removed {np.sum(ransac_remove)} geometry-inconsistent "
              f"isolated points.")
        audit_log.log_removal(output_path.parent, "Stage 5 - RANSAC Plane Consistency", "plane-inconsistent and spatially isolated", int(np.sum(ransac_remove)), threshold=None)
    else:
        print("  [Stage 5] RANSAC skipped (USE_RANSAC=False).")
    
    if USE_COLOR_FILTER:
        from backend.cleanup.appearance_filters import filter_color_consistency
        cleaned_data = filter_color_consistency(cleaned_data, properties, [x_i, y_i, z_i], output_path)

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
    _set(jobs, job_id, status="processing", progress=0)

    images_dir = session_dir / "images"
    db_path    = session_dir / "database.db"
    sparse_dir = session_dir / "sparse"
    dense_dir  = session_dir / "dense"
    output_dir = session_dir / "output"

    # --- STAGE 1: Video Extraction (FFmpeg) ---
    try:
        _set(jobs, job_id, status="colmap", progress=5)

        run_step([
            "ffmpeg", "-i", str(video_path),
            "-qscale:v", "1",
            "-vf", "fps=3,scale=800:800:force_original_aspect_ratio=decrease",
            str(images_dir / "%04d.jpg")
        ], session_dir)

        _set(jobs, job_id, progress=10)

    except Exception as e:
        print(f"\n[ERROR] FFmpeg extraction failed for job {job_id}: {e}")
        _set(jobs, job_id, status="failed_ffmpeg")
        raise RuntimeError(f"FFmpeg extraction stage failed: {e}") from e


    # --- STAGE 2: Structure from Motion (COLMAP) ---
    _set(jobs, job_id, status="colmap")
    try:
        run_step(["colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", "OPENCV",
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.max_num_features", "16384"
        ], session_dir)
        _set(jobs, job_id, progress=20)

        run_step(["colmap", "sequential_matcher",
            "--database_path", str(db_path)
        ], session_dir)
        _set(jobs, job_id, progress=25)

        os.makedirs(sparse_dir / "0", exist_ok=True)
        run_step(["colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.init_min_num_inliers", "10"
        ], session_dir)
        _set(jobs, job_id, progress=35)

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
        _set(jobs, job_id, progress=45)

        run_step(["colmap", "model_converter",
            "--input_path", str(dense_dir / "sparse"),
            "--output_path", str(dense_dir / "sparse"),
            "--output_type", "TXT"
        ], session_dir)

        copy_sparse_txt(dense_dir)

        _set(jobs, job_id, status="fastgs", progress=50)

    except Exception as e:
        print(f"\n[ERROR] COLMAP reconstruction failed for job {job_id}: {e}")
        _set(jobs, job_id, status="failed_colmap")
        raise RuntimeError(f"COLMAP stage failed: {e}") from e


    # --- STAGE 3: 3D Gaussian Splatting (FastGS) ---
    _set(jobs, job_id, status="fastgs")
    try:
        fastgs_dir = Path(os.environ.get("FASTGS_DIR", "/home/cave/3dapp/FastGS"))
        fastgs_python = os.environ.get("FASTGS_PYTHON", "/home/cave/miniconda3/envs/fastgs/bin/python")
        os.makedirs(output_dir, exist_ok=True)

        run_step([
            fastgs_python, "train.py",
            "-s", str(dense_dir),
            "-m", str(output_dir),
            "--iterations", "10000",
            "--eval"
        ], fastgs_dir)
        
        try:
            run_step([
                fastgs_python, "render.py",
                "-m", str(output_dir),
                "--iteration", "10000",
                "--skip_train"
            ], fastgs_dir)
        except Exception as e:
            print(f"[WARNING] render.py failed for job {job_id}: {e} — PSNR/SSIM will be unavailable.")

        _set(jobs, job_id, status="post_processing", progress=80)

        raw_ply     = output_dir / "point_cloud" / "iteration_10000" / "point_cloud.ply"
        cleaned_ply = session_dir / "point_cloud.ply"

        clean_splat(raw_ply, cleaned_ply)
        _set(jobs, job_id, progress=90)
        
        quality_metrics = _gather_quality_metrics(session_dir, dense_dir, output_dir)
        if quality_metrics:
            _set(jobs, job_id, metrics=quality_metrics)

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

        _set(
            jobs, job_id,
            status="done",
            progress=100,
            modelUrl=f"/api/download/{job_id}/point_cloud.ply",
        )
        return cleaned_ply

    except Exception as e:
        print(f"\n[ERROR] FastGS training / post-processing failed for job {job_id}: {e}")
        _set(jobs, job_id, status="failed_fastgs")
        raise RuntimeError(f"FastGS training stage failed: {e}") from e


def run_pipeline_from_images(job_id: str, session_dir: Path, jobs: dict):
    _set(jobs, job_id, status="processing", progress=0)

    images_dir = session_dir / "images"
    db_path    = session_dir / "database.db"
    sparse_dir = session_dir / "sparse"
    dense_dir  = session_dir / "dense"
    output_dir = session_dir / "output"

    # --- STAGE 1: Structure from Motion (COLMAP) ---
    _set(jobs, job_id, status="colmap")
    try:
        _set(jobs, job_id, status="colmap", progress=10)

        run_step(["colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", "OPENCV",
            "--SiftExtraction.estimate_affine_shape", "1",
            "--SiftExtraction.max_num_features", "16384"
        ], session_dir)
        _set(jobs, job_id, progress=20)

        run_step(["colmap", "sequential_matcher",
            "--database_path", str(db_path)
        ], session_dir)
        _set(jobs, job_id, progress=25)

        os.makedirs(sparse_dir / "0", exist_ok=True)
        run_step(["colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.init_min_num_inliers", "10"
        ], session_dir)
        _set(jobs, job_id, progress=35)

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
        _set(jobs, job_id, progress=45)

        run_step(["colmap", "model_converter",
            "--input_path", str(dense_dir / "sparse"),
            "--output_path", str(dense_dir / "sparse"),
            "--output_type", "TXT"
        ], session_dir)

        copy_sparse_txt(dense_dir)

        _set(jobs, job_id, status="fastgs", progress=50)

    except Exception as e:
        print(f"\n[ERROR] COLMAP reconstruction failed for job {job_id}: {e}")
        _set(jobs, job_id, status="failed_colmap")
        raise RuntimeError(f"COLMAP stage failed: {e}") from e


    # --- STAGE 2: 3D Gaussian Splatting (FastGS) ---
    _set(jobs, job_id, status="fastgs")
    try:
        fastgs_dir = Path(os.environ.get("FASTGS_DIR", "/home/cave/3dapp/FastGS"))
        fastgs_python = os.environ.get("FASTGS_PYTHON", "/home/cave/miniconda3/envs/fastgs/bin/python")
        os.makedirs(output_dir, exist_ok=True)

        run_step([
            fastgs_python, "train.py",
            "-s", str(dense_dir),
            "-m", str(output_dir),
            "--iterations", "10000",
            "--eval"
        ], fastgs_dir)
        
        try:
            run_step([
                fastgs_python, "render.py",
                "-m", str(output_dir),
                "--iteration", "10000",
                "--skip_train"
            ], fastgs_dir)
        except Exception as e:
            print(f"[WARNING] render.py failed for job {job_id}: {e} — PSNR/SSIM will be unavailable.")

        _set(jobs, job_id, status="post_processing", progress=80)

        raw_ply     = output_dir / "point_cloud" / "iteration_10000" / "point_cloud.ply"
        cleaned_ply = session_dir / "point_cloud.ply"

        clean_splat(raw_ply, cleaned_ply)
        _set(jobs, job_id, progress=90)

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

        _set(
            jobs, job_id,
            status="done",
            progress=100,
            modelUrl=f"/api/download/{job_id}/point_cloud.ply",
        )
        return cleaned_ply

    except Exception as e:
        print(f"\n[ERROR] FastGS training / post-processing failed for job {job_id}: {e}")
        _set(jobs, job_id, status="failed_fastgs")
        raise RuntimeError(f"FastGS training stage failed: {e}") from e
