import subprocess
import os
import sys
from pathlib import Path

FASTGS_ROOT = Path("/home/cave/3dapp/FastGS")

def run_step(command, cwd):
    print(f"\n[EXECUTING]: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, env={**os.environ, "PYTHONPATH": str(FASTGS_ROOT)}, stdout=None, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {command[0]}")

def run_pipeline(job_id: str, video_path: Path, session_dir: Path, jobs: dict):
    try:
        images_dir = session_dir / "images"
        db_path = session_dir / "database.db"
        sparse_dir = session_dir / "sparse"
        dense_dir = session_dir / "dense"

        # 1. FFmpeg
        run_step(["ffmpeg", "-i", str(video_path), "-qscale:v", "1", "-vf", "fps=5", str(images_dir / "%04d.jpg")], session_dir)

        # 2. COLMAP Feature Extraction
        run_step(["colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", "SIMPLE_RADIAL",
            "--SiftExtraction.max_num_features", "8192"], session_dir)

        # 3. COLMAP Matching
        run_step(["colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SequentialMatching.overlap", "10"], session_dir)

        # 4. COLMAP Mapper (SfM)
        os.makedirs(sparse_dir / "0", exist_ok=True)
        run_step(["colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.min_num_matches", "3",
            "--Mapper.init_min_num_inliers", "5",
            "--Mapper.init_min_tri_angle", "1",
            "--Mapper.multiple_models", "0"], session_dir)

        # 5. COLMAP Image Undistorter
        os.makedirs(dense_dir, exist_ok=True)
        run_step(["colmap", "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(sparse_dir / "0"),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP"], session_dir)

        # 5b. Move dense/sparse → dense/sparse/0
        dense_sparse = dense_dir / "sparse"
        dense_sparse_0 = dense_sparse / "0"
        if dense_sparse.exists() and not dense_sparse_0.exists():
            os.makedirs(dense_sparse_0, exist_ok=True)
            for f in dense_sparse.iterdir():
                if f.is_file():
                    f.rename(dense_sparse_0 / f.name)

        # 5c. Convert binary → text format
        run_step(["colmap", "model_converter",
            "--input_path", str(dense_sparse_0),
            "--output_path", str(dense_sparse_0),
            "--output_type", "TXT"], session_dir)

        # 6. FastGS Training
        train_script = FASTGS_ROOT / "train.py"
        run_step([
            sys.executable, str(train_script),
            "-s", str(dense_dir),
            "--model_path", str(session_dir / "output"),
            "--iterations", "7000"
        ], session_dir)

        # 7. Open3D Cleanup
        ply_input = session_dir / "output" / "point_cloud" / "iteration_7000" / "point_cloud.ply"
        ply_cleaned = session_dir / "output" / "point_cloud" / "iteration_7000" / "point_cloud_cleaned.ply"
        run_step([
            sys.executable, "/home/cave/3dapp/backend/cleanup.py",
            str(ply_input),
            str(ply_cleaned)
        ], session_dir)

        jobs[job_id] = "done"
        print(f"\n[SUCCESS] Pipeline finished for job {job_id}")

    except Exception as e:
        jobs[job_id] = f"failed: {str(e)}"
        print(f"\n[FAILED] Job {job_id}: {str(e)}")
        raise
