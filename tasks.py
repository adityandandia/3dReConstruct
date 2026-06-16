import subprocess
import os
from pathlib import Path

def run_step(command, cwd):
    print(f"\n[RUNNING]: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, stdout=None, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {command[0]}")

def run_pipeline(job_id: str, video_path: Path, session_dir: Path):
    images_dir = session_dir / "images"
    db_path = session_dir / "database.db"
    sparse_dir = session_dir / "sparse"
    dense_dir = session_dir / "dense"
    
    # 1. FFmpeg
    run_step(["ffmpeg", "-i", str(video_path), "-qscale:v", "1", "-vf", "fps=2", str(images_dir / "%04d.jpg")], session_dir)
    
    # 2. COLMAP
    run_step(["colmap", "feature_extractor", "--database_path", str(db_path), "--image_path", str(images_dir)], session_dir)
    run_step(["colmap", "exhaustive_matcher", "--database_path", str(db_path)], session_dir)
    
    os.makedirs(sparse_dir / "0", exist_ok=True)
    run_step([
        "colmap", "mapper", "--database_path", str(db_path), "--image_path", str(images_dir), 
        "--output_path", str(sparse_dir), "--Mapper.init_min_num_inliers", "10"
    ], session_dir)

    # 3. FastGS / Pruning logic can be added here
    print(f"Pipeline finished for {job_id}")
