# backend/persistence.py
import json
from pathlib import Path

WORKSPACE = Path("/home/cave/3dapp_workspace_data")


def _job_meta_path(job_id: str) -> Path:
    return WORKSPACE / job_id / "job_meta.json"


def save_job(jobs: dict, job_id: str):
    if job_id not in jobs:
        return
    meta_path = _job_meta_path(job_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(jobs[job_id], f)


def load_all_jobs(jobs: dict):
    if not WORKSPACE.exists():
        return
    for job_dir in WORKSPACE.iterdir():
        if not job_dir.is_dir():
            continue
        meta_path = job_dir / "job_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    job_data = json.load(f)
                jobs[job_data["id"]] = job_data
            except Exception:
                continue
