import uuid
import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
from backend.api_routes import jobs
from backend.persistence import load_all_jobs
from backend.api_routes import router
from backend.tasks import run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_WORKSPACE = Path("/home/cave/3dapp/workspace")

app.include_router(router)



@app.on_event("startup")
async def startup_event():
    load_all_jobs(jobs)

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    session_dir = BASE_WORKSPACE / job_id
    os.makedirs(session_dir / "images", exist_ok=True)

    video_path = session_dir / "input.mp4"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(run_pipeline, job_id, video_path, session_dir)

    return {"job_id": job_id, "status": "started"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi.responses import FileResponse
import os

PLY_PATH = "/home/cave/3dapp/workspace"  # we'll find the actual ply below

@app.get("/health")
def health():
    ply_files = []
    for root, dirs, files in os.walk("/home/cave/3dapp/workspace"):
        for f in files:
            if f.endswith(".ply"):
                ply_files.append(os.path.join(root, f))
    latest = sorted(ply_files)[-1] if ply_files else None
    return {"status": "ok", "ply_ready": latest is not None, "ply_path": latest}

@app.get("/splat/latest.ply")
def serve_ply():
    ply_files = []
    for root, dirs, files in os.walk("/home/cave/3dapp/workspace"):
        for f in files:
            if f.endswith("cleaned_point_cloud.ply"):
                ply_files.append(os.path.join(root, f))
    if not ply_files:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="No PLY found. Run pipeline first.")
    latest = sorted(ply_files)[-1]
    return FileResponse(path=latest, media_type="application/octet-stream", filename="latest.ply")
