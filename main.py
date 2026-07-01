import uuid
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.api_routes import router
from backend.tasks import run_pipeline

app = FastAPI()

# CORS Middleware (Correctly configured)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"] 
)

BASE_WORKSPACE = Path("/home/cave/3dapp/workspace")
PLY_PATH = "/home/cave/3dapp/workspace" 

# Include routes from api_routes.py
app.include_router(router)

# --- Routes ---

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

@app.get("/health")
def health():
    ply_files = []
    for root, dirs, files in os.walk(BASE_WORKSPACE):
        for f in files:
            if f.endswith(".ply"):
                ply_files.append(os.path.join(root, f))
    latest = sorted(ply_files)[-1] if ply_files else None
    return {"status": "ok", "ply_ready": latest is not None, "ply_path": latest}

@app.get("/splat/latest.ply")
def serve_ply():
    ply_files = []
    for root, dirs, files in os.walk(BASE_WORKSPACE):
        for f in files:
            if f.endswith("cleaned_point_cloud.ply"):
                ply_files.append(os.path.join(root, f))
    if not ply_files:
        raise HTTPException(status_code=404, detail="No PLY found. Run pipeline first.")
    latest = sorted(ply_files)[-1]
    return FileResponse(path=latest, media_type="application/octet-stream", filename="latest.ply")

# --- Startup ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
