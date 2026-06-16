import uuid
import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn

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
