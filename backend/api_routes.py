from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
import uuid, shutil, os, zipfile
from pathlib import Path
from backend.tasks import run_pipeline, run_pipeline_from_images

router = APIRouter()
WORKSPACE = Path("/home/cave/3dapp/workspace")
BACKEND_URL = "https://upset-eyes-roll.loca.lt"
jobs = {}

@router.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    session_dir = WORKSPACE / job_id
    images_dir = session_dir / "images"
    os.makedirs(images_dir, exist_ok=True)

    # save uploaded file
    upload_path = session_dir / file.filename
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # if zip of frames, extract to images_dir
    if file.filename.endswith(".zip"):
        with zipfile.ZipFile(upload_path, 'r') as z:
            z.extractall(images_dir)
        os.remove(upload_path)
        jobs[job_id] = "processing"
        background_tasks.add_task(run_pipeline_from_images, job_id, session_dir, jobs)
    else:
        # treat as video
        video_path = session_dir / "input.mp4"
        os.rename(upload_path, video_path)
        jobs[job_id] = "processing"
        background_tasks.add_task(run_pipeline, job_id, video_path, session_dir, jobs)

    return {"job_id": job_id}

@router.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": jobs[job_id]}

# 1. New route to download the .sog file
@router.get("/download/{job_id}/sog")
def download_sog(job_id: str):
    sog_path = WORKSPACE / job_id / "optimized_scene.sog"
    
    if not sog_path.exists():
        raise HTTPException(status_code=404, detail="Optimized file not found")
        
    return FileResponse(
        path=str(sog_path),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "inline; filename=scene.sog",
            "Cache-Control": "no-cache"
        }
    )

# 2. Update the view route to point SuperSplat to the new .sog endpoint
@router.get("/view/{job_id}")
def view_splat(job_id: str):
    # This tells SuperSplat to load the optimized .sog file instead of the raw .ply
    sog_url = f"{BACKEND_URL}/download/{job_id}/sog"
    return RedirectResponse(url=f"https://playcanvas.com/supersplat/viewer?load={sog_url}")
