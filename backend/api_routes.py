from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
import uuid
import shutil
import os
import zipfile
import json
import torch
from pathlib import Path
from backend.tasks import run_pipeline, run_pipeline_from_images
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from backend.persistence import load_all_jobs

router = APIRouter(prefix="/api")

WORKSPACE = Path("/home/cave/3dapp_workspace_data")
BACKEND_URL = "https://glazing-chaperone-bazooka.ngrok-free.dev"

# In-memory job store. Each job is a dict so we can track
# title/status/progress/modelUrl per the Android contract.
#
# IMPORTANT: every write to jobs[job_id] must go through backend.tasks._set()
# (or replicate its dict-preserving behavior). Never do
# `jobs[job_id] = "some_string"` anywhere — that overwrites this dict and
# breaks every route below that calls job.get(...).
jobs = {}

FAILURE_REASONS = {
    "failed_ffmpeg": "Video extraction failed. The uploaded file may be corrupted or in an unsupported format.",
    "failed_colmap": "Camera tracking failed (COLMAP). The video likely lacks sufficient texture, overlap, or steady movement for reconstruction.",
    "failed_fastgs": "Scene training failed (FastGS). This is typically caused by memory limits or an empty point cloud from the previous step.",
    "failed_cleanup": "Post-processing failed while attempting to clean noise and floaters from the generated scene.",
}

STAGE_PROGRESS = {
    "processing": 10,
    "colmap": 35,
    "fastgs": 65,
    "post_processing": 85,
    "done": 100,
    "failed": 0,
    "failed_ffmpeg": 0,
    "failed_colmap": 0,
    "failed_fastgs": 0,
    "failed_cleanup": 0,
}


@router.get("/ping")
async def ping():
    """Heartbeat check polled by the Android client's Config tab."""
    return {
        "status": "healthy",
        "version": "1.1",
        "gpu_available": torch.cuda.is_available(),
    }


@router.get("/health")
def health():
    ply_files = []
    for root, dirs, files in os.walk(WORKSPACE):
        for f in files:
            if f.endswith(".ply"):
                ply_files.append(os.path.join(root, f))
    latest = sorted(ply_files)[-1] if ply_files else None
    return {"status": "ok", "ply_ready": latest is not None, "ply_path": latest}


@router.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    title: str = Form(...),
):
    """
    Video/zip ingestion. Field name must be 'video' to match the
    Android Retrofit @Part video: MultipartBody.Part declaration.
    """
    job_id = str(uuid.uuid4())
    session_dir = WORKSPACE / job_id
    images_dir = session_dir / "images"
    os.makedirs(images_dir, exist_ok=True)

    # Save uploaded file
    upload_path = session_dir / video.filename
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Initialize job record in the shape the Android client expects
    jobs[job_id] = {
        "id": job_id,
        "title": title,
        "status": "colmap",
        "progress": 0,
        "modelUrl": None,
    }

    # If zip of frames, extract to images_dir; otherwise treat as video
    if video.filename.endswith(".zip"):
        with zipfile.ZipFile(upload_path, "r") as z:
            z.extractall(images_dir)
        os.remove(upload_path)
        background_tasks.add_task(run_pipeline_from_images, job_id, session_dir, jobs)
    else:
        video_path = session_dir / "input.mp4"
        os.rename(upload_path, video_path)
        background_tasks.add_task(run_pipeline, job_id, video_path, session_dir, jobs)

    return {
        "success": True,
        "jobId": job_id,
        "message": "Upload complete. Pipeline scheduled.",
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    status = job.get("status", "")

    # Mask granular failures as just "failed" for the primary status field
    # so standard UI logic doesn't break, but include the granular details too.
    display_status = "failed" if status.startswith("failed") else status
    job["progress"] = STAGE_PROGRESS.get(status, job.get("progress", 0))

    if display_status == "done" and not job.get("modelUrl"):
        job["modelUrl"] = f"/api/download/{job_id}/point_cloud.ply"

    response = dict(job)
    response["status"] = display_status

    if status.startswith("failed"):
        response["error_stage"] = status
        response["error_message"] = FAILURE_REASONS.get(
            status, "An unknown error occurred during pipeline execution."
        )

    return response


@router.get("/jobs")
async def get_all_jobs():
    response_list = []
    for job in jobs.values():
        status = job.get("status", "")
        display_status = "failed" if status.startswith("failed") else status
        job_data = dict(job)
        job_data["status"] = display_status
        job_data["progress"] = STAGE_PROGRESS.get(status, job.get("progress", 0))

        if display_status == "done" and not job_data.get("modelUrl"):
            job_data["modelUrl"] = f"/api/download/{job['id']}/point_cloud.ply"

        if status.startswith("failed"):
            job_data["error_stage"] = status
            job_data["error_message"] = FAILURE_REASONS.get(
                status, "An unknown error occurred during pipeline execution."
            )

        response_list.append(job_data)

    return response_list

@router.get("/audit/{job_id}")
def get_audit_log(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = WORKSPACE / job_id / "removed_points.json"
    if not log_path.exists():
        raise HTTPException(
            status_code=404, detail="Audit log not found — job may still be processing."
        )
    with open(log_path, "r") as f:
        return json.load(f)
        
@router.get("/metrics/{job_id}")
def get_metrics(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    metrics = jobs[job_id].get("metrics")
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="Metrics not available — job may still be processing or PSNR/SSIM evaluation was skipped."
        )
    return metrics
    
def _format_field_name(key: str) -> str:
    return key.replace("_", " ").title()
    	
    	
def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _flatten_metrics(metrics: dict) -> list[tuple[str, str]]:
    rows = []
    for k, v in metrics.items():
        if isinstance(v, dict):
            # flatten nested dicts (e.g. colmap) into their own labeled sub-rows
            for sub_k, sub_v in v.items():
                label = f"{_format_field_name(k)} — {_format_field_name(sub_k)}"
                rows.append((label, _format_value(sub_v)))
        else:
            rows.append((_format_field_name(k), _format_value(v)))
    return rows

def _build_pdf(title: str, job_id: str, rows: list[tuple[str, str]]) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    cell_style = styles["Normal"]
    cell_style.fontSize = 9

    elements = [
        Paragraph(f"SplatStudio — {title}", styles["Title"]),
        Paragraph(f"Job ID: {job_id}", styles["Normal"]),
        Spacer(1, 16),
    ]

    header = [Paragraph("<b>Field</b>", cell_style), Paragraph("<b>Value</b>", cell_style)]
    table_data = [header] + [
        [Paragraph(str(k), cell_style), Paragraph(str(v), cell_style)] for k, v in rows
    ]

    table = Table(table_data, colWidths=[160, 320])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#161B22")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#30363D")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F0F0")]),
    ]))
    elements.append(table)
    doc.build(elements)
    return buf.getvalue()


@router.get("/metrics/{job_id}/report")
def download_metrics_report(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    metrics = jobs[job_id].get("metrics")
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="Metrics not available — job may still be processing or PSNR/SSIM evaluation was skipped."
        )

    rows = _flatten_metrics(metrics)
    pdf_bytes = _build_pdf("Quality Metrics", job_id, rows)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_quality_metrics.pdf"'},
    )


@router.get("/audit/{job_id}/report")
def download_audit_report(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    log_path = WORKSPACE / job_id / "removed_points.json"
    if not log_path.exists():
        raise HTTPException(
            status_code=404, detail="Audit log not found — job may still be processing."
        )
    with open(log_path, "r") as f:
        audit_entries = json.load(f)

    rows = []
    # audit_entries is the same list/dict structure your Audit Log modal renders —
    # adjust key names below only if they differ from what's shown on screen.
    entries = audit_entries if isinstance(audit_entries, list) else audit_entries.get("entries", [])
    for entry in entries:
        stage = entry.get("stage", "")
        timestamp = entry.get("timestamp", "")
        rows.append((f"{stage}", timestamp))
        rows.append(("  Reason", str(entry.get("reason", ""))))
        rows.append(("  Points Removed", str(entry.get("points_removed", ""))))
        rows.append(("  Threshold", str(entry.get("threshold", ""))))

    pdf_bytes = _build_pdf("Audit Log", job_id, rows)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_audit_log.pdf"'},
    )


# Kept for backwards compatibility with anything still calling the old route.
# Not used by the new Android client, safe to remove once fully migrated.
@router.get("/status/{job_id}")
async def get_status_legacy(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": jobs[job_id].get("status")}


# Download route serving the raw .ply file directly to the Three.js frontend
@router.get("/download/{job_id}/point_cloud.ply")
def download_ply(job_id: str):
    ply_path = WORKSPACE / job_id / "point_cloud.ply"

    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="Cleaned point cloud file (.ply) not found")

    return FileResponse(
        path=str(ply_path),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "inline"},
    )


@router.get("/download/{job_id}/point_cloud.ply/export")
def export_ply(job_id: str):
    """Same file as download_ply, but forces a browser/app download
    instead of inline rendering — for users who want to manually
    edit/clean the splat in an external tool."""
    ply_path = WORKSPACE / job_id / "point_cloud.ply"

    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="Cleaned point cloud file (.ply) not found")

    return FileResponse(
        path=str(ply_path),
        media_type="application/octet-stream",
        filename=f"{job_id}.ply",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.ply"'},
    )

# --- Job persistence ---

def _job_meta_path(job_id: str) -> Path:
    return WORKSPACE / job_id / "job_meta.json"


def save_job(job_id: str):
    """Call this any time jobs[job_id] is updated, so it survives restarts."""
    if job_id not in jobs:
        return
    meta_path = _job_meta_path(job_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(jobs[job_id], f)


def load_all_jobs():
    """Call once on server startup to repopulate the in-memory jobs dict."""
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
                continue  # skip corrupted/partial job folders
                

# View route returning the lightweight self-contained Three.js 3D renderer.
# This is what gets exposed as `modelUrl` once a job is done.
@router.get("/view/{job_id}", response_class=HTMLResponse)
def view_splat(job_id: str):
    html_content = f"""
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
    <title>3D Splat Viewer</title>
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background-color: #111;
            font-family: monospace;
        }}
        #canvas-container {{
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 14px;
            max-width: 90%;
            word-break: break-all;
            text-align: center;
            z-index: 10;
        }}
    </style>

    <script async src="https://unpkg.com/es-module-shims@1.8.0/dist/es-module-shims.js"></script>

    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@0.157.0/build/three.module.js",
            "@mkkellogg/gaussian-splats-3d": "https://unpkg.com/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js"
        }}
    }}
    </script>
</head>
<body>
    <div id="loading">Initializing 3D Splat Engine...</div>
    <div id="canvas-container"></div>

    <script type="module">
        const jobId = "{job_id}";
        const plyUrl = `/api/download/${{jobId}}/point_cloud.ply?ngrok-skip-browser-warning=true`;
        const loadingEl = document.getElementById('loading');

        function showError(label, err) {{
            loadingEl.innerText = label + ": " + (err && err.message ? err.message : String(err));
            loadingEl.style.color = "#ff6b6b";
        }}

        import('@mkkellogg/gaussian-splats-3d')
            .then(GaussianSplats3D => {{
                try {{
                    const viewer = new GaussianSplats3D.Viewer({{
                        'container': document.getElementById('canvas-container'),
                        'initialCameraPosition': [0, 0, 5],
                        'initialCameraLookAt': [0, 0, 0],
                        'ignoreDevicePixelRatio': false,
                        'sharedMemoryForWorkers': false,
                        'selfClosed': true
                    }});

                    loadingEl.innerText = "Downloading and processing 3D data...";

                    viewer.addSplatScene(plyUrl, {{
                        'splatAlphaRemovalThreshold': 5,
                        'showLoadingUI': false
                    }})
                    .then(() => {{
                        loadingEl.style.display = 'none';
                        viewer.start();
                    }})
                    .catch(err => showError("Viewer load error", err));

                    window.addEventListener('resize', () => {{
                        viewer.resize();
                    }});

                }} catch (err) {{
                    showError("Viewer init error", err);
                }}
            }})
            .catch(err => showError("Module import error", err));
    </script>
</body>
</html>
    """
    return html_content
