"""FastAPI web dashboard: post history, manual trigger, scheduler toggle."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from db import get_posts, init_db
from config import OUTPUT_DIR

app = FastAPI(title="AI Influencer Dashboard")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Mount output folder so images/videos can be previewed
output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_path)), name="output")

_scheduler_running = False
_scheduler_task: asyncio.Task | None = None


@app.on_event("startup")
async def startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    posts = get_posts(limit=50)
    return templates.TemplateResponse("index.html", {"request": request, "posts": posts})


@app.get("/api/posts")
async def api_posts():
    return get_posts(limit=50)


@app.post("/api/trigger")
async def trigger(background_tasks: BackgroundTasks):
    """Manually trigger the pipeline in the background."""
    def _run():
        from main import run_once
        asyncio.run(run_once())

    background_tasks.add_task(_run)
    return {"status": "triggered"}


@app.get("/api/scheduler/status")
async def scheduler_status():
    return {"running": _scheduler_running}


@app.post("/api/scheduler/start")
async def scheduler_start():
    global _scheduler_running
    if _scheduler_running:
        return {"status": "already running"}
    _scheduler_running = True
    return {"status": "started"}


@app.post("/api/scheduler/stop")
async def scheduler_stop():
    global _scheduler_running
    _scheduler_running = False
    return {"status": "stopped"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8000, reload=True)
