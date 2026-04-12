"""
FastAPI web dashboard for content review and approval.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database.db import get_db
from database.models import ContentItem
from sqlalchemy.orm import Session

app = FastAPI(title="AI Influencer Dashboard")
templates = Jinja2Templates(directory="dashboard/templates")
templates.env.filters['from_json'] = json.loads
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

# Serve output files for media preview
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    pending = db.query(ContentItem).filter(ContentItem.status == "pending").order_by(ContentItem.created_at.desc()).all()
    approved = db.query(ContentItem).filter(ContentItem.status == "approved").order_by(ContentItem.approved_at.desc()).limit(10).all()
    return templates.TemplateResponse("queue.html", {
        "request": request,
        "pending": pending,
        "approved": approved,
    })


@app.get("/accounts", response_class=HTMLResponse)
async def accounts(request: Request, db: Session = Depends(get_db)):
    personas = ["mia", "zara", "luna"]
    stats = {}
    for p in personas:
        stats[p] = {
            "pending": db.query(ContentItem).filter(ContentItem.persona_id == p, ContentItem.status == "pending").count(),
            "approved": db.query(ContentItem).filter(ContentItem.persona_id == p, ContentItem.status == "approved").count(),
            "rejected": db.query(ContentItem).filter(ContentItem.persona_id == p, ContentItem.status == "rejected").count(),
        }
    return templates.TemplateResponse("accounts.html", {"request": request, "stats": stats})


@app.get("/analytics", response_class=HTMLResponse)
async def analytics(request: Request, db: Session = Depends(get_db)):
    items = db.query(ContentItem).order_by(ContentItem.created_at.desc()).limit(50).all()
    return templates.TemplateResponse("analytics.html", {"request": request, "items": items})


@app.post("/approve/{item_id}")
async def approve(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ContentItem).get(item_id)
    if not item:
        raise HTTPException(404)
    item.status = "approved"
    item.approved_at = datetime.utcnow()
    if item.video_path and Path(item.video_path).exists():
        dst = Path("output/approved") / item.persona_id / Path(item.video_path).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(item.video_path, dst)
    db.commit()
    return {"status": "approved"}


@app.post("/reject/{item_id}")
async def reject(item_id: int, db: Session = Depends(get_db)):
    item = db.query(ContentItem).get(item_id)
    if not item:
        raise HTTPException(404)
    item.status = "rejected"
    db.commit()
    return {"status": "rejected"}


@app.patch("/caption/{item_id}")
async def update_caption(item_id: int, request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    item = db.query(ContentItem).get(item_id)
    if not item:
        raise HTTPException(404)
    item.caption = body.get("caption", item.caption)
    db.commit()
    return {"status": "updated"}


@app.get("/api/stats")
async def stats(db: Session = Depends(get_db)):
    personas = ["mia", "zara", "luna"]
    result = {}
    for p in personas:
        result[p] = {
            "pending": db.query(ContentItem).filter(ContentItem.persona_id == p, ContentItem.status == "pending").count(),
            "approved": db.query(ContentItem).filter(ContentItem.persona_id == p, ContentItem.status == "approved").count(),
            "rejected": db.query(ContentItem).filter(ContentItem.persona_id == p, ContentItem.status == "rejected").count(),
        }
    return result
