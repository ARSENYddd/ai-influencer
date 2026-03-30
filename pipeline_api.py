"""
FastAPI server — HTTP interface between N8N and the pipeline.

Endpoints:
  GET  /health              — liveness probe (no auth required)
  POST /run-pipeline        — trigger the full pipeline
  GET  /status              — last 10 posts from Qdrant
  GET  /style-rotation      — current style index + usage stats
  GET  /collection-stats    — Qdrant collection info
"""

import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

load_dotenv()

# Configure logging
log_dir = Path("output/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / f"api_{datetime.now().strftime('%Y%m%d')}.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── App instance ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Influencer Pipeline API",
    description="HTTP interface between N8N and the AI influencer pipeline.",
    version="1.0.0",
)


# ── Startup / shutdown ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize Qdrant collection on startup."""
    logger.info("[api] FastAPI starting up...")
    try:
        from vector_store import init_collection
        init_collection()
        logger.info("[api] Qdrant collection ready.")
    except Exception as exc:
        # Non-fatal: pipeline can run without Qdrant (it degrades gracefully)
        logger.warning(
            f"[api] Qdrant initialization failed (non-fatal): {type(exc).__name__}: {exc}\n"
            "The pipeline will run without vector store features."
        )


# ── Auth ───────────────────────────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)


def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> None:
    """
    Validate the Bearer token from Authorization header.
    Token is compared against PIPELINE_API_SECRET env var.
    """
    expected = os.environ.get("PIPELINE_API_SECRET", "").strip()

    if not expected:
        # Secret not configured — warn but allow (dev mode)
        logger.warning(
            "[api] PIPELINE_API_SECRET is not set. "
            "All authenticated endpoints are accessible without a token! "
            "Set this in your .env file for production."
        )
        return

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Use: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != expected:
        logger.warning(
            f"[api] Rejected request with invalid token "
            f"(got: {credentials.credentials[:8]}...)"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API token.",
        )


# ── Request / Response models ──────────────────────────────────────────────────

class RunPipelineRequest(BaseModel):
    dry_run: bool = False
    style_index: Optional[int] = None   # override style; None = auto from Qdrant


class RunPipelineResponse(BaseModel):
    success: bool
    timestamp: str
    post_id: Optional[str] = None
    caption: Optional[str] = None
    image_path: Optional[str] = None
    style_index: Optional[int] = None
    vector_id: Optional[str] = None
    duration_seconds: Optional[float] = None
    dry_run: bool = False
    error: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health_check():
    """
    Liveness probe — no authentication required.
    N8N's health-check node calls this before each scheduled run.
    """
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ai-influencer-pipeline-api",
    }


@app.post(
    "/run-pipeline",
    response_model=RunPipelineResponse,
    tags=["pipeline"],
    dependencies=[Depends(verify_token)],
)
async def run_pipeline_endpoint(body: RunPipelineRequest = RunPipelineRequest()):
    """
    Trigger the AI influencer pipeline.

    N8N calls this at 09:00 and 18:00 daily.

    - Asks Qdrant for the best style index (least recently used)
    - Generates image via Replicate Flux Dev
    - Generates caption via Claude Haiku
    - Posts to Instagram via instagrapi
    - Stores result in Qdrant
    """
    start = time.time()
    mode = "[DRY RUN]" if body.dry_run else "[LIVE]"
    logger.info(f"[api] POST /run-pipeline {mode}")

    # Resolve style index
    style_index = body.style_index
    if style_index is None:
        try:
            from vector_store import get_next_style_index
            from generator import STYLES
            style_index = get_next_style_index(num_styles=len(STYLES))
            logger.info(f"[api] Style index from Qdrant rotation: {style_index}")
        except Exception as exc:
            logger.warning(f"[api] Could not get style from Qdrant: {exc}. Using time-based fallback.")
            import time as _time
            from generator import STYLES
            style_index = int(_time.time() / 3600) % len(STYLES)

    # Run pipeline
    try:
        from pipeline import run_pipeline
        result = run_pipeline(dry_run=body.dry_run, style_index=style_index)
    except EnvironmentError as exc:
        logger.error(f"[api] Configuration error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Configuration error: {exc}",
        )
    except Exception as exc:
        duration = round(time.time() - start, 1)
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error(f"[api] Pipeline failed after {duration}s: {error_msg}")
        logger.debug(traceback.format_exc())
        # Return 500 with structured error body
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error": error_msg,
                "duration_seconds": duration,
                "timestamp": datetime.utcnow().isoformat(),
                "dry_run": body.dry_run,
            },
        )

    duration = round(time.time() - start, 1)
    result.setdefault("style_index", style_index)
    result.setdefault("duration_seconds", duration)

    logger.info(
        f"[api] Pipeline complete in {duration}s. "
        f"post_id={result.get('post_id')}, vector_id={result.get('vector_id')}"
    )

    return RunPipelineResponse(
        success=result.get("success", True),
        timestamp=result.get("timestamp", datetime.utcnow().isoformat()),
        post_id=result.get("post_id"),
        caption=result.get("caption"),
        image_path=result.get("image_path"),
        style_index=result.get("style_index"),
        vector_id=result.get("vector_id"),
        duration_seconds=result.get("duration_seconds"),
        dry_run=body.dry_run,
        error=result.get("error"),
    )


@app.get("/status", tags=["monitoring"], dependencies=[Depends(verify_token)])
async def get_status(limit: int = 10, include_dry_runs: bool = False):
    """
    Return recent posts from the Qdrant vector store.

    Used by N8N to display a dashboard-style summary of recent activity.
    """
    try:
        from vector_store import get_recent_posts, get_collection_stats
        posts = get_recent_posts(limit=limit, include_dry_runs=include_dry_runs)
        stats = get_collection_stats()
    except Exception as exc:
        logger.error(f"[api] /status failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector store unavailable: {exc}",
        )

    return {
        "collection": stats,
        "recent_posts": posts,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/style-rotation", tags=["monitoring"], dependencies=[Depends(verify_token)])
async def get_style_rotation():
    """
    Show current style rotation status.

    Returns the next recommended style index and recent usage counts.
    Useful for understanding what content variety the system has.
    """
    try:
        from vector_store import get_recent_posts
        from generator import STYLES

        posts = get_recent_posts(limit=50, include_dry_runs=False)
        usage: dict[int, int] = {i: 0 for i in range(len(STYLES))}
        for p in posts:
            idx = p.get("style_index")
            if idx is not None and 0 <= idx < len(STYLES):
                usage[idx] += 1

        from vector_store import get_next_style_index
        next_idx = get_next_style_index(num_styles=len(STYLES))

        return {
            "next_style_index": next_idx,
            "next_style_name": STYLES[next_idx],
            "usage_in_last_50_posts": usage,
            "all_styles": {i: s for i, s in enumerate(STYLES)},
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:
        logger.error(f"[api] /style-rotation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not compute style rotation: {exc}",
        )


@app.get("/collection-stats", tags=["monitoring"], dependencies=[Depends(verify_token)])
async def collection_stats():
    """Return raw Qdrant collection statistics."""
    try:
        from vector_store import get_collection_stats
        return get_collection_stats()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant unavailable: {exc}",
        )


# ── Dev entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("API_PORT", "8000"))
    reload_mode = os.environ.get("API_RELOAD", "false").lower() == "true"

    logger.info(f"[api] Starting on port {port} (reload={reload_mode})")
    uvicorn.run(
        "pipeline_api:app",
        host="0.0.0.0",
        port=port,
        reload=reload_mode,
        log_level="info",
    )
