"""
Vector store module — Qdrant integration.

Responsibilities:
  - Init/ensure the `posts` collection exists
  - Embed captions via OpenAI text-embedding-3-small
  - Store each post (embedding + metadata) as a Qdrant point
  - Detect near-duplicate captions before posting (cosine ≥ 0.92)
  - Track style rotation: return the least-recently-used style index
  - Expose recent history for the API /status endpoint
"""

import logging
import os
import traceback
import uuid
from datetime import datetime
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

COLLECTION_NAME = "posts"
VECTOR_SIZE = 1536            # text-embedding-3-small output dimension
DUPLICATE_THRESHOLD = 0.92    # cosine similarity ≥ this → duplicate
ROTATION_WINDOW = 50          # look at last N posts for style rotation


# ── Custom exceptions ──────────────────────────────────────────────────────────

class VectorStoreError(Exception):
    """Base class for vector store errors."""

class VectorStoreConnectionError(VectorStoreError):
    """Cannot reach Qdrant."""

class VectorStoreCollectionError(VectorStoreError):
    """Collection creation or access failed."""

class DuplicateCaptionError(VectorStoreError):
    """Near-duplicate caption detected."""

    def __init__(self, score: float, existing_caption: str):
        self.score = score
        self.existing_caption = existing_caption
        super().__init__(
            f"Near-duplicate caption detected (similarity={score:.3f}). "
            f"Existing: {existing_caption[:80]!r}"
        )


# ── Client factory ─────────────────────────────────────────────────────────────

def _get_client() -> QdrantClient:
    """Create a Qdrant client from environment variables."""
    host = os.environ.get("QDRANT_HOST", "localhost").strip()
    port_str = os.environ.get("QDRANT_PORT", "6333").strip()

    if not host:
        raise VectorStoreConnectionError("QDRANT_HOST is empty.")

    try:
        port = int(port_str)
    except ValueError:
        raise VectorStoreConnectionError(
            f"QDRANT_PORT must be an integer, got: {port_str!r}"
        )

    try:
        client = QdrantClient(host=host, port=port, timeout=10)
        logger.debug(f"[vector_store] Qdrant client created: {host}:{port}")
        return client
    except Exception as exc:
        raise VectorStoreConnectionError(
            f"Failed to create Qdrant client at {host}:{port}: {exc}"
        ) from exc


# ── Collection management ──────────────────────────────────────────────────────

def init_collection() -> None:
    """
    Ensure the `posts` collection exists in Qdrant.
    Safe to call multiple times (idempotent).

    Raises:
        VectorStoreConnectionError: Qdrant is unreachable.
        VectorStoreCollectionError: Collection creation failed.
    """
    client = _get_client()

    try:
        collections = client.get_collections().collections
        existing_names = {c.name for c in collections}
    except Exception as exc:
        raise VectorStoreConnectionError(
            f"Cannot list Qdrant collections (is Qdrant running?): {exc}"
        ) from exc

    if COLLECTION_NAME in existing_names:
        logger.info(f"[vector_store] Collection '{COLLECTION_NAME}' already exists.")
        return

    logger.info(f"[vector_store] Creating collection '{COLLECTION_NAME}'...")
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qm.VectorParams(
                size=VECTOR_SIZE,
                distance=qm.Distance.COSINE,
            ),
        )

        # Create payload indexes for efficient filtering
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="style_index",
            field_schema=qm.PayloadSchemaType.INTEGER,
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="timestamp",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="is_dry_run",
            field_schema=qm.PayloadSchemaType.BOOL,
        )

        logger.info(f"[vector_store] Collection '{COLLECTION_NAME}' created with indexes.")
    except UnexpectedResponse as exc:
        raise VectorStoreCollectionError(
            f"Qdrant rejected collection creation: {exc}"
        ) from exc
    except Exception as exc:
        raise VectorStoreCollectionError(
            f"Unexpected error creating collection: {exc}"
        ) from exc


# ── Embeddings ─────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    """
    Embed text using OpenAI text-embedding-3-small.

    Args:
        text: Text to embed (caption).

    Returns:
        List of 1536 floats.

    Raises:
        EnvironmentError: OPENAI_API_KEY missing.
        VectorStoreError: Embedding API call failed.
    """
    import openai

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is required for caption embeddings. "
            "Get your key at https://platform.openai.com/api-keys"
        )
    if not api_key.startswith("sk-"):
        raise EnvironmentError(
            f"OPENAI_API_KEY looks invalid (should start with 'sk-'). "
            f"Got prefix: {api_key[:8]}..."
        )

    openai_client = openai.OpenAI(api_key=api_key)

    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[text],
        )
    except openai.AuthenticationError as exc:
        raise EnvironmentError(f"OpenAI API key rejected: {exc}") from exc
    except openai.RateLimitError as exc:
        raise VectorStoreError(f"OpenAI rate limit hit during embedding: {exc}") from exc
    except openai.APIConnectionError as exc:
        raise VectorStoreError(f"Network error reaching OpenAI embedding API: {exc}") from exc
    except openai.APIError as exc:
        raise VectorStoreError(f"OpenAI API error during embedding: {exc}") from exc
    except Exception as exc:
        raise VectorStoreError(
            f"Unexpected error during embedding: {type(exc).__name__}: {exc}"
        ) from exc

    vector = response.data[0].embedding
    if len(vector) != VECTOR_SIZE:
        raise VectorStoreError(
            f"Embedding dimension mismatch: expected {VECTOR_SIZE}, got {len(vector)}"
        )

    logger.debug(f"[vector_store] Embedded text ({len(text)} chars) → {VECTOR_SIZE}-dim vector")
    return vector


# ── Duplicate detection ────────────────────────────────────────────────────────

def check_duplicate_caption(caption_text: str) -> bool:
    """
    Check if a similar caption already exists in Qdrant.

    Args:
        caption_text: Caption to check.

    Returns:
        False if no near-duplicate found.

    Raises:
        DuplicateCaptionError: If cosine similarity ≥ DUPLICATE_THRESHOLD.
        VectorStoreConnectionError: If Qdrant is unreachable (caller should treat as non-fatal).
    """
    client = _get_client()

    try:
        count = client.count(collection_name=COLLECTION_NAME).count
    except Exception as exc:
        raise VectorStoreConnectionError(
            f"Cannot reach Qdrant for duplicate check: {exc}"
        ) from exc

    if count == 0:
        logger.debug("[vector_store] Collection empty — no duplicate check needed.")
        return False

    try:
        vector = _embed(caption_text)
    except EnvironmentError:
        raise
    except Exception as exc:
        logger.warning(f"[vector_store] Embedding failed, skipping duplicate check: {exc}")
        return False  # non-fatal: proceed without duplicate check

    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=1,
            with_payload=True,
            score_threshold=DUPLICATE_THRESHOLD,
        )
    except Exception as exc:
        raise VectorStoreConnectionError(
            f"Qdrant search failed during duplicate check: {exc}"
        ) from exc

    if results:
        top = results[0]
        existing = top.payload.get("caption_text", "")
        raise DuplicateCaptionError(score=top.score, existing_caption=existing)

    logger.debug(f"[vector_store] No duplicate found (threshold={DUPLICATE_THRESHOLD})")
    return False


# ── Storing posts ──────────────────────────────────────────────────────────────

def store_post(
    caption_text: str,
    style_index: int,
    post_id: str,
    image_path: str,
    timestamp: str,
    is_dry_run: bool = False,
) -> str:
    """
    Store a post in Qdrant with its caption embedding and metadata.

    Args:
        caption_text: Raw caption text (without hashtags).
        style_index: Index into generator.STYLES (0-7).
        post_id: Instagram media ID or dry-run fake ID.
        image_path: Local path to the image file.
        timestamp: ISO 8601 timestamp string.
        is_dry_run: True if this was a dry-run (no actual Instagram post).

    Returns:
        UUID string of the created Qdrant point.

    Raises:
        VectorStoreConnectionError: Qdrant unreachable.
        VectorStoreError: Point upsert failed.
    """
    client = _get_client()
    point_id = str(uuid.uuid4())

    try:
        vector = _embed(caption_text)
    except Exception as exc:
        logger.warning(f"[vector_store] Embedding failed, storing with zero vector: {exc}")
        vector = [0.0] * VECTOR_SIZE  # fallback: store without meaningful embedding

    payload = {
        "caption_text": caption_text,
        "style_index": style_index,
        "post_id": post_id,
        "image_path": image_path,
        "timestamp": timestamp,
        "is_dry_run": is_dry_run,
        "stored_at": datetime.utcnow().isoformat(),
    }

    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
    except UnexpectedResponse as exc:
        raise VectorStoreError(f"Qdrant rejected point upsert: {exc}") from exc
    except Exception as exc:
        raise VectorStoreConnectionError(
            f"Failed to store post in Qdrant: {exc}"
        ) from exc

    logger.info(
        f"[vector_store] Post stored: id={point_id}, style={style_index}, "
        f"post_id={post_id}, dry_run={is_dry_run}"
    )
    return point_id


# ── Style rotation ─────────────────────────────────────────────────────────────

def get_next_style_index(num_styles: int = 8) -> int:
    """
    Return the style index least recently used across stored posts.

    Algorithm:
      1. Scroll the last ROTATION_WINDOW points by timestamp descending
      2. Count how many times each style_index (0..num_styles-1) appears
      3. Return the index with the fewest occurrences
         (ties broken by whichever hasn't appeared most recently)
      4. Fallback to time-based rotation if Qdrant is unreachable or empty

    Args:
        num_styles: Total number of available styles.

    Returns:
        Integer style index in range [0, num_styles).
    """
    try:
        client = _get_client()
        count = client.count(collection_name=COLLECTION_NAME).count
    except Exception as exc:
        logger.warning(f"[vector_store] Cannot reach Qdrant for style rotation: {exc}. Using time-based fallback.")
        return _time_based_style(num_styles)

    if count == 0:
        logger.debug("[vector_store] Collection empty, using time-based style selection.")
        return _time_based_style(num_styles)

    try:
        # Scroll recent posts; Qdrant scroll doesn't support ORDER BY natively,
        # so we retrieve more points and sort client-side by timestamp.
        scroll_limit = min(ROTATION_WINDOW, count)
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=scroll_limit,
            with_payload=["style_index", "timestamp"],
            with_vectors=False,
        )
    except Exception as exc:
        logger.warning(f"[vector_store] Scroll failed for style rotation: {exc}. Using fallback.")
        return _time_based_style(num_styles)

    # Sort by timestamp descending, take the most recent window
    def _ts_key(p):
        ts = p.payload.get("timestamp", "")
        return ts if ts else ""

    points_sorted = sorted(points, key=_ts_key, reverse=True)[:ROTATION_WINDOW]

    # Count style usage in the window
    usage: dict[int, int] = {i: 0 for i in range(num_styles)}
    last_seen: dict[int, str] = {}

    for p in points_sorted:
        idx = p.payload.get("style_index")
        if idx is not None and 0 <= idx < num_styles:
            usage[idx] += 1
            ts = p.payload.get("timestamp", "")
            if idx not in last_seen or ts > last_seen[idx]:
                last_seen[idx] = ts

    # Pick the style used least often; break ties by oldest last-seen
    min_count = min(usage.values())
    candidates = [i for i, c in usage.items() if c == min_count]

    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        # Among tied candidates, pick the one with oldest last-seen timestamp
        chosen = min(candidates, key=lambda i: last_seen.get(i, ""))

    logger.info(
        f"[vector_store] Style rotation: chose index={chosen} "
        f"(used {usage[chosen]}x in last {len(points_sorted)} posts). "
        f"Usage: {usage}"
    )
    return chosen


def _time_based_style(num_styles: int) -> int:
    """Fallback: rotate style by current hour."""
    import time
    idx = int(time.time() / 3600) % num_styles
    logger.debug(f"[vector_store] Time-based style fallback: index={idx}")
    return idx


# ── History ────────────────────────────────────────────────────────────────────

def get_recent_posts(limit: int = 10, include_dry_runs: bool = False) -> list[dict]:
    """
    Return recent posts from Qdrant for the API /status endpoint.

    Args:
        limit: Maximum number of posts to return.
        include_dry_runs: If False, filter out dry-run posts.

    Returns:
        List of payload dicts sorted by timestamp descending.
    """
    try:
        client = _get_client()
    except VectorStoreConnectionError as exc:
        logger.warning(f"[vector_store] Cannot fetch history: {exc}")
        return []

    scroll_filter = None
    if not include_dry_runs:
        scroll_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="is_dry_run",
                    match=qm.MatchValue(value=False),
                )
            ]
        )

    try:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=max(limit * 3, 50),  # over-fetch, then sort client-side
            with_payload=True,
            with_vectors=False,
            scroll_filter=scroll_filter,
        )
    except Exception as exc:
        logger.warning(f"[vector_store] Scroll failed for recent posts: {exc}")
        return []

    # Sort by timestamp descending
    points_sorted = sorted(
        points,
        key=lambda p: p.payload.get("timestamp", ""),
        reverse=True,
    )[:limit]

    return [{"id": str(p.id), **p.payload} for p in points_sorted]


# ── Collection stats ───────────────────────────────────────────────────────────

def get_collection_stats() -> dict:
    """Return basic collection statistics (point count, status)."""
    try:
        client = _get_client()
        info = client.get_collection(COLLECTION_NAME)
        return {
            "status": str(info.status),
            "points_count": info.points_count,
            "collection": COLLECTION_NAME,
            "vector_size": VECTOR_SIZE,
            "distance": "cosine",
        }
    except VectorStoreConnectionError as exc:
        return {"status": "unreachable", "error": str(exc)}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
