"""AWS S3 upload: stores generated media and returns public HTTPS URLs."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

import boto3
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    PartialCredentialsError,
)
from loguru import logger

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    S3_BUCKET,
    S3_PUBLIC_BASE_URL,
)


# ── Custom exceptions ──────────────────────────────────────────────────────────

class S3Error(Exception):
    """Base S3 error."""

class S3AuthError(S3Error):
    """AWS credentials missing or invalid."""

class S3BucketError(S3Error):
    """Bucket doesn't exist or access denied."""

class S3UploadError(S3Error):
    """Upload failed (network, quota, permissions)."""

class S3FileError(S3Error):
    """Local file not found or unreadable."""


# ── Client ─────────────────────────────────────────────────────────────────────

def _get_client():
    """Build boto3 S3 client. Raises S3AuthError if credentials are missing."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise S3AuthError(
            "AWS credentials not set.\n"
            "  Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env\n"
            "  Get them at: https://console.aws.amazon.com/iam → Users → Security credentials"
        )
    if not S3_BUCKET:
        raise S3BucketError(
            "S3_BUCKET is not set in .env.\n"
            "  Create a bucket at https://s3.console.aws.amazon.com/s3"
        )
    try:
        return boto3.client(
            "s3",
            region_name=AWS_REGION or "us-east-1",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    except Exception as exc:
        raise S3AuthError(f"Failed to create S3 client: {exc}") from exc


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_file(local_path: str, s3_key: str | None = None) -> str:
    """
    Upload a local file to S3 and return its public HTTPS URL.

    Args:
        local_path: Absolute or relative path to the local file.
        s3_key: S3 object key (path inside bucket). Auto-derived from filename if None.

    Returns:
        Public HTTPS URL to the uploaded file.

    Raises:
        S3FileError: File doesn't exist or is empty.
        S3AuthError: AWS credentials are missing or invalid.
        S3BucketError: Bucket not found or access denied.
        S3UploadError: Upload failed for any other reason.
    """
    path = Path(local_path)

    # ── Validate local file ────────────────────────────────────────────────────
    if not path.exists():
        raise S3FileError(
            f"File not found: {local_path}\n"
            "  Make sure the generator saved the file before uploading."
        )
    if not path.is_file():
        raise S3FileError(f"Path is not a file: {local_path}")
    if path.stat().st_size == 0:
        raise S3FileError(
            f"File is empty (0 bytes): {local_path}\n"
            "  The generator may have failed silently."
        )

    size_mb = path.stat().st_size / 1024 / 1024
    logger.debug(f"[s3] Uploading {path.name} ({size_mb:.1f} MB) → s3://{S3_BUCKET}/{s3_key or path.name}")

    # ── Derive S3 key ──────────────────────────────────────────────────────────
    if s3_key is None:
        # e.g. output/images/sofia_20240101_120000.jpg → images/sofia_20240101_120000.jpg
        parts = path.parts
        if "output" in parts:
            idx = parts.index("output")
            s3_key = "/".join(parts[idx + 1:])
        else:
            s3_key = path.name

    content_type = _guess_content_type(path)
    client = _get_client()

    try:
        client.upload_file(
            str(path),
            S3_BUCKET,
            s3_key,
            ExtraArgs={
                "ContentType": content_type,
                # ACL: use "public-read" only if bucket has ACLs enabled.
                # Modern S3 best practice: use bucket policy for public access instead.
                # Uncomment if your bucket uses ACLs:
                # "ACL": "public-read",
            },
        )
    except NoCredentialsError:
        raise S3AuthError(
            "AWS credentials not found.\n"
            "  Check AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env"
        )
    except PartialCredentialsError as exc:
        raise S3AuthError(f"Incomplete AWS credentials: {exc}") from exc
    except EndpointConnectionError as exc:
        raise S3UploadError(
            f"Cannot connect to S3 endpoint (region: {AWS_REGION}).\n"
            f"  Check your internet connection and AWS_REGION in .env\n"
            f"  Error: {exc}"
        ) from exc
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        message = exc.response["Error"]["Message"]

        if code in ("NoSuchBucket", "404"):
            raise S3BucketError(
                f"Bucket '{S3_BUCKET}' not found.\n"
                f"  Create it at https://s3.console.aws.amazon.com/s3\n"
                f"  Make sure S3_BUCKET in .env matches exactly (case-sensitive)"
            ) from exc
        if code in ("AccessDenied", "403"):
            raise S3BucketError(
                f"Access denied to bucket '{S3_BUCKET}'.\n"
                f"  Your IAM user needs: s3:PutObject, s3:GetObject on this bucket.\n"
                f"  Check IAM policy at https://console.aws.amazon.com/iam"
            ) from exc
        if code == "InvalidAccessKeyId":
            raise S3AuthError(
                "Invalid AWS_ACCESS_KEY_ID.\n"
                "  Regenerate at: https://console.aws.amazon.com/iam → Users → Security credentials"
            ) from exc
        if code == "SignatureDoesNotMatch":
            raise S3AuthError(
                "AWS_SECRET_ACCESS_KEY is incorrect.\n"
                "  Regenerate at: https://console.aws.amazon.com/iam → Users → Security credentials"
            ) from exc
        if code == "EntityTooLarge":
            raise S3UploadError(
                f"File too large for single upload: {size_mb:.1f} MB.\n"
                "  S3 multipart upload is needed for files > 5 GB."
            ) from exc

        raise S3UploadError(
            f"S3 upload failed [{code}]: {message}\n"
            f"  File: {local_path}"
        ) from exc
    except OSError as exc:
        raise S3FileError(
            f"Cannot read file for upload: {local_path}\n"
            f"  OS error: {exc}"
        ) from exc

    public_url = _build_url(s3_key)
    logger.info(f"[s3] Uploaded: {public_url}")
    return public_url


def _build_url(s3_key: str) -> str:
    """Build public URL from S3 key."""
    if S3_PUBLIC_BASE_URL:
        base = S3_PUBLIC_BASE_URL.rstrip("/")
        return f"{base}/{s3_key}"
    region = AWS_REGION or "us-east-1"
    if region == "us-east-1":
        return f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
    return f"https://{S3_BUCKET}.s3.{region}.amazonaws.com/{s3_key}"


def _guess_content_type(path: Path) -> str:
    ct, _ = mimetypes.guess_type(str(path))
    return ct or "application/octet-stream"


# ── Verify connectivity ────────────────────────────────────────────────────────

def check_s3_access() -> None:
    """
    Verify S3 credentials and bucket access.
    Raises descriptive exception if anything is wrong.
    """
    client = _get_client()
    try:
        client.head_bucket(Bucket=S3_BUCKET)
        logger.info(f"[s3] Bucket '{S3_BUCKET}' is accessible.")
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            raise S3BucketError(
                f"Bucket '{S3_BUCKET}' does not exist in region '{AWS_REGION}'.\n"
                "  Create it at https://s3.console.aws.amazon.com/s3"
            ) from exc
        if code in ("403", "AccessDenied"):
            raise S3BucketError(
                f"Access denied to bucket '{S3_BUCKET}'.\n"
                "  Minimum required IAM permissions:\n"
                "    s3:PutObject\n"
                "    s3:GetObject\n"
                "    s3:HeadBucket\n"
                "  Attach these at: https://console.aws.amazon.com/iam"
            ) from exc
        raise S3UploadError(f"S3 connectivity check failed [{code}]: {exc}") from exc
    except EndpointConnectionError as exc:
        raise S3UploadError(
            f"Cannot reach S3 (region: {AWS_REGION}). Check internet and AWS_REGION.\n"
            f"  Error: {exc}"
        ) from exc
