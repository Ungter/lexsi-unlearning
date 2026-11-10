"""Tigris (S3-compatible) helpers for artifacts in the lexsi-hackathon bucket.

Credentials come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_ENDPOINT_URL_S3 (default https://t3.storage.dev).
CLI:  python -m lfm3u.storage put FILE s3://bucket/key | get s3://bucket/key FILE | upload DIR s3://bucket/prefix
      | download s3://bucket/prefix DIR | cat s3://bucket/key
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_ENDPOINT = "https://t3.storage.dev"


def client():
    import boto3
    from botocore.config import Config

    return boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3") or DEFAULT_ENDPOINT, region_name="auto",
                        config=Config(s3={"addressing_style": "virtual"}, retries={"max_attempts": 5, "mode": "standard"}))


def split_uri(uri: str) -> tuple[str, str]:
    for p in ("s3://", "t3://", "tigris://"):
        if uri.startswith(p):
            bucket, _, key = uri[len(p):].partition("/")
            return bucket, key
    raise ValueError(f"not an s3 uri: {uri}")


def put(local: str | Path, uri: str) -> None:
    b, k = split_uri(uri)
    client().upload_file(str(local), b, k)


def put_bytes(data: bytes, uri: str) -> None:
    b, k = split_uri(uri)
    client().put_object(Bucket=b, Key=k, Body=data)


def upload_dir(local: str | Path, uri: str) -> int:
    b, prefix = split_uri(uri)
    c, n = client(), 0
    for f in sorted(Path(local).rglob("*")):
        if f.is_file():
            c.upload_file(str(f), b, f"{prefix.rstrip('/')}/{f.relative_to(local).as_posix()}")
            n += 1
    return n


def download_prefix(uri: str, local: str | Path) -> int:
    b, prefix = split_uri(uri)
    prefix = prefix.rstrip("/") + "/"
    c, n = client(), 0
    for page in c.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=prefix):
        for obj in page.get("Contents", []):
            dest = Path(local) / obj["Key"][len(prefix):]
            dest.parent.mkdir(parents=True, exist_ok=True)
            c.download_file(b, obj["Key"], str(dest))
            n += 1
    return n


def get_text(uri: str) -> str | None:
    b, k = split_uri(uri)
    try:
        return client().get_object(Bucket=b, Key=k)["Body"].read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        if "NoSuchKey" in type(e).__name__ or "NoSuchKey" in str(e) or "404" in str(e):
            return None
        raise


def presign(uri: str, expires: int = 6 * 3600) -> str:
    b, k = split_uri(uri)
    return client().generate_presigned_url("get_object", Params={"Bucket": b, "Key": k}, ExpiresIn=expires)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "put":
        put(a[0], a[1])
    elif cmd == "get":
        b, k = split_uri(a[0])
        Path(a[1]).parent.mkdir(parents=True, exist_ok=True)
        client().download_file(b, k, a[1])
    elif cmd == "upload":
        print(f"uploaded {upload_dir(a[0], a[1])} files")
    elif cmd == "download":
        print(f"downloaded {download_prefix(a[0], a[1])} files")
    elif cmd == "cat":
        print(get_text(a[0]) or "")
    else:
        sys.exit(f"unknown command {cmd}")
