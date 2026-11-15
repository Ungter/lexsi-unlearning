"""Compare one tensor of an uploaded checkpoint (S3, sharded safetensors) with the base model on HF, by ranged reads.

  python scripts/vast/cmp_tensor.py s3://bucket/models/RUN/NAME [TENSOR ...]
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "vast"), str(ROOT / "src")]

import launch  # noqa: E402
from lfm3u import storage  # noqa: E402


def s3_range(c, b, k, start, end) -> bytes:
    return c.get_object(Bucket=b, Key=k, Range=f"bytes={start}-{end - 1}")["Body"].read()


def hf_range(url, start, end, token) -> bytes:
    import requests

    r = requests.get(url, headers={"Range": f"bytes={start}-{end - 1}", "Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    return r.content


def header(read) -> tuple[dict, int]:
    n = struct.unpack("<Q", read(0, 8))[0]
    return json.loads(read(8, 8 + n)), 8 + n


def main() -> None:
    sec = launch.load_secrets()
    uri, names = sys.argv[1], sys.argv[2:] or ["model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight"]
    import numpy as np
    from huggingface_hub import hf_hub_url

    c = storage.client()
    b, k = storage.split_uri(uri.rstrip("/") + "/model.safetensors.index.json")
    idx = json.loads(c.get_object(Bucket=b, Key=k)["Body"].read())["weight_map"]
    base_url = hf_hub_url("CohereLabs/tiny-aya-global", "model.safetensors")
    try:
        bh, boff = header(lambda s, e: hf_range(base_url, s, e, sec["HF_API"]))
    except Exception:  # noqa: BLE001  base may be sharded
        bidx = json.loads(hf_range(hf_hub_url("CohereLabs/tiny-aya-global", "model.safetensors.index.json"), 0, 10 ** 7, sec["HF_API"]))
        bh = boff = None
    for name in names:
        shard = idx[name]
        sk = k.rsplit("/", 1)[0] + "/" + shard
        sh, soff = header(lambda s, e: s3_range(c, b, sk, s, e))
        m = sh[name]
        s0, s1 = m["data_offsets"]
        n = min(s1 - s0, 2 * 4096)  # first 4096 bf16 values
        ours = np.frombuffer(s3_range(c, b, sk, soff + s0, soff + s0 + n), dtype=np.uint16)
        if bh is None:
            print("base sharded; skipping", name)
            continue
        bm = bh.get(name) or bh.get(name.replace("model.", "model.", 1))
        if bm is None:
            print(name, "not in base header; base keys e.g.", list(bh)[:3])
            continue
        t0 = bm["data_offsets"][0]
        base = np.frombuffer(hf_range(base_url, boff + t0, boff + t0 + n, sec["HF_API"]), dtype=np.uint16)
        diff = int((ours != base).sum())
        print(f"{name}: dtype ours {m['dtype']} base {bm['dtype']}, {diff}/{len(ours)} of the first values differ")


if __name__ == "__main__":
    main()
