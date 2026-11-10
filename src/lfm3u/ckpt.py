"""Save a (possibly fp32-upcast or LoRA-merged) LFM3 model as a self-contained bf16 checkpoint."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REMOTE_FILES = ("*.py", "generation_config.json", "chat_template.jinja")


def save(model, tok, out: str | Path, source: str, meta: dict | None = None) -> Path:
    """Write bf16 weights, tokenizer, the remote-code files of source (a local dir or HF id) and lfm3u_meta.json."""
    import torch

    out = Path(out)
    model.eval()
    model.to(torch.bfloat16)
    model.config.use_cache = True
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    src = Path(source)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(source, allow_patterns=list(REMOTE_FILES)))
    for pattern in REMOTE_FILES:
        for f in src.glob(pattern):
            if f.is_file() and not (out / f.name).exists():
                shutil.copy2(f, out / f.name)
    if meta is not None:
        (out / "lfm3u_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out
