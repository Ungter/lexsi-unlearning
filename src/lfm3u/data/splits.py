import hashlib
import json
import logging
import math
import random
import re
from pathlib import Path

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FRACTIONS = {"train": 0.6, "dev": 0.15, "test": 0.25}
LOGGER = logging.getLogger(__name__)


def frozen_root(path):
    path = Path(path).resolve()
    for parent in (path, *path.parents):
        manifest = parent / "MANIFEST.json"
        if manifest.is_file():
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if metadata.get("frozen") is True:
                return parent
    return None


def ensure_writable(path):
    root = frozen_root(path)
    if root is not None:
        raise PermissionError(f"data is frozen by {root / 'MANIFEST.json'}")
    return Path(path)


def read_jsonl(path, *, allow_test=False):
    path = Path(path)
    root = frozen_root(path)
    if root is not None and not allow_test:
        relative = path.resolve().relative_to(root).as_posix()
        if relative.startswith("curation/") or re.search(r"(^|[/_.-])test([/_.-]|$)", relative):
            raise PermissionError("frozen test data is reserved for the audit; pass allow_test=True there")
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"expected an object in {path} at line {number}")
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    path = ensure_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def write_json(path, value):
    path = ensure_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(row):
    for field in ("source_id", "parent_id", "id"):
        if row.get(field) not in (None, ""):
            return str(row[field])
    text = next((str(row[k]) for k in ("question", "instruction", "prompt", "text") if row.get(k)), None)
    if text is None:
        raise ValueError("an item needs a source ID or source text")
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:20]


def split_counts(rows, counts, seed=0):
    rows = list(rows)
    if not counts or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts.values()):
        raise ValueError("split counts must be nonnegative integers")
    if sum(counts.values()) != len(rows):
        raise ValueError(f"counts sum to {sum(counts.values())} but there are {len(rows)} rows")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    out, pos = {}, 0
    for name, n in counts.items():
        out[name] = [rows[i] for i in order[pos:pos + n]]
        pos += n
    return out


def group_split(rows, fractions=None, key=source_id, seed=0):
    rows = list(rows)
    fractions = FRACTIONS if fractions is None else fractions
    if not fractions or any(not math.isfinite(v) or v < 0 for v in fractions.values()):
        raise ValueError("split fractions must be finite and nonnegative")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to one")
    groups = sorted({key(row) for row in rows}, key=lambda value: (str(type(value)), str(value)))
    random.Random(seed).shuffle(groups)
    counts = {name: math.floor(len(groups) * fraction) for name, fraction in fractions.items()}
    remainder = len(groups) - sum(counts.values())
    ranked = sorted(fractions, key=lambda name: len(groups) * fractions[name] - counts[name], reverse=True)
    for name in ranked[:remainder]:
        counts[name] += 1
    assignments, pos = {}, 0
    for name, count in counts.items():
        assignments.update((group, name) for group in groups[pos:pos + count])
        pos += count
    out = {name: [] for name in fractions}
    for row in rows:
        out[assignments[key(row)]].append(row)
    return out


def assert_disjoint(parts, key=source_id):
    owners = {}
    for name, rows in parts.items():
        for row in rows:
            group = key(row)
            if group in owners and owners[group] != name:
                raise ValueError(f"source {group!r} appears in both {owners[group]} and {name}")
            owners[group] = name
    return True


def leakage_filter(test_rows, train_rows, text_key="question", threshold=0.9, model_name=EMBED_MODEL,
                   *, model=None, batch_size=256, block_size=1024):
    import numpy as np

    if not -1 <= threshold <= 1 or batch_size < 1 or block_size < 1:
        raise ValueError("invalid similarity threshold or batch size")
    test_rows, train_rows = list(test_rows), list(train_rows)
    stats = {"checked": len(test_rows), "train_size": len(train_rows), "threshold": threshold,
             "model": model_name, "dropped": 0, "dropped_source_ids": []}
    if not test_rows or not train_rows:
        return test_rows, stats
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

    def encode(rows):
        texts = [str(text_key(row) if callable(text_key) else row[text_key]) for row in rows]
        result = np.asarray(model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                         convert_to_numpy=True), dtype=np.float32)
        if result.ndim != 2 or len(result) != len(rows) or not np.isfinite(result).all():
            raise ValueError("the embedding model returned invalid vectors")
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("the embedding model returned zero vectors")
        return result / norms

    train, test = encode(train_rows), encode(test_rows)
    max_sim = np.full(len(test), -np.inf, dtype=np.float32)
    for start in range(0, len(test), block_size):
        end = start + block_size
        for offset in range(0, len(train), block_size):
            scores = test[start:end] @ train[offset:offset + block_size].T
            max_sim[start:end] = np.maximum(max_sim[start:end], scores.max(axis=1))
    train_ids = {source_id(row) for row in train_rows}
    dropped = {source_id(row) for row, sim in zip(test_rows, max_sim)
               if sim > threshold or source_id(row) in train_ids}
    kept = [row for row in test_rows if source_id(row) not in dropped]
    stats.update(dropped=len(test_rows) - len(kept), kept=len(kept), dropped_source_ids=sorted(dropped))
    LOGGER.info("leakage check: %d/%d items dropped", stats["dropped"], stats["checked"])
    return kept, stats


def write_manifest(data_dir, extra=None):
    data_dir = ensure_writable(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    files, manifests = [], []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        if path.suffix not in (".json", ".jsonl"):
            continue
        entry = {"path": path.relative_to(data_dir).as_posix(), "sha256": sha256(path)}
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8-sig") as handle:
                entry["rows"] = sum(bool(line.strip()) for line in handle)
        files.append(entry)
        if path.name == "manifest.json":
            manifests.append(entry["path"])
    manifest = {"version": 1, "frozen": True, "files": files,
                "total_rows": sum(f.get("rows", 0) for f in files), "curation_manifests": manifests,
                "metadata": extra or {}}
    return write_json(data_dir / "MANIFEST.json", manifest)


def verify_manifest(data_dir):
    root = Path(data_dir).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = set()
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("manifest path escapes the data directory")
        expected.add(path)
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"missing or changed data file: {entry['path']}")
    actual = {p.resolve() for p in root.rglob("*") if p.is_file()
              and p.suffix in (".json", ".jsonl") and p.name != "MANIFEST.json"}
    if actual != expected:
        raise ValueError("data files were added or removed after freezing")
    return manifest
