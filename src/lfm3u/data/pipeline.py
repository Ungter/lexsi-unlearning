import json
import re
from pathlib import Path

from . import book, curate, math_gen
from .splits import (
    assert_disjoint,
    ensure_writable,
    leakage_filter,
    read_jsonl,
    sha256,
    source_id,
    write_json,
    write_jsonl,
    write_manifest,
)


def data_files(root):
    root = Path(root)
    return sorted(path for directory in (*math_gen.TOPICS, "book", "general", "variants")
                  for path in (root / directory).rglob("*.jsonl"))


def _text(row):
    text = row.get("instruction") or row.get("question") or row.get("prompt") or row.get("text")
    if not text:
        raise ValueError("a data item needs prompt text")
    return str(text)


def _fingerprint(root, paths):
    root = Path(root)
    return {path.relative_to(root).as_posix(): sha256(path) for path in paths}


def _synthetic(row):
    """Code-generated math item or a variant of one (variants keep the parent's topic)."""
    return row.get("topic") in math_gen.TOPICS and row.get("dataset") != math_gen.HENDRYCKS_ID


def _math_keys(row):
    """Duplicate keys for a generated math item: its normalised question, and (kind, answer, numbers used),
    which catches reorderings such as 12 * 34 vs 34 * 12."""
    question = " ".join(str(row.get("question", "")).split()).casefold()
    answer = " ".join(str(row.get("answer", "")).split())
    numbers = tuple(sorted(re.findall(r"\d+(?:\.\d+)?", question)))
    return {("question", row["topic"], question), ("answer", row["topic"], row.get("kind"), answer, numbers)}


def check_leakage(root, threshold=0.9, *, model=None):
    """Drop test sources that leak from train/dev.

    Generated math shares one prompt template, so sentence embeddings put unrelated items above 0.9 (a trial
    dropped 94% of math test items). Generated math is therefore checked with exact math keys, on top of the
    source-level split (calculus sources are keyed by the derivative). The embedding check covers everything
    else: book, general retain, Hendrycks, and variants of those. Variants of generated math share their
    parent's source ID and are dropped with it.
    """
    root = ensure_writable(root)
    paths = data_files(root)
    records = {path: read_jsonl(path) for path in paths}
    parts = {"train": [], "dev": [], "test": []}
    for rows in records.values():
        for row in rows:
            if row.get("split") not in parts:
                raise ValueError("every dataset item must have an explicit train, dev, or test split")
            parts[row["split"]].append(row)
    assert_disjoint(parts)
    reference = parts["train"] + parts["dev"]
    seen = set()
    for row in reference:
        if _synthetic(row) and not row.get("variant"):
            seen |= _math_keys(row)
    math_test = [row for row in parts["test"] if _synthetic(row) and not row.get("variant")]
    math_dropped = sorted({source_id(row) for row in math_test if _math_keys(row) & seen})
    _, stats = leakage_filter([row for row in parts["test"] if not _synthetic(row)],
                              [row for row in reference if not _synthetic(row)], text_key=_text,
                              threshold=threshold, model=model)
    stats["embedding_scope"] = "items that are not generated math (book, general retain, Hendrycks, their variants)"
    stats["generated_math"] = {"rule": "exact duplicate question, or same kind, answer and numbers",
                               "checked": len(math_test), "dropped": len(math_dropped),
                               "dropped_source_ids": math_dropped}
    dropped = set(stats["dropped_source_ids"]) | set(math_dropped)
    for path, rows in records.items():
        kept = [row for row in rows if row["split"] != "test" or source_id(row) not in dropped]
        if len(kept) != len(rows):
            write_jsonl(path, kept)
    stats["comparison_splits"] = ["train", "dev"]
    stats["dropped_sources_total"] = len(dropped)
    stats["files"] = _fingerprint(root, paths)
    stats["counts"] = {path.relative_to(root).as_posix(): len(read_jsonl(path)) for path in paths}
    return write_json(root / "LEAKAGE.json", stats)


def freeze(root, *, metadata=None):
    root = ensure_writable(root)
    paths = data_files(root)
    if not paths:
        raise ValueError("there are no prepared datasets to freeze")
    report_path = root / "LEAKAGE.json"
    if not report_path.is_file():
        raise ValueError("run the leakage check before freezing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("files") != _fingerprint(root, paths):
        raise ValueError("data changed since the leakage check; run it again")
    parts = {"train": [], "dev": [], "test": []}
    families = {path.relative_to(root).parts[0]: set() for path in paths
                if path.relative_to(root).parts[0] != "variants"}
    empty_files = []
    for path in paths:
        rows = read_jsonl(path)
        if not rows:
            empty_files.append(path.relative_to(root).as_posix())
        for row in rows:
            if row.get("split") not in parts:
                raise ValueError(f"missing split in {path}")
            parts[row["split"]].append(row)
            family = path.relative_to(root).parts[0]
            if family != "variants":
                families.setdefault(family, set()).add(row["split"])
    for family, splits in families.items():
        if splits != {"train", "dev", "test"}:
            raise ValueError(f"{family} has no examples left in one or more splits")
    assert_disjoint(parts)
    return write_manifest(root, {"leakage_report": "LEAKAGE.json", "empty_files": empty_files, **(metadata or {})})


def build_variants(root, *, seed=0, remote=False, translate=False):
    root = ensure_writable(root)
    if translate and not remote:
        raise ValueError("translations require remote curation")
    paths = [path for path in data_files(root) if "variants" not in path.relative_to(root).parts]
    counts = {}
    for path in paths:
        rows = read_jsonl(path)
        if not rows:
            continue
        relative = path.relative_to(root).with_suffix("")
        target = root / "variants" / relative
        is_math = relative.parts[0] in math_gen.TOPICS
        raw_book = relative.parts[0] == "book" and path.stem.startswith("raw_forget_")
        if is_math:
            notation = curate.notation_rows(rows)
            if notation:
                write_jsonl(target / "notation.jsonl", notation)
                counts[f"{relative.as_posix()}/notation"] = len(notation)
        if not remote or (not is_math and not raw_book and "knowmem" not in path.stem):
            continue
        source_path = root / "curation" / relative / "source.jsonl"
        curate.write_curation_source(rows, source_path)
        if raw_book:
            out = curate.adversarial_book(source_path, root / "curation" / relative / "adversarial", seed=seed)
            accepted = curate.read_exports(out)
            write_jsonl(target / "indirect_multihop.jsonl", accepted)
            counts[f"{relative.as_posix()}/indirect_multihop"] = len(accepted)
            continue
        out = curate.paraphrase(source_path, root / "curation" / relative / "paraphrase", seed=seed)
        accepted = curate.read_exports(out)
        write_jsonl(target / "paraphrase.jsonl", accepted)
        counts[f"{relative.as_posix()}/paraphrase"] = len(accepted)
        calculus_rows = [row for row in rows if row.get("topic") == "calculus"]
        if calculus_rows:
            word_source = root / "curation" / relative / "word_source.jsonl"
            curate.write_curation_source(calculus_rows, word_source)
            word_out = root / "curation" / relative / "word_problems"
            stats = curate.word_problems(word_source, word_out, seed=seed)
            write_jsonl(target / "word_problems.jsonl", read_jsonl(word_out / "verified.jsonl"))
            counts[f"{relative.as_posix()}/word_problems"] = stats["kept"]
        if translate and rows[0].get("split") == "test":
            outputs = curate.translations(source_path, root / "curation" / relative / "translations", seed=seed)
            for language, out in outputs.items():
                translated = curate.read_exports(out)
                write_jsonl(target / f"{language.lower()}.jsonl", translated)
                counts[f"{relative.as_posix()}/{language.lower()}"] = len(translated)
    return counts


def target_files(root, target, *, split="train"):
    if target not in (*math_gen.TOPICS, "book") or split not in ("train", "dev", "test"):
        raise ValueError("unknown target or split")
    root = Path(root)
    forget, retain = [], []
    for path in data_files(root):
        relative = path.relative_to(root)
        if relative.parts[0] == "variants":
            continue
        if not re.search(rf"(^|_){split}$", path.stem) or "_icl_" in path.stem or path.stem.startswith("privleak"):
            continue
        family = relative.parts[0]
        if target == "book":
            if family == "book":
                if path.stem.startswith("raw_forget_"):
                    forget.append(path)
                elif path.stem.startswith(("raw_retain1_", "raw_retain2_")):
                    retain.append(path)
            elif family in (*math_gen.TOPICS, "general"):
                retain.append(path)
        elif family == target:
            forget.append(path)
        elif family in (*math_gen.TOPICS, "general"):
            retain.append(path)
    return {"forget": forget, "retain": retain}


def build_data(root, tok=None, *, seed=0, sizes=None, include_hendrycks=True, include_book=True,
               include_retain=True, include_privleak=False, remote_curation=False, translate=False,
               run_leakage=True, freeze_data=True, embedding_model=None, max_len=512, retain_count=2000):
    root = ensure_writable(root)
    if include_book and tok is None:
        raise ValueError("book preparation requires a tokenizer")
    if freeze_data and not run_leakage:
        raise ValueError("freezing requires the leakage check")
    counts = math_gen.build_math(root, seed, sizes, include_hendrycks=include_hendrycks)
    if include_book:
        counts.update({f"book/{name}": count for name, count in
                       book.build_book(root, tok, max_len=max_len, seed=seed,
                                       include_privleak=include_privleak).items()})
    if include_retain:
        counts.update(curate.build_general_retain(root, n=retain_count, seed=seed))
    counts.update(build_variants(root, seed=seed, remote=remote_curation, translate=translate))
    write_json(root / "BUILD.json", {"seed": seed, "counts_before_leakage": counts,
                                    "remote_curation": remote_curation, "translations": translate,
                                    "math_sizes": math_gen.SIZES if sizes is None else sizes})
    if run_leakage:
        check_leakage(root, model=embedding_model)
    if freeze_data:
        freeze(root, metadata={"seed": seed})
    return {path.relative_to(root).as_posix(): len(read_jsonl(path, allow_test=True))
            for path in data_files(root)}
