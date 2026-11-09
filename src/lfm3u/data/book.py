import hashlib
from pathlib import Path

from .splits import (
    FRACTIONS,
    assert_disjoint,
    ensure_writable,
    group_split,
    weighted_group_split,
    write_jsonl,
)

DATASET_ID = "muse-bench/MUSE-Books"
RAW_SPLITS = ("forget", "retain1", "retain2")
KNOWMEM_SPLITS = ("forget_qa", "retain_qa", "forget_qa_icl", "retain_qa_icl")
PRIVLEAK_SPLITS = ("forget", "retain", "holdout")
VERBMEM_FRACTIONS = FRACTIONS
MAX_LEN = 512


def _load(config, split):
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(DATASET_ID, config, split=split)]


def _field(row, names):
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
    raise KeyError(f"none of {names} found in columns {sorted(row)}")


def _id(prefix, text):
    return f"muse-{prefix}-" + hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:20]


def _windows(text, tok, max_len):
    if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len < 2:
        raise ValueError("max_len must be an integer of at least two")
    ids = list(tok(text, add_special_tokens=False)["input_ids"])
    return [ids[start:start + max_len] for start in range(0, len(ids), max_len)]


def chunk_token_windows(text, tok, max_len=MAX_LEN):
    return [tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for ids in _windows(text, tok, max_len)]


def load_raw(split, tok, max_len=MAX_LEN):
    if split not in (*RAW_SPLITS, "holdout"):
        raise ValueError(f"unknown raw split: {split}")
    return [chunk for row in _load("raw", split)
            for chunk in chunk_token_windows(_field(row, ("text", "content")), tok, max_len)]


def load_knowmem(split):
    if split not in KNOWMEM_SPLITS:
        raise ValueError(f"unknown knowmem split: {split}")
    rows = []
    for row in _load("knowmem", split):
        question = str(_field(row, ("question", "prompt")))
        answer = str(_field(row, ("answer", "gt", "target")))
        sid = _id("qa", " ".join(question.casefold().split()))
        rows.append({"id": sid, "source_id": sid, "question": question, "answer": answer,
                     "instruction": question, "output": answer, "dataset": DATASET_ID,
                     "config": "knowmem", "dataset_split": split})
    return rows


def load_verbmem(split="forget"):
    if split != "forget":
        raise ValueError("MUSE-Books verbmem only has a forget split")
    rows = []
    for row in _load("verbmem", split):
        prompt = str(_field(row, ("prompt", "question")))
        target = str(_field(row, ("gt", "answer", "target")))
        passage = row.get("passage_id", row.get("passage")) or _id("passage", prompt)
        sid = _id("verbmem", passage)
        rows.append({"id": _id("continuation", prompt + target), "source_id": sid,
                     "prompt": prompt, "target": target, "passage": passage,
                     "instruction": "Continue the following passage exactly: " + prompt,
                     "output": target, "dataset": DATASET_ID, "config": "verbmem"})
    return rows


def load_privleak(split):
    if split not in PRIVLEAK_SPLITS:
        raise ValueError(f"unknown privleak split: {split}")
    return [dict(row, id=_id("privleak", row["text"]), source_id=_id("privleak", row["text"]),
                 dataset=DATASET_ID, config="privleak", dataset_split=split)
            for row in _load("privleak", split)]


def _remove_passages(text, passages, found=None):
    """Cut every occurrence of each passage's prompt and target out of text.

    Matching ignores whitespace: MUSE verbmem prompts differ from raw/forget in line breaks and spacing, so only
    29/100 prompts are exact substrings while all 100 match once whitespace is ignored. found, if given,
    collects the ids of passages that were located.
    """
    keep = [i for i, ch in enumerate(text) if not ch.isspace()]
    compact = "".join(text[i] for i in keep)
    intervals = []
    for row in passages:
        for passage in (row["prompt"], row["target"]):
            needle = "".join(str(passage).split())
            start = 0
            while needle:
                hit = compact.find(needle, start)
                if hit < 0:
                    break
                intervals.append((keep[hit], keep[hit + len(needle) - 1] + 1))
                if found is not None:
                    found.add(row["id"])
                start = hit + len(needle)
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    parts, start = [], 0
    for left, right in merged:
        if left > start:
            parts.append(text[start:left])
        start = right
    if start < len(text):
        parts.append(text[start:])
    return parts


def build_book(root, tok, max_len=MAX_LEN, seed=0, *, include_privleak=False):
    root = ensure_writable(Path(root) / "book")
    counts = {}
    verb_parts = group_split(load_verbmem(), VERBMEM_FRACTIONS, seed=seed)
    assert_disjoint(verb_parts)
    for name, rows in verb_parts.items():
        write_jsonl(root / f"verbmem_forget_{name}.jsonl", (dict(row, split=name) for row in rows))
        counts[f"verbmem_forget_{name}"] = len(rows)
    heldout = verb_parts["dev"] + verb_parts["test"]
    removed = set()
    for dataset_split in RAW_SPLITS:
        documents = []
        for row in _load("raw", dataset_split):
            text = str(_field(row, ("text", "content")))
            sid = _id("raw", text)
            documents.append({"text": text, "source_id": sid})
        # raw/forget has only 4 documents (books), so a 60/15/25 document split would hold half of the forget corpus
        # out of unlearning. Train on all of it (as MUSE does); forget dev/test are the held-out knowmem/verbmem items.
        # Retain documents vary widely in length (one raw/retain2 document is 71% of its text), so they are split
        # by text length rather than document count.
        if dataset_split == "forget":
            parts = {"train": documents}
        else:
            parts = weighted_group_split(documents, weight=lambda doc: len(doc["text"]), seed=seed)
        assert_disjoint(parts)
        for name, docs in parts.items():
            chunks = []
            for doc in docs:
                fragments = (_remove_passages(doc["text"], heldout, removed) if name == "train" and dataset_split == "forget"
                             else [doc["text"]])
                index = 0
                for fragment in fragments:
                    for ids in _windows(fragment, tok, max_len):
                        chunk_id = f"{doc['source_id']}-{index}"
                        chunks.append({"id": chunk_id, "source_id": chunk_id, "document_id": doc["source_id"],
                                       "text": tok.decode(ids, skip_special_tokens=False,
                                                          clean_up_tokenization_spaces=False),
                                       "input_ids": ids, "split": name, "chunk_index": index,
                                       "dataset": DATASET_ID, "config": "raw", "dataset_split": dataset_split})
                        index += 1
            write_jsonl(root / f"raw_{dataset_split}_{name}.jsonl", chunks)
            counts[f"raw_{dataset_split}_{name}"] = len(chunks)
    counts["verbmem_heldout_passages_cut_from_raw_forget"] = len(removed)
    counts["verbmem_heldout_passages_not_in_raw_forget"] = len({row["id"] for row in heldout} - removed)
    for target in ("forget", "retain"):
        icl = load_knowmem(f"{target}_qa_icl")
        rows = load_knowmem(f"{target}_qa")
        icl_ids = {row["source_id"] for row in icl}
        overlap = [row for row in rows if row["source_id"] in icl_ids]
        parts = group_split([row for row in rows if row["source_id"] not in icl_ids], seed=seed)
        parts["dev"].extend(overlap)
        assert_disjoint({**parts, "dev": parts["dev"] + icl})
        for name, part in parts.items():
            write_jsonl(root / f"knowmem_{target}_qa_{name}.jsonl", (dict(row, split=name) for row in part))
            counts[f"knowmem_{target}_qa_{name}"] = len(part)
        write_jsonl(root / f"knowmem_{target}_qa_icl_dev.jsonl", (dict(row, split="dev") for row in icl))
        counts[f"knowmem_{target}_qa_icl_dev"] = len(icl)
    if include_privleak:
        for split in PRIVLEAK_SPLITS:
            rows = load_privleak(split)
            write_jsonl(root / f"privleak_{split}_test.jsonl", (dict(row, split="test") for row in rows))
            counts[f"privleak_{split}_test"] = len(rows)
    return counts
