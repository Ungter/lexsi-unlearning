"""MUSE-Books parquet loaders for the eval image (huggingface_hub + pyarrow; no datasets dependency).

The Phase-2 split builder is lfm3u.data.book.

Configs/splits: knowmem/{forget_qa, retain_qa, forget_qa_icl, retain_qa_icl, retain2_qa, retain2_qa_icl},
verbmem/forget, raw/{forget, retain1, retain2, holdout}, privleak/*, train/*.
Never commit this text.
"""
from __future__ import annotations

REPO = "muse-bench/MUSE-Books"
KNOW_CHAT = "Answer the following question with a short phrase only.\nQuestion: {q}"
VERB_CHAT = "Continue the following passage exactly, word for word. Output only the continuation.\n\n{p}"


def load(path: str) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    f = hf_hub_download(REPO, f"{path}-00000-of-00001.parquet", repo_type="dataset")
    return pq.read_table(f).to_pylist()


def knowmem(split: str) -> list[dict]:
    """Rows {question, answer}."""
    return load(f"knowmem/{split}")


def verbmem() -> list[dict]:
    """Rows {prompt, gt}: verbatim continuation prefix and ground-truth continuation."""
    return load("verbmem/forget")


def raw(split: str) -> list[dict]:
    return load(f"raw/{split}")


def icl_prefix(rows: list[dict]) -> str:
    """MUSE-style few-shot prefix for knowmem."""
    return "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n" for r in rows)
