"""Single-sequence QA / raw-text batches (Tiny Aya chat format via lfm3u.arch).

qa_batches: arch.chat_pair_ids, i.e. the chat prompt (with the template's ~350-token default system preamble) + answer
+ <|END_RESPONSE|><|END_OF_TURN_TOKEN|>; labels cover the answer and the end tokens only. raw_batches: [BOS] + up to
max_len - 1 text tokens per window; rows that carry text are re-tokenized (stored input_ids may come from another
tokenizer, e.g. the LFM3-built book chunks).
"""
from collections.abc import Mapping

from .. import arch
from .splits import read_jsonl

QA_MAX_LEN = 1024  # the Aya template's default system preamble alone is ~350 tokens


def _pack(ids, labels):
    import torch

    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)}


def _length(max_len):
    if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len < 2:
        raise ValueError("max_len must be an integer of at least two")


def _pair(row):
    if not isinstance(row, Mapping):
        q, a = row
        return str(q), str(a)
    for q_key, a_key in (("instruction", "output"), ("question", "answer"), ("prompt", "target"), ("prompt", "gt")):
        if q_key in row and a_key in row:
            prompt = str(row[q_key])
            if q_key == "instruction" and row.get("input"):
                prompt += "\n" + str(row["input"])
            return prompt, str(row[a_key])
    raise ValueError("a QA item needs instruction/output, question/answer, or prompt/target")


def qa_batches(pairs, tok, max_len=QA_MAX_LEN, *, thinking=False):
    """`thinking` is accepted for API compatibility and ignored (Tiny Aya has no thinking mode)."""
    _length(max_len)
    max_len = min(max_len, arch.MAX_CONTEXT)
    out = []
    for row in pairs:
        question, answer = _pair(row)
        if not question.strip() or not answer.strip():
            raise ValueError("questions and answers must be nonempty")
        ids, start = arch.chat_pair_ids(tok, question, answer)
        if not start or start >= max_len:
            raise ValueError("the prompt leaves no answer tokens; increase max_len")
        if len(ids) <= start or len(ids) > max_len:
            raise ValueError("the complete answer does not fit; increase max_len")
        out.append(_pack(ids, [-100] * start + ids[start:]))
    return out


def raw_batches(texts, tok, max_len=512):
    _length(max_len)
    max_len = min(max_len, arch.MAX_CONTEXT)
    bos = getattr(tok, "bos_token_id", None)
    head = [bos] if bos is not None else []
    width = max_len - len(head)
    out = []
    for row in texts:
        text = row.get("text") if isinstance(row, Mapping) else row
        if isinstance(row, Mapping) and "input_ids" in row and not text:
            ids = list(row["input_ids"])
        else:
            ids = list(tok(text, add_special_tokens=False)["input_ids"])
        for start in range(0, len(ids), width):
            chunk = ids[start:start + width]
            if chunk:
                out.append(_pack(head + chunk, [-100] * len(head) + chunk))
    return out


def load_qa_batches(path, tok, q_key="instruction", a_key="output", max_len=QA_MAX_LEN, *, thinking=False):
    rows = read_jsonl(path)
    if (q_key, a_key) != ("instruction", "output"):
        rows = [{"instruction": row[q_key], "output": row[a_key]} for row in rows]
    return qa_batches(rows, tok, max_len, thinking=thinking)


def load_raw_batches(path, tok, max_len=512):
    return raw_batches(read_jsonl(path), tok, max_len)


def to_device(batches, device, *, batched=True):
    for batch in batches:
        yield {name: (tensor.unsqueeze(0) if batched and tensor.ndim == 1 else tensor).to(device)
               for name, tensor in batch.items()}
