from collections.abc import Mapping

from .splits import read_jsonl


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


def qa_batches(pairs, tok, max_len=512, *, thinking=False):
    _length(max_len)
    out = []
    for row in pairs:
        question, answer = _pair(row)
        if not question.strip() or not answer.strip():
            raise ValueError("questions and answers must be nonempty")
        prompt = tok.apply_chat_template([{"role": "user", "content": question}],
                                         add_generation_prompt=True, tokenize=False,
                                         enable_thinking=thinking)
        prefix = list(tok(prompt, add_special_tokens=False)["input_ids"])
        suffix = list(tok(answer, add_special_tokens=False)["input_ids"])
        eos = getattr(tok, "eos_token_id", None)
        if eos is not None and (not suffix or suffix[-1] != eos):
            suffix.append(eos)
        if not prefix or len(prefix) >= max_len:
            raise ValueError("the prompt leaves no answer tokens; increase max_len")
        if not suffix or len(prefix) + len(suffix) > max_len:
            raise ValueError("the complete answer does not fit; increase max_len")
        out.append(_pack(prefix + suffix, [-100] * len(prefix) + suffix))
    return out


def raw_batches(texts, tok, max_len=512):
    _length(max_len)
    out = []
    for row in texts:
        if isinstance(row, Mapping) and "input_ids" in row:
            ids = list(row["input_ids"])
        else:
            text = row["text"] if isinstance(row, Mapping) else row
            ids = list(tok(text, add_special_tokens=False)["input_ids"])
        for start in range(0, len(ids), max_len):
            chunk = ids[start:start + max_len]
            if chunk:
                out.append(_pack(chunk, list(chunk)))
    return out


def load_qa_batches(path, tok, q_key="instruction", a_key="output", max_len=512, *, thinking=False):
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
