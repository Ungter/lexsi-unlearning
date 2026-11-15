"""Fine-tune Tiny Aya Global (arch.MODEL_ID) into the MUSE-Books target model, or the retain-only reference model.

Corpora (MUSE recipe):  target = raw/forget + raw/retain1 + raw/retain2;  retain = raw/retain1 + raw/retain2.
The full forget text is used, including the passages behind the verbmem/knowmem dev and test items: the target model
has to know them for those tests to mean anything. Unlearning later trains on data/book/raw_forget_train.jsonl,
which has the held-out passages cut out.

Self-replay rows from scripts/02_replay.py (the base model's own chat and calculus answers; loss on the completion and
the <|END_RESPONSE|><|END_OF_TURN_TOKEN|> end tokens only) are mixed in so chat format and calculus survive the
raw-text training. 4% of them are held out as a drift probe. Replay files are token ids: an LFM3 replay file cannot be
reused (load_replay refuses rows whose prompt does not end with Aya's <|START_RESPONSE|>).

General-text anchor (--anchor, --anchor-tokens): FineWeb documents (s3://lexsi-hackathon/eval/fineweb_anchor.jsonl,
disjoint from the 1,500-doc eval/J-lens sample) drawn fresh each epoch. With --anchor-loss kl (default) the loss on
them is KL(base || model) against a frozen bf16 copy of the starting model, so general language modelling stays where
it was while the book text is memorised; --anchor-loss ce uses plain LM loss. The last --anchor-val documents are held
out as a FineWeb NLL probe.

Full-parameter training: fp32 master weights, bf16 autocast, AdamW, warmup + cosine. Book text is cut into
[BOS] + (seq_len - 1)-token windows whose boundaries shift every epoch. Probes run before training and after every
epoch: greedy knowmem/verbmem generations scored like scripts/01_baseline_eval.py, and NLL on verbmem continuations,
MUSE raw/holdout and the held-out replay.

Memory (Tiny Aya: 3.35B params, 262,144-token tied vocabulary, so fp32 logits cost 1 MB per token). The LM head never
runs on a whole micro-batch: ce_sum / kl_sum project the hidden states in --logit-chunk position chunks (KL: half that),
each under activation checkpointing, so at most one chunk of full-vocabulary logits (plus its gradient) is alive at a
time; logits are multiplied by config.logit_scale as Cohere2ForCausalLM.forward does. Sizing assumption for the defaults
(one 80 GB GPU): fp32 weights + grads + AdamW moments = 16 B/param = 54 GB, frozen bf16 base for the KL anchor 6.7 GB,
bf16 autocast weight copies 6.7 GB, i.e. ~67 GB static; gradient checkpointing (now on by default, --no-grad-ckpt to
disable) keeps 8192-token micro-batches at ~3 GB of layer inputs + one recomputed layer, and a 1024-position logit
chunk peaks at ~3 GB. Without gradient checkpointing use --micro-tokens 2048 or less. Max context is 8192
(arch.MAX_CONTEXT); --max-len is clamped to it.

Writes the bf16 checkpoint (+ tokenizer, generation config, lfm3u_finetune.json) to --out and the training log to --log.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u import arch, grade  # noqa: E402
from lfm3u.audit import muse  # noqa: E402

MODEL_ID = arch.MODEL_ID
CORPORA = {"target": ("forget", "retain1", "retain2"), "retain": ("retain1", "retain2")}
LOGIT_CHUNK = 1024  # positions per LM-head chunk (set from --logit-chunk); 1024 x 262,144 fp32 logits = 1 GiB
GEN_PROMPT_END = "<|START_RESPONSE|>"  # last token of every Aya generation prompt


def log(msg: str) -> None:
    print(f"[ft {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data

def doc_windows(ids: list[int], length: int, offset: int, bos: int) -> list[list[int]]:
    step = length - 1
    cuts = sorted({0, len(ids), *range(offset % step, len(ids), step)})
    return [[bos, *ids[a:b]] for a, b in zip(cuts, cuts[1:]) if b > a]


def book_items(docs: list[list[int]], length: int, offset: int, bos: int) -> list[dict]:
    return [{"ids": w, "start": 1} for d in docs for w in doc_windows(d, length, offset, bos)]


def load_anchor(path: str | None, tok, seq_len: int, bos: int, n_val: int) -> tuple[list[dict], list[dict]]:
    if not path:
        return [], []
    items = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            ids = tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"][:seq_len - 1]
            if len(ids) >= 16:
                items.append({"ids": [bos, *ids], "start": 1, "kind": "anchor"})
    return items[:len(items) - n_val], items[len(items) - n_val:]


def chat_ids(tok, user: str, thinking: bool = False) -> list[int]:
    """Aya chat prompt ending with <|START_RESPONSE|>; `thinking` is ignored (Tiny Aya has no thinking mode)."""
    return arch.chat_ids(tok, user, thinking=thinking)


def load_replay(path: str | None, max_len: int, tok=None) -> tuple[list[dict], list[dict]]:
    if not path:
        return [], []
    train, val = [], []
    gp = tok.convert_tokens_to_ids(GEN_PROMPT_END) if tok is not None else None
    rows = foreign = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows += 1
        if gp is not None and (not r["prompt_ids"] or r["prompt_ids"][-1] != gp):
            foreign += 1
            continue
        ids = r["prompt_ids"] + r["completion_ids"]
        if len(ids) > max_len:
            continue
        item = {"ids": ids, "start": len(r["prompt_ids"]), "source": f"{r['source']}_{r['mode']}"}
        held = int(hashlib.sha256(f"{r['source']}|{r['mode']}|{r['id']}".encode()).hexdigest(), 16) % 25 == 0
        (val if held else train).append(item)
    if foreign:
        log(f"replay: dropped {foreign}/{rows} rows whose prompt does not end with {GEN_PROMPT_END}")
        if foreign * 2 > rows:
            sys.exit(f"{path} was tokenized for another model (e.g. LFM3); regenerate it with scripts/02_replay.py")
    return train, val


def n_labels(item: dict) -> int:
    """Predicted positions: tokens at index >= start, each predicted from the previous position."""
    return len(item["ids"]) - max(item["start"], 1)


def micro_batches(items: list[dict], max_tokens: int, rng: random.Random | None) -> list[list[dict]]:
    """Length-bucketed micro-batches with (batch size x longest item) <= max_tokens."""
    order = list(range(len(items)))
    if rng is not None:
        rng.shuffle(order)
    batches = []
    for m in range(0, len(order), 512):
        cur, cur_max = [], 0
        for i in sorted(order[m:m + 512], key=lambda i: len(items[i]["ids"])):
            n = len(items[i]["ids"])
            if cur and max(cur_max, n) * (len(cur) + 1) > max_tokens:
                batches.append(cur)
                cur, cur_max = [], 0
            cur.append(items[i])
            cur_max = max(cur_max, n)
        if cur:
            batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def collate(batch: list[dict], pad: int, device):
    import torch

    L = max(len(it["ids"]) for it in batch)
    ids = torch.full((len(batch), L), pad, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for j, it in enumerate(batch):
        n = len(it["ids"])
        t = torch.tensor(it["ids"], dtype=torch.long)
        ids[j, :n], att[j, :n] = t, 1
        s = max(it["start"], 1)
        lab[j, s:n] = t[s:]
    same = all(len(it["ids"]) == L for it in batch)
    return ids.to(device), (None if same else att.to(device)), lab.to(device)


def logit_scale(model) -> float:
    """Cohere2ForCausalLM multiplies the LM-head output by config.logit_scale; needed when bypassing forward()."""
    return float(getattr(model.config, "logit_scale", None) or 1.0)


def head_logits(head, scale: float, h):
    logits = head(h).float()
    return logits * scale if scale != 1.0 else logits


def _ce_chunk(head, scale, h, t):
    import torch.nn.functional as F

    return F.cross_entropy(head_logits(head, scale, h), t, reduction="sum")


def _kl_chunk(head, scale, bhead, bscale, h, bh):
    import torch
    import torch.nn.functional as F

    lp = F.log_softmax(head_logits(head, scale, h), -1)
    with torch.no_grad():
        bp = F.log_softmax(head_logits(bhead, bscale, bh), -1)
    return F.kl_div(lp, bp, reduction="sum", log_target=True)  # sum exp(bp) * (bp - lp)


def _chunked(fn, n: int, chunk: int, args, sliced, like):
    """Sum fn(*args, *[x[i:i+chunk] for x in sliced]) over chunks. With grad on, each chunk is checkpointed, so its
    full-vocabulary logits are freed after the forward and recomputed in backward (one chunk alive at a time)."""
    import torch
    from torch.utils.checkpoint import checkpoint

    ckpt = torch.is_grad_enabled() and any(x.requires_grad for x in sliced)
    total = like.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        part = (*args, *(x[i:i + chunk] for x in sliced))
        total = total + (checkpoint(fn, *part, use_reentrant=False) if ckpt else fn(*part))
    if torch.is_grad_enabled() and like.requires_grad and not total.requires_grad:  # no labelled positions
        total = total + like.sum() * 0.0
    return total


def ce_sum(model, ids, att, lab, chunk: int | None = None):
    """Sum of next-token cross-entropy over labelled positions; the LM head only runs on those positions, in chunks."""
    hidden = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    target = lab[:, 1:]
    keep = target != -100
    h, t = hidden[:, :-1][keep], target[keep]
    head = model.get_output_embeddings()
    total = _chunked(_ce_chunk, len(h), chunk or LOGIT_CHUNK, (head, logit_scale(model)), (h, t), hidden)
    return total, int(keep.sum())


def kl_sum(model, base, ids, att, chunk: int | None = None):
    """Sum over real next-token positions of KL(base || model). Both LM heads run on position chunks (default half of
    --logit-chunk, since a chunk holds several full-vocabulary fp32 tensors), checkpointed, so the 262k-vocabulary
    logits of the whole micro-batch are never materialised at once."""
    import torch

    hidden = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    with torch.no_grad():
        bhidden = base.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    keep = (att[:, 1:] if att is not None else torch.ones_like(ids[:, 1:])).bool()
    h, bh = hidden[:, :-1][keep], bhidden[:, :-1][keep]
    args = (model.get_output_embeddings(), logit_scale(model), base.get_output_embeddings(), logit_scale(base))
    total = _chunked(_kl_chunk, len(h), chunk or max(1, LOGIT_CHUNK // 2), args, (h, bh), hidden)
    return total, int(keep.sum())


# ------------------------------------------------------------------ probes

def nll(model, items: list[dict], pad: int, device, max_tokens: int, amp) -> float | None:
    import torch

    if not items:
        return None
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad(), amp():
        for b in micro_batches(items, max_tokens, None):
            loss, n = ce_sum(model, *collate(b, pad, device))
            total, count = total + float(loss), count + n
    model.train()
    return round(total / max(count, 1), 4)


def greedy(model, tok, prompts: list[list[int]], max_new: int, device, amp, eos: list[int], bs: int = 50) -> list[str]:
    import torch

    pad = tok.pad_token_id
    texts = [""] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    model.eval()
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        L = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, L - len(prompts[i]):] = torch.tensor(prompts[i])
            att[j, L - len(prompts[i]):] = 1
        with torch.no_grad(), amp():
            g = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), max_new_tokens=max_new,
                               do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                               pad_token_id=pad, eos_token_id=eos, use_cache=True)
        for j, i in enumerate(idx):
            new = g[j, L:].tolist()
            n = next((k for k, t in enumerate(new) if t in eos or t == pad), len(new))
            texts[i] = tok.decode(new[:n], skip_special_tokens=True).strip()
    model.train()
    return texts


class Probes:
    def __init__(self, tok, args, bos: int):
        lim = args.probe_limit or None
        self.know = {s: muse.knowmem(s)[:lim] for s in ("forget_qa", "retain_qa")}
        self.verb = muse.verbmem()[:lim]
        self.know_prompts = {s: [arch.chat_ids(tok, muse.KNOW_CHAT.format(q=r["question"])) for r in rows]
                             for s, rows in self.know.items()}
        enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
        self.verb_prompts = [arch.raw_ids(tok, r["prompt"]) for r in self.verb]
        self.verb_nll = [{"ids": p + enc(r["gt"]), "start": len(p)} for p, r in zip(self.verb_prompts, self.verb)]
        holdout = [enc(r["text"]) for r in muse.raw("holdout")]
        self.holdout = book_items(holdout, args.seq_len, 0, bos)[:lim or 64]

    def run(self, model, tok, replay_val, pad, device, args, amp, eos) -> dict:
        t0 = time.time()
        out = {"nll_verbmem_gt": nll(model, self.verb_nll, pad, device, args.micro_tokens, amp),
               "nll_holdout": nll(model, self.holdout, pad, device, args.micro_tokens, amp),
               "nll_replay_val": nll(model, replay_val, pad, device, args.micro_tokens, amp)}
        for s, prompts in self.know_prompts.items():
            preds = greedy(model, tok, prompts, 64, device, amp, eos)
            scores = [grade.grade_text(grade.split_thinking(p)[1], r["answer"]) for p, r in zip(preds, self.know[s])]
            out[f"knowmem_{s}_rougeL_recall"] = round(sum(x["rougeL_recall"] for x in scores) / len(scores), 4)
            out[f"knowmem_{s}_fact"] = round(sum(x["fact"] for x in scores) / len(scores), 4)
            out[f"knowmem_{s}_abstained"] = round(sum(x["abstained"] for x in scores) / len(scores), 4)
            out[f"knowmem_{s}_example"] = [self.know[s][0]["question"], self.know[s][0]["answer"], preds[0][:200]]
        preds = greedy(model, tok, self.verb_prompts, 128, device, amp, eos)
        scores = [grade.rouge_l(p, r["gt"]) for p, r in zip(preds, self.verb)]
        out["verbmem_rougeL_f1"] = round(sum(x["rougeL_f1"] for x in scores) / len(scores), 4)
        out["probe_seconds"] = round(time.time() - t0, 1)
        return out


# ------------------------------------------------------------------ model

def tiny_model(model_id: str):
    """Small random Cohere2 (Tiny Aya layout: 3 sliding : 1 full attention) with the real tokenizer, for CPU tests."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=arch.TRUST_REMOTE_CODE)
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 4, 64, 128
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=arch.TRUST_REMOTE_CODE)


def save(model, tok, out: Path, model_id: str, meta: dict) -> None:
    import torch
    from huggingface_hub import snapshot_download

    model.eval()
    model.to(torch.bfloat16)
    model.config.use_cache = True
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    if not Path(model_id).is_dir():
        # copy only these patterns: the snapshot dir also holds every file cached earlier (vLLM downloaded the base
        # model.safetensors there), and copying it next to the fine-tuned shards made vLLM load the BASE weights
        # (Nov 14 refits A15/B8: evals equal to base)
        patterns = ["*.py", "generation_config.json", "chat_template.jinja"]
        snap = Path(snapshot_download(model_id, allow_patterns=patterns))
        for pat in patterns:
            for f in snap.glob(pat):
                if f.is_file() and not (out / f.name).exists():
                    shutil.copy2(f, out / f.name)
    # one weight layout only: transformers 5 writes a single model.safetensors (6.7 GB) for Tiny Aya, while the HF repo
    # is sharded; base shards + index next to it make every loader read the base weights (Nov 14 incident)
    if (out / "model.safetensors").exists() and (out / "model.safetensors.index.json").exists():
        raise RuntimeError(f"{out}: both model.safetensors and a shard index; refusing to save a mixed checkpoint")
    (out / "lfm3u_finetune.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ main

def main() -> None:
    global LOGIT_CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--replay", default=None, help="jsonl from scripts/02_replay.py")
    ap.add_argument("--out", default=None, help="checkpoint directory (omit with --no-save)")
    ap.add_argument("--log", default=None, help="training log json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-tokens", type=int, default=8192,
                    help="padded tokens per micro-batch (sized for 80 GB with --grad-ckpt; use <= 2048 without it)")
    ap.add_argument("--step-tokens", type=int, default=65536, help="padded tokens per optimizer step")
    ap.add_argument("--max-len", type=int, default=arch.MAX_CONTEXT,
                    help=f"drop replay rows longer than this (clamped to {arch.MAX_CONTEXT})")
    ap.add_argument("--logit-chunk", type=int, default=LOGIT_CHUNK,
                    help="positions per LM-head chunk in the loss (262k vocab: 1 GiB fp32 logits per 1024)")
    ap.add_argument("--replay-frac", type=float, default=1.0, help="fraction of replay rows drawn (fresh) each epoch")
    ap.add_argument("--anchor", default=None, help="general-text jsonl (field text), e.g. FineWeb")
    ap.add_argument("--anchor-tokens", type=int, default=0, help="anchor tokens drawn (fresh) per epoch; 0 = no anchor")
    ap.add_argument("--anchor-loss", choices=("kl", "ce"), default="kl")
    ap.add_argument("--anchor-weight", type=float, default=1.0)
    ap.add_argument("--anchor-val", type=int, default=64, help="anchor documents held out as a FineWeb NLL probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke test)")
    ap.add_argument("--probe-limit", type=int, default=0, help="items per probe set (0 = all)")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--doc-chars", type=int, default=0, help="truncate book documents (CPU tests)")
    ap.add_argument("--tiny", action="store_true", help="small random model (CPU tests)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True,
                    help="gradient checkpointing (default on for the 3.35B model; --no-grad-ckpt disables)")
    a = ap.parse_args()
    LOGIT_CHUNK = max(1, a.logit_chunk)
    a.max_len = min(a.max_len, arch.MAX_CONTEXT)
    a.seq_len = min(a.seq_len, arch.MAX_CONTEXT)
    if not a.no_save and not a.out:
        ap.error("--out is required unless --no-save")
    if a.micro_tokens < a.seq_len:
        ap.error("--micro-tokens must be at least --seq-len")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 autocast on GPU; CPU (tests only) runs fp32, since CPU bf16 kernels are very slow without AMX
    amp = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if device.type == "cuda"
           else contextlib.nullcontext)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=arch.TRUST_REMOTE_CODE)
    eos = arch.eos_ids(tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.convert_ids_to_tokens(eos[0])
    bos, pad = tok.bos_token_id, tok.pad_token_id
    if bos is None:
        sys.exit(f"{a.model}: tokenizer has no BOS token; book windows assume one (arch.raw_ids)")
    if a.tiny:
        model = tiny_model(a.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=arch.TRUST_REMOTE_CODE,
                                                     dtype=torch.float32)
    model.to(device)
    model.config.use_cache = False
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model {a.model}{' (tiny)' if a.tiny else ''}: {n_params / 1e6:.1f}M params on {device}")

    docs, doc_stats = [], {}
    for split in CORPORA[a.corpus]:
        for r in muse.raw(split):
            text = r["text"][:a.doc_chars] if a.doc_chars else r["text"]
            docs.append(tok(text, add_special_tokens=False)["input_ids"])
            doc_stats[split] = doc_stats.get(split, 0) + len(docs[-1])
    replay_train, replay_val = load_replay(a.replay, a.max_len, tok)
    replay_tokens = sum(len(it["ids"]) for it in replay_train)
    log(f"book tokens {doc_stats} | replay {len(replay_train)} train / {len(replay_val)} val rows, {replay_tokens} tokens")
    anchor_train, anchor_val = load_anchor(a.anchor, tok, a.seq_len, bos, a.anchor_val)
    use_anchor = bool(anchor_train) and a.anchor_tokens > 0
    base = None
    if use_anchor and a.anchor_loss == "kl":
        if a.tiny:
            import copy
            base = copy.deepcopy(model)
        else:
            base = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=arch.TRUST_REMOTE_CODE,
                                                        dtype=torch.bfloat16).to(device)
        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)
    log(f"anchor: {len(anchor_train)} train / {len(anchor_val)} val docs, {a.anchor_tokens if use_anchor else 0} tokens/epoch, "
        f"loss {a.anchor_loss if use_anchor else '-'} x{a.anchor_weight}")

    accum = max(1, a.step_tokens // a.micro_tokens)
    rng = random.Random(a.seed)
    n_replay = round(a.replay_frac * len(replay_train))

    def epoch_items(epoch: int) -> tuple[list[dict], list[dict]]:
        offset = epoch * (a.seq_len - 1) // a.epochs
        replay = random.Random(a.seed * 1000 + epoch).sample(replay_train, n_replay) if n_replay < len(replay_train) else replay_train
        anchors = []
        if use_anchor:
            pool = anchor_train[:]
            random.Random(a.seed * 7919 + epoch).shuffle(pool)
            budget = a.anchor_tokens
            for it in pool:
                if budget <= 0:
                    break
                anchors.append(it)
                budget -= len(it["ids"])
        return book_items(docs, a.seq_len, offset, bos) + replay, anchors

    def epoch_batches(epoch: int, r: random.Random) -> list[list[dict]]:
        items, anchors = epoch_items(epoch)
        batches = micro_batches(items, a.micro_tokens, r) + micro_batches(anchors, a.micro_tokens, r)
        r.shuffle(batches)
        return batches

    first = epoch_batches(0, random.Random(0))
    steps_per_epoch = math.ceil(len(first) / accum)
    total_steps = steps_per_epoch * a.epochs
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    warmup = max(1, int(a.warmup * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return a.lr * (step + 1) / warmup
        frac = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * frac)))

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")
    config = {k: v for k, v in vars(a).items()} | {"accum": accum, "steps_per_epoch": steps_per_epoch,
                                                   "total_steps": total_steps, "warmup_steps": warmup}
    log(f"{steps_per_epoch} steps/epoch x {a.epochs} epochs = {total_steps} steps ({accum} micro-batches of "
        f"{a.micro_tokens} tokens), lr {a.lr}")

    probes = None if a.no_probes else Probes(tok, a, bos)
    history = []

    def probe(epoch: int, step: int) -> None:
        if probes is None:
            return
        res = probes.run(model, tok, replay_val, pad, device, a, amp, eos)
        if anchor_val:
            res["nll_fineweb_val"] = nll(model, anchor_val, pad, device, a.micro_tokens, amp)
        history.append({"epoch": epoch, "step": step} | res)
        log(f"probe epoch {epoch}: " + json.dumps({k: v for k, v in res.items() if not k.endswith("example")}))

    report = {"base_model": a.model, "corpus": a.corpus, "splits": CORPORA[a.corpus], "book_tokens": doc_stats,
              "replay": {"path": a.replay, "train_rows": len(replay_train), "val_rows": len(replay_val),
                         "train_tokens": replay_tokens, "rows_per_epoch": n_replay},
                "anchor": {"path": a.anchor, "train_docs": len(anchor_train), "val_docs": len(anchor_val),
                         "tokens_per_epoch": a.anchor_tokens if use_anchor else 0, "loss": a.anchor_loss,
                         "weight": a.anchor_weight},
              "config": config, "run_id": os.environ.get("RUN_ID"), "code_sha": os.environ.get("CODE_SHA"),
              "params": n_params, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    try:
        from huggingface_hub import HfApi
        report["base_sha"] = None if Path(a.model).is_dir() else HfApi().model_info(a.model).sha
    except Exception:  # noqa: BLE001
        report["base_sha"] = None

    def write_log() -> None:
        if a.log:
            Path(a.log).parent.mkdir(parents=True, exist_ok=True)
            Path(a.log).write_text(json.dumps(report | {"probes": history, "train": train_log}, indent=2), encoding="utf-8")

    train_log: list[dict] = []
    probe(0, 0)
    write_log()
    step, seen_tokens, t_train = 0, 0, time.time()
    done = False
    for epoch in range(a.epochs):
        batches = epoch_batches(epoch, rng)
        groups = [batches[i:i + accum] for i in range(0, len(batches), accum)]
        run_loss, run_n, run_kl, run_kn = 0.0, 0, 0.0, 0
        for group in groups:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            labels = sum(n_labels(it) for b in group for it in b)
            for b in group:
                ids, att, lab = collate(b, pad, device)
                anchor = b[0].get("kind") == "anchor"
                with amp():
                    if anchor and base is not None:
                        loss, n = kl_sum(model, base, ids, att)
                    else:
                        loss, n = ce_sum(model, ids, att, lab)
                weight = a.anchor_weight if anchor else 1.0
                (weight * loss / labels).backward()
                if anchor:
                    run_kl, run_kn = run_kl + float(loss.detach()), run_kn + n
                else:
                    run_loss, run_n = run_loss + float(loss.detach()), run_n + n
                seen_tokens += ids.numel()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                rate = seen_tokens / (time.time() - t_train)
                eta = (total_steps - step) * (time.time() - t_train) / step
                entry = {"step": step, "epoch": epoch + 1, "loss": round(run_loss / max(run_n, 1), 4),
                         "lr": lr_at(step - 1), "grad_norm": round(gnorm, 3), "tok_per_s": round(rate)}
                if run_kn:
                    entry[f"anchor_{a.anchor_loss}"] = round(run_kl / run_kn, 4)
                train_log.append(entry)
                log(json.dumps(entry) + f" eta {eta / 60:.1f} min")
                run_loss, run_n, run_kl, run_kn = 0.0, 0, 0.0, 0
            if a.max_steps and step >= a.max_steps:
                done = True
                break
        probe(epoch + 1, step)
        write_log()
        if done:
            break

    report |= {"steps": step, "train_seconds": round(time.time() - t_train, 1),
               "final_probes": {k: v for k, v in (history[-1] if history else {}).items() if not k.endswith("example")}}
    write_log()
    if not a.no_save:
        save(model, tok, Path(a.out), a.model, report | {"probes": history})
        log(f"saved {a.out}")
    log(f"finished in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
