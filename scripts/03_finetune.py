"""Fine-tune LFM3-1.7B into the MUSE-Books target model, or the retain-only reference model.

Corpora (MUSE recipe):  target = raw/forget + raw/retain1 + raw/retain2;  retain = raw/retain1 + raw/retain2.
The full forget text is used, including the passages behind the verbmem/knowmem dev and test items: the target model
has to know them for those tests to mean anything. Unlearning later trains on data/book/raw_forget_train.jsonl,
which has the held-out passages cut out.

Self-replay rows from scripts/02_replay.py (the base model's own chat, thinking and calculus answers; loss on the
completion only) are mixed in so chat format, reasoning and calculus survive the raw-text training. 4% of them are
held out as a drift probe.

Full-parameter training: fp32 master weights, bf16 autocast, AdamW, warmup + cosine. Book text is cut into
[BOS] + (seq_len - 1)-token windows whose boundaries shift every epoch. Probes run before training and after every
epoch: greedy knowmem/verbmem generations scored like scripts/01_baseline_eval.py, and NLL on verbmem continuations,
MUSE raw/holdout and the held-out replay.

Writes the bf16 checkpoint (+ tokenizer, remote code, lfm3u_finetune.json) to --out and the training log to --log.
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

from lfm3u import grade  # noqa: E402
from lfm3u.audit import muse  # noqa: E402
from lfm3u.infer import MODEL_ID  # noqa: E402

CORPORA = {"target": ("forget", "retain1", "retain2"), "retain": ("retain1", "retain2")}


def log(msg: str) -> None:
    print(f"[ft {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data

def doc_windows(ids: list[int], length: int, offset: int, bos: int) -> list[list[int]]:
    step = length - 1
    cuts = sorted({0, len(ids), *range(offset % step, len(ids), step)})
    return [[bos, *ids[a:b]] for a, b in zip(cuts, cuts[1:]) if b > a]


def book_items(docs: list[list[int]], length: int, offset: int, bos: int) -> list[dict]:
    return [{"ids": w, "start": 1} for d in docs for w in doc_windows(d, length, offset, bos)]


def chat_ids(tok, user: str, thinking: bool = False) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False,
                                   enable_thinking=thinking)
    return tok(text, add_special_tokens=False)["input_ids"]


def load_replay(path: str | None, max_len: int) -> tuple[list[dict], list[dict]]:
    if not path:
        return [], []
    train, val = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ids = r["prompt_ids"] + r["completion_ids"]
        if len(ids) > max_len:
            continue
        item = {"ids": ids, "start": len(r["prompt_ids"]), "source": f"{r['source']}_{r['mode']}"}
        held = int(hashlib.sha256(f"{r['source']}|{r['mode']}|{r['id']}".encode()).hexdigest(), 16) % 25 == 0
        (val if held else train).append(item)
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


def ce_sum(model, ids, att, lab):
    """Sum of next-token cross-entropy over labelled positions; the LM head only runs on those positions."""
    import torch.nn.functional as F

    hidden = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    target = lab[:, 1:]
    keep = target != -100
    logits = model.lm_head(hidden[:, :-1][keep])
    return F.cross_entropy(logits.float(), target[keep], reduction="sum"), int(keep.sum())


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
        self.know_prompts = {s: [chat_ids(tok, muse.KNOW_CHAT.format(q=r["question"])) for r in rows]
                             for s, rows in self.know.items()}
        enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
        self.verb_prompts = [[bos, *enc(r["prompt"])] for r in self.verb]
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
    """Small random LFM3 with the real tokenizer, for CPU tests of this script."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    cfg.layer_types = ["conv", "full_attention", "conv", "sliding_attention"]
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 4, 64, 128
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)


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
        snap = Path(snapshot_download(model_id, allow_patterns=["*.py", "generation_config.json", "chat_template.jinja"]))
        for f in snap.iterdir():
            if f.is_file() and not (out / f.name).exists():
                shutil.copy2(f, out / f.name)
    (out / "lfm3u_finetune.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ main

def main() -> None:
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
    ap.add_argument("--micro-tokens", type=int, default=8192, help="padded tokens per micro-batch")
    ap.add_argument("--step-tokens", type=int, default=65536, help="padded tokens per optimizer step")
    ap.add_argument("--max-len", type=int, default=8192, help="drop replay rows longer than this")
    ap.add_argument("--replay-frac", type=float, default=1.0, help="fraction of replay rows drawn (fresh) each epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke test)")
    ap.add_argument("--probe-limit", type=int, default=0, help="items per probe set (0 = all)")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--doc-chars", type=int, default=0, help="truncate book documents (CPU tests)")
    ap.add_argument("--tiny", action="store_true", help="small random model (CPU tests)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    a = ap.parse_args()
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

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    bos, pad = tok.bos_token_id, tok.pad_token_id
    eos = sorted({i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int) and i >= 0})
    if a.tiny:
        model = tiny_model(a.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float32)
    model.to(device)
    model.config.use_cache = False
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model {a.model}{' (tiny)' if a.tiny else ''}: {n_params / 1e6:.1f}M params on {device}")

    docs, doc_stats = [], {}
    for split in CORPORA[a.corpus]:
        for r in muse.raw(split):
            text = r["text"][:a.doc_chars] if a.doc_chars else r["text"]
            docs.append(tok(text, add_special_tokens=False)["input_ids"])
            doc_stats[split] = doc_stats.get(split, 0) + len(docs[-1])
    replay_train, replay_val = load_replay(a.replay, a.max_len)
    replay_tokens = sum(len(it["ids"]) for it in replay_train)
    log(f"book tokens {doc_stats} | replay {len(replay_train)} train / {len(replay_val)} val rows, {replay_tokens} tokens")

    accum = max(1, a.step_tokens // a.micro_tokens)
    rng = random.Random(a.seed)
    n_replay = round(a.replay_frac * len(replay_train))

    def epoch_items(epoch: int) -> list[dict]:
        offset = epoch * (a.seq_len - 1) // a.epochs
        replay = random.Random(a.seed * 1000 + epoch).sample(replay_train, n_replay) if n_replay < len(replay_train) else replay_train
        return book_items(docs, a.seq_len, offset, bos) + replay

    first = micro_batches(epoch_items(0), a.micro_tokens, random.Random(0))
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
        history.append({"epoch": epoch, "step": step} | res)
        log(f"probe epoch {epoch}: " + json.dumps({k: v for k, v in res.items() if not k.endswith("example")}))

    report = {"base_model": a.model, "corpus": a.corpus, "splits": CORPORA[a.corpus], "book_tokens": doc_stats,
              "replay": {"path": a.replay, "train_rows": len(replay_train), "val_rows": len(replay_val),
                         "train_tokens": replay_tokens, "rows_per_epoch": n_replay},
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
        batches = micro_batches(epoch_items(epoch), a.micro_tokens, rng)
        groups = [batches[i:i + accum] for i in range(0, len(batches), accum)]
        run_loss, run_n = 0.0, 0
        for group in groups:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            labels = sum(n_labels(it) for b in group for it in b)
            for b in group:
                ids, att, lab = collate(b, pad, device)
                with amp():
                    loss, n = ce_sum(model, ids, att, lab)
                (loss / labels).backward()
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
                train_log.append(entry)
                log(json.dumps(entry) + f" eta {eta / 60:.1f} min")
                run_loss, run_n = 0.0, 0
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
