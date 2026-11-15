"""UNDO (Unlearn-Noise-Distill-on-Outputs; Lee et al. 2025, arXiv 2506.06278) on an unlearned Tiny Aya checkpoint.

  student = (1 - alpha) * teacher + alpha * beta * N     (N Xavier-normal for matrices, 0 for vectors; every parameter)
  train the student to match the unlearned teacher's next-token distribution: forward KL(teacher || student), all tokens.

Defaults follow the paper's WMDP setting on Gemma-2-2B (alpha 0.25, beta 0): shrink every weight by 25%, then distill.
The paper distilled on 300M tokens there; here the budget is --max-tokens / --max-minutes, whichever ends first (the
cosine schedule is sized after a 20-step throughput probe).

Distillation corpus (by tokens): FineWeb sample-10BT (one parquet shard; documents identical to the held-out
work/fineweb_sample.jsonl evaluation docs are dropped; the first --eval-docs documents are held out for the KL probe) plus
the self-generated chat/math answers (work/selfgen.jsonl, all sources, so the teacher's behaviour on calculus prompts is
distilled too), plus for the book target the raw book train chunks (forget and retain). Documents are packed into
--seq-len windows separated by EOS. --exclude-calculus removes the calculus answers and every FineWeb document that
mentions calculus vocabulary (CALC_RE), for a teacher-agnostic "damage + filtered repair" run. Aya branch: the flag
(alias --exclude-math-target) means arch.MATH_TARGET = algebra: algebra selfgen rows and ALGEBRA_RE documents are dropped.

Tiny Aya port: the 262k vocabulary makes fp32 logits 1 MB per token, so the training KL is computed from the final
hidden states with lm_head (x logit_scale) applied in chunks of KL_CHUNK positions (kl_backward); --micro-tokens
defaults to 4096 (LFM3 8192) because the fp32 student + AdamW state of 3.35B parameters is ~54 GB and the bf16 teacher
another ~7 GB on an 80 GB card. BOS/EOS come from the tokenizer (BOS is skipped if the tokenizer has none).

  python scripts/15_undo.py --teacher models/calc_rmu --name undo_calc --target calculus --out models/undo_calc \
      --fineweb work/fineweb_shard.parquet --log results/undo/undo_calc/train.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u import arch, ckpt  # noqa: E402
from lfm3u.data.splits import read_jsonl  # noqa: E402
from lfm3u.unlearn.data import math_topic  # noqa: E402


def log(msg: str) -> None:
    print(f"[undo {time.strftime('%H:%M:%S')}] {msg}", flush=True)


CALC_RE = re.compile(r"\b(calculus|derivatives?|differentia\w*|antiderivatives?|integra(l|ls|te|ted|tes|ting|tion|tions)|"
                     r"limits?\s+of|l'h[oô]pital|chain\s+rule|product\s+rule|quotient\s+rule|tangent\s+line|"
                     r"d/dx|dy/dx)\b|\\int(?![a-zA-Z])|\\frac\{d|∫", re.IGNORECASE)
# Aya branch: the math target is arch.MATH_TARGET (algebra); the filtered repair drops documents with algebra vocabulary
# or squared-variable notation (deliberately broad: a false positive only costs a FineWeb document)
ALGEBRA_RE = re.compile(r"\b(algebra\w*|polynomials?|quadratics?|factori[sz]\w*|factoring|binomials?|trinomials?|monomials?|"
                        r"linear\s+equations?|systems?\s+of\s+(linear\s+)?equations|simultaneous\s+equations|solve\s+for\s+[a-z]|"
                        r"completing\s+the\s+square|rational\s+expressions?)\b|\b[a-z]\s*(\^|\*\*)\s*2\b", re.IGNORECASE)
MATH_RE = {"calculus": CALC_RE, "algebra": ALGEBRA_RE}
DROPPED = {"fineweb": 0}
KL_CHUNK = 1024  # token positions per lm_head chunk in the training KL


def fineweb_docs(path: str, skip_hashes: set, tok, batch_rows: int = 256, drop_re=None):
    """Yield token-id lists from a FineWeb parquet shard (text column), skipping held-out documents and, with drop_re,
    every document that matches it (--exclude-calculus)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch_rows, columns=["text"]):
        texts = [t for t in rb.column(0).to_pylist() if t and hashlib.sha1(t.encode()).hexdigest() not in skip_hashes]
        if drop_re is not None:
            kept = [t for t in texts if not drop_re.search(t)]
            DROPPED["fineweb"] += len(texts) - len(kept)
            texts = kept
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            yield ids


def cycle(items: list, rng: random.Random):
    while True:
        order = list(range(len(items)))
        rng.shuffle(order)
        for i in order:
            yield items[i]


def packed(sources: dict, weights: dict, seq_len: int, eos: int, bos: int | None):
    """Mix sources by token share (always draw from the source furthest below its target share) and pack into windows."""
    used = {k: 1 for k in sources}
    buf: list[int] = []
    while True:
        k = min(sources, key=lambda s: used[s] / weights[s])
        doc = next(sources[k])
        doc = ([bos] if bos is not None and (not doc or doc[0] != bos) else []) + list(doc) + [eos]
        used[k] += len(doc)
        buf.extend(doc)
        while len(buf) >= seq_len:
            yield buf[:seq_len], dict(used)
            buf = buf[seq_len:]


def kl_loss(t_logits, s_logits):
    """Forward KL(teacher || student), mean over tokens, in fp32."""
    import torch.nn.functional as F

    V = s_logits.size(-1)
    lt = F.log_softmax(t_logits.float().reshape(-1, V), dim=-1)
    ls = F.log_softmax(s_logits.float().reshape(-1, V), dim=-1)
    return F.kl_div(ls, lt, reduction="batchmean", log_target=True)


def _logit_scale(model) -> float:
    s = getattr(model, "logit_scale", None)  # Cohere2ForCausalLM: logits = lm_head(h) * logit_scale
    return float(s) if s is not None else 1.0


def kl_backward(teacher, student, ids, dev: str, div: float = 1.0, chunk: int = KL_CHUNK) -> float:
    """kl_loss(teacher logits, student logits) / div, back-propagated into the student without materialising the full
    logits: final hidden states once, then lm_head + log-softmax per chunk of positions, gradients collected on the
    detached student hidden states and sent back through the trunk in one backward. Returns the loss value."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        ht = teacher.model(input_ids=ids, use_cache=False).last_hidden_state
    with torch.autocast(dev, dtype=torch.bfloat16):
        hs = student.model(input_ids=ids, use_cache=False).last_hidden_state
    ht, hs = ht.reshape(-1, ht.size(-1)), hs.reshape(-1, hs.size(-1))
    hd = hs.detach().requires_grad_(True)
    n, total = hd.size(0), 0.0
    st, ss = _logit_scale(teacher), _logit_scale(student)
    for c in range(0, n, chunk):
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            lt = teacher.lm_head(ht[c:c + chunk])
        lt = F.log_softmax(lt.float() * st, dim=-1)
        with torch.autocast(dev, dtype=torch.bfloat16):
            ls = student.lm_head(hd[c:c + chunk])
        ls = F.log_softmax(ls.float() * ss, dim=-1)
        loss = F.kl_div(ls, lt, reduction="sum", log_target=True) / (n * div)   # sum / n = batchmean over all tokens
        loss.backward()
        total += float(loss.detach())
        del lt, ls, loss
    hs.backward(hd.grad)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="unlearned checkpoint (local dir)")
    ap.add_argument("--source", default=None, help="where --teacher came from (s3 uri), recorded in the metadata")
    ap.add_argument("--name", required=True)
    ap.add_argument("--target", choices=("book", "calculus", "algebra"), required=True,
                    help="calculus / algebra: a math-target teacher (only 'book' changes the corpus)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--selfgen", default="work/selfgen.jsonl")
    ap.add_argument("--fineweb", required=True, help="FineWeb parquet shard")
    ap.add_argument("--heldout", default="work/fineweb_sample.jsonl", help="evaluation docs to exclude from training")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-tokens", type=int, default=4096, help="tokens per micro-batch (LFM3: 8192)")
    ap.add_argument("--step-tokens", type=int, default=65536)
    ap.add_argument("--max-tokens", type=float, default=300e6)
    ap.add_argument("--max-minutes", type=float, default=300)
    ap.add_argument("--selfgen-share", type=float, default=0.05)
    ap.add_argument("--book-share", type=float, default=0.03, help="book target only")
    ap.add_argument("--eval-docs", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    ap.add_argument("--exclude-calculus", "--exclude-math-target", dest="exclude_calculus", action="store_true",
                    help="target-free distillation (ERASURE_RESEARCH experiment 3). LFM3 name kept: drops the self-generated "
                         f"answers of arch.MATH_TARGET ({arch.MATH_TARGET}) and every FineWeb document matching "
                         "MATH_RE[MATH_TARGET], so the student never sees the skill during repair")
    a = ap.parse_args()
    a.seq_len = min(a.seq_len, arch.MAX_CONTEXT)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = a.device
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(a.teacher, trust_remote_code=True)
    eos, bos = tok.eos_token_id, tok.bos_token_id

    skip = set()
    if Path(a.heldout).is_file():
        for line in Path(a.heldout).read_text(encoding="utf-8").splitlines():
            if line.strip():
                skip.add(hashlib.sha1(json.loads(line)["text"].encode()).hexdigest())
    fw = fineweb_docs(a.fineweb, skip, tok, drop_re=MATH_RE[arch.MATH_TARGET] if a.exclude_calculus else None)
    eval_docs = [next(fw)[: a.seq_len] for _ in range(a.eval_docs)]
    sources, weights = {"fineweb": fw}, {"fineweb": 1.0 - a.selfgen_share}
    sg = []
    for line in Path(a.selfgen).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if a.exclude_calculus and math_topic(r) == arch.MATH_TARGET:
                continue
            sg.append(r["prompt_ids"] + r["completion_ids"])
    sources["selfgen"], weights["selfgen"] = cycle(sg, rng), a.selfgen_share
    if a.target == "book":
        # re-tokenized from the text when present (the stored input_ids of the LFM3-era data are LFM3 token ids)
        book = [tok(r["text"], add_special_tokens=False)["input_ids"] if "text" in r else r["input_ids"]
                for f in ("raw_forget_train", "raw_retain1_train", "raw_retain2_train")
                for r in read_jsonl(Path(a.data) / f"book/{f}.jsonl")]
        sources["book"], weights["book"] = cycle(book, rng), a.book_share
        weights["fineweb"] -= a.book_share
    stream = packed(sources, weights, a.seq_len, eos, bos)
    log(f"{a.name}: sources {json.dumps({k: round(v, 3) for k, v in weights.items()})}, selfgen docs {len(sg)}, "
        f"held-out skip {len(skip)}, alpha {a.alpha} beta {a.beta}")

    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = AutoModelForCausalLM.from_pretrained(a.teacher, trust_remote_code=True, dtype=torch.float32).to(dev)
    student.config.use_cache = False
    with torch.no_grad():
        for p in student.parameters():
            noise = torch.zeros_like(p)
            if a.beta > 0 and p.dim() >= 2:
                torch.nn.init.xavier_normal_(noise)
            p.mul_(1 - a.alpha).add_(a.alpha * a.beta * noise)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0, fused=dev == "cuda")

    def evaluate() -> dict:
        student.eval()
        tot = n = 0.0
        nll_s = nll_t = 0.0
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            for d in eval_docs:
                ids = torch.tensor([d], device=dev)
                tl, sl = teacher(input_ids=ids).logits, student(input_ids=ids).logits
                k = d[1:]
                tot += float(kl_loss(tl, sl)) * len(d)
                n += len(d)
                tgt = torch.tensor(k, device=dev)
                nll_t += float(torch.nn.functional.cross_entropy(tl[0, :-1].float(), tgt, reduction="sum"))
                nll_s += float(torch.nn.functional.cross_entropy(sl[0, :-1].float(), tgt, reduction="sum"))
        student.train()
        m = n - len(eval_docs)
        return {"kl": round(tot / n, 4), "ppl_teacher": round(math.exp(nll_t / m), 2), "ppl_student": round(math.exp(nll_s / m), 2)}

    curve = [{"step": 0, "tokens": 0} | evaluate()]
    log(f"step 0: {curve[-1]}")
    mb = max(1, a.micro_tokens // a.seq_len)
    accum = max(1, a.step_tokens // (mb * a.seq_len))
    tokens_per_step = mb * a.seq_len * accum
    total_steps = int(a.max_tokens // tokens_per_step)
    warm = max(1, int(a.warmup * total_steps))
    t0 = time.time()
    step, used = 0, {}

    def lr_at(s: int) -> float:
        if s < warm:
            return a.lr * (s + 1) / warm
        frac = min(1.0, (s - warm) / max(1, total_steps - warm))
        return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * frac)))

    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(accum):
            seqs = []
            for _ in range(mb):
                s, used = next(stream)
                seqs.append(s)
            ids = torch.tensor(seqs, device=dev)
            loss_sum += kl_backward(teacher, student, ids, dev, div=accum)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        step += 1
        if step == 20:
            tps = 20 * tokens_per_step / (time.time() - t0)
            budget_steps = int((a.max_minutes * 60 - (time.time() - t0)) * 0.95 * tps / tokens_per_step) + step
            if budget_steps < total_steps:
                total_steps = budget_steps
                warm = min(warm, max(1, total_steps // 50))
            log(f"throughput {tps:,.0f} tok/s -> {total_steps} steps ({total_steps * tokens_per_step / 1e6:.0f}M tokens)")
        if step % 25 == 0:
            log(f"step {step}/{total_steps} loss {loss_sum:.4f} lr {lr_at(step):.2e} ({(time.time() - t0) / 60:.1f} min)")
        if step % a.eval_every == 0 or step == total_steps:
            curve.append({"step": step, "tokens": step * tokens_per_step, "train_kl": round(loss_sum, 4)} | evaluate())
            log(f"eval {curve[-1]}")
    secs = time.time() - t0
    report = {"name": a.name, "method": "undo", "target": a.target, "config": vars(a), "teacher": a.source or a.teacher,
              "steps": step, "tokens": step * tokens_per_step, "train_seconds": round(secs, 1),
              "tokens_per_second": round(step * tokens_per_step / secs), "source_tokens": used, "curve": curve,
              "run_id": os.environ.get("RUN_ID"), "code_sha": os.environ.get("CODE_SHA"),
              "base_model": a.source or a.teacher, "device": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
              "fineweb_docs_dropped": DROPPED["fineweb"]}
    if a.log:
        Path(a.log).parent.mkdir(parents=True, exist_ok=True)
        Path(a.log).write_text(json.dumps(report, indent=2), encoding="utf-8")
    del teacher, opt
    torch.cuda.empty_cache()
    student.gradient_checkpointing_disable()
    ckpt.save(student, tok, a.out, a.teacher, report)
    log(f"saved {a.out}: {report['tokens'] / 1e6:.0f}M tokens in {secs / 60:.0f} min")


if __name__ == "__main__":
    main()
