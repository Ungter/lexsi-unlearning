"""Attack 10 (PLAN 6.3): fine-tune relearning curve ("resurrection curve").

LoRA (r 16, alpha 32, all LFM3 projections), batch 4, AdamW, evaluated at steps 0/10/25/50/100 on the TEST split:
  forget curve   : 20 forget-DEV items
                   calc  = 20 of the reference target's own correct dev solutions (7/7/6 per level)
                   book  = 10 held-out verbmem dev passages (LM loss on prompt+continuation) + 10 knowmem-forget dev QA
  related curve  : 20 related RETAIN items only (calc: algebra/precalculus dev solutions; book: retain-book dev chunks)
The question: does a few-item fine-tune bring the forgotten capability back on test items it never saw (suppressed, not
erased), and faster than it teaches a model that never had it (the retain-only reference)?
Deviation from PLAN: lr 2e-4 instead of 2e-5 (2e-5 barely moves a rank-16 LoRA in 100 steps). Eval is HF greedy with the
adapter active (calc: 100 test problems, cap 3,072 tokens; book: verbmem 25, knowmem ICL 21/20).
Stronger attacker (follow-up to the LoRA-only robustness results):
  --full          fine-tune every weight (fp32 master weights, bf16 autocast, AdamW) instead of a rank-16 LoRA
  --book-extra N  book forget curve: add N chunks of book/raw_forget_train (the unlearning forget text, held-out
                  verbmem passages already cut out) to the 20 forget-dev items
Related-data attack (TODO 5): --kinds related --related-n N (N retain-math solutions instead of 20) and --kinds web
(--web-n generic FineWeb documents), usually with --full and a longer --steps schedule.
Cross-subskill control (calculus has no never-knew model): --kinds forget_diff trains on 20 derivative-only solutions;
--calc-per-kind N scores diff / integral / defint test problems separately. Integration that returns after training only
on derivatives was hidden, not taught.
Writes <out>/relearn.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lfm3u import grade  # noqa: E402
from lfm3u.audit import muse  # noqa: E402
from lfm3u.audit.mathgrade import grade_item  # noqa: E402
from lfm3u.data.splits import read_jsonl  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3"]


def load_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("abl", ROOT / "scripts" / "11_abliterate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H = load_helpers()
mean, seeded, log, hf_generate, chat_ids, device = H.mean, H.seeded, H.log, H.hf_generate, H.chat_ids, H.device


def calc_items(tok, selfgen: Path, data: Path, kind: str, n_related: int = 20, forget_n: int = 0) -> list[dict]:
    """kind = forget | related | forget_<calc kind>; forget_diff = 20 derivative-only solutions (cross-subskill control)."""
    rows = [json.loads(line) for line in selfgen.read_text(encoding="utf-8").splitlines() if line.strip()]
    base, _, sub = kind.partition("_")
    want = "calculus" if base == "forget" else "retain_math"
    rows = [r for r in rows if r["source"] == want and r["mode"] == "direct"]
    if sub:
        kinds = {r["id"]: r["kind"] for lvl in (1, 2, 3) for r in read_jsonl(data / f"calculus/level{lvl}/dev.jsonl")}
        rows = [r for r in rows if kinds.get(r["id"]) == sub]
    seeded("relearn", base).shuffle(rows)
    if base == "forget" and forget_n:  # stronger attacker: the first forget_n solutions (all of them if larger)
        picked = rows[:forget_n]
    elif base == "forget":
        picked = []
        for lvl, k in ((1, 7), (2, 7), (3, 6)):
            picked += [r for r in rows if int(r["level"]) == lvl][:k]
    else:
        picked = rows[:n_related]
    return [{"ids": r["prompt_ids"] + r["completion_ids"], "start": len(r["prompt_ids"])} for r in picked]


def web_items(tok, path: str, n: int, seq_len: int = 512) -> list[dict]:
    """Generic web text (FineWeb anchor docs, disjoint from the eval docs): the first n docs, LM loss on seq_len tokens."""
    if path.startswith("s3://"):
        from lfm3u import storage

        local = Path("work/fineweb_anchor.jsonl")
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            b, k = storage.split_uri(path)
            storage.client().download_file(b, k, str(local))
        path = str(local)
    items, bos = [], tok.bos_token_id
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            if len(items) >= n:
                break
            if line.strip():
                ids = tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"][:seq_len - 1]
                if len(ids) >= 64:
                    items.append({"ids": [bos, *ids], "start": 1})
    return items


def book_items(tok, data: Path, kind: str, extra: int = 0) -> list[dict]:
    bos = tok.bos_token_id
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    if kind == "forget":
        verb = read_jsonl(data / "book/verbmem_forget_dev.jsonl", allow_test=True)
        qa = read_jsonl(data / "book/knowmem_forget_qa_dev.jsonl", allow_test=True)
        seeded("relearn", "v").shuffle(verb)
        seeded("relearn", "q").shuffle(qa)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        items = [{"ids": [bos, *enc(r["prompt"] + r["target"])][:384], "start": 1} for r in verb[:10]]
        for r in qa[:10]:
            p = chat_ids(tok, muse.KNOW_CHAT.format(q=r["question"]))
            items.append({"ids": p + enc(r["answer"]) + [im_end], "start": len(p)})
        if extra:
            chunks = read_jsonl(data / "book/raw_forget_train.jsonl")
            seeded("relearn", "fx").shuffle(chunks)
            items += [{"ids": [bos, *r["input_ids"][:256]], "start": 1} for r in chunks[:extra]]
        return items
    chunks = read_jsonl(data / "book/raw_retain1_dev.jsonl", allow_test=True)
    seeded("relearn", "rc").shuffle(chunks)
    return [{"ids": [bos, *r["input_ids"][:256]], "start": 1} for r in chunks[:20]]


def evaluate(model, tok, a, tasks) -> dict:
    if a.domain == "calc":
        return H.eval_calc(model, tok, tasks["rows"], tasks["retain"], a.max_new, a.gen_bs)
    return H.eval_book(model, tok, Path(a.data), Path(a.attack_data))


def train_curve(a, tok, kind: str, items: list[dict], tasks) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    import math

    dev = device()
    torch.manual_seed(a.seed)  # LoRA init; batch order uses rng below
    # full fine-tune: fp32 master weights (bf16 AdamW updates at lr ~1e-5 mostly round away); LoRA: bf16 base
    base = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True,
                                                dtype=torch.float32 if a.full else torch.bfloat16).to(dev)
    base.config.use_cache = True
    if a.full:
        model = base
        for p in model.parameters():
            p.requires_grad_(True)
        # long calculus solutions (up to ~3k tokens) x batch 4 with fp32 weights + AdamW overflowed an 80 GB card;
        # checkpointing only acts in train mode, so evaluation still generates with the KV cache
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = sorted({int(s) for s in a.steps.split(",")})
    total = max(steps)

    def lr_factor(s: int) -> float:  # constant (default) or linear warmup then cosine decay to 0 over the curve
        if s < a.warmup:
            return (s + 1) / a.warmup
        if a.schedule == "cosine":
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, total - a.warmup))))
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    curve, losses = [], []
    pad = tok.pad_token_id
    rng = random.Random(a.seed)
    order, t0 = [], time.time()

    def evaluate_point(step: int) -> None:
        model.eval()
        res = evaluate(model, tok, a, tasks) | {"step": step, "train_loss": mean(losses[-10:])}
        curve.append(res)
        log(f"{a.name} {kind} step {step}: {json.dumps(res)} ({time.time() - t0:.0f}s)")

    for step in range(0, max(steps) + 1):
        if step in steps:
            evaluate_point(step)
        if step == max(steps):
            break
        model.train()
        batch = []
        while len(batch) < a.batch:
            if not order:
                order = list(range(len(items)))
                rng.shuffle(order)
            batch.append(items[order.pop()])
        L = max(len(it["ids"]) for it in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        for j, it in enumerate(batch):
            t = torch.tensor(it["ids"])
            ids[j, :len(t)], att[j, :len(t)] = t, 1
            lab[j, it["start"]:len(t)] = t[it["start"]:]
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev), use_cache=False).logits
        tgt = lab[:, 1:].to(dev)
        keep = tgt != -100  # fp32 logits only at labelled positions (same loss, far less memory)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1][keep].float(), tgt[keep])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    del model, base, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return {"kind": kind, "n_items": len(items), "curve": curve, "losses": [round(x, 4) for x in losses]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--domain", choices=("book", "calc"), required=True)
    ap.add_argument("--kinds", default="forget,related")
    ap.add_argument("--steps", default="0,10,25,50,100")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--attack-data", default="attack_data")
    ap.add_argument("--selfgen-dev", default="work/selfgen_dev.jsonl")
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--n-calc", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="full-parameter fine-tune instead of rank-16 LoRA")
    ap.add_argument("--book-extra", type=int, default=0, help="book forget curve: add N raw_forget_train chunks")
    ap.add_argument("--calc-per-kind", type=int, default=0,
                    help="calc eval: N test problems per kind (diff/integral/defint, N/3 per level) from data/calculus "
                         "instead of the 100 attack-input problems")
    ap.add_argument("--gen-bs", type=int, default=100, help="generation batch size")
    ap.add_argument("--related-n", type=int, default=20, help="calc related curve: number of retain-math solutions")
    ap.add_argument("--forget-n", type=int, default=0, help="calc forget curves: use the first N solutions (0 = the standard 7/7/6)")
    ap.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    ap.add_argument("--warmup", type=int, default=0, help="linear warmup steps")
    ap.add_argument("--web", default="s3://lexsi-hackathon/eval/fineweb_anchor.jsonl", help="kind 'web': generic text source")
    ap.add_argument("--web-n", type=int, default=1000, help="kind 'web': number of documents")
    a = ap.parse_args()

    from transformers import AutoTokenizer

    out = Path(a.out or f"results/relearn/{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tasks = {}
    if a.domain == "calc":
        rows = read_jsonl(Path(a.attack_data) / "_inputs/math_calc_test.jsonl", allow_test=True)
        seeded("abl", "test").shuffle(rows)  # same 100 problems as the abliteration attack
        rows = rows[:a.n_calc]
        if a.calc_per_kind:
            rows = []
            for k in ("diff", "integral", "defint"):
                for lvl in (1, 2, 3):
                    pool = [r for r in read_jsonl(Path(a.data) / f"calculus/level{lvl}/test.jsonl", allow_test=True) if r["kind"] == k]
                    seeded("relearn-eval", k, lvl).shuffle(pool)
                    assert len(pool) >= a.calc_per_kind // 3, (k, lvl, len(pool))
                    rows += pool[:a.calc_per_kind // 3]
        tasks = {"rows": rows, "retain": read_jsonl(Path(a.attack_data) / "_inputs/math_retain_test.jsonl", allow_test=True)[:40]}
    report = {"name": a.name, "model": a.model, "domain": a.domain, "args": vars(a), "curves": {}}
    for kind in a.kinds.split(","):
        if kind == "web":
            items = web_items(tok, a.web, a.web_n)
        elif a.domain == "calc":
            items = calc_items(tok, Path(a.selfgen_dev), Path(a.data), kind, a.related_n, a.forget_n)
        else:
            items = book_items(tok, Path(a.data), kind, a.book_extra)
        log(f"{a.name} {kind}: {len(items)} training items")
        report["curves"][kind] = train_curve(a, tok, kind, items, tasks)
        (out / "relearn.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
