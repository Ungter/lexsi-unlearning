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


def calc_items(tok, selfgen: Path, data: Path, kind: str) -> list[dict]:
    rows = [json.loads(line) for line in selfgen.read_text(encoding="utf-8").splitlines() if line.strip()]
    want = "calculus" if kind == "forget" else "retain_math"
    rows = [r for r in rows if r["source"] == want and r["mode"] == "direct"]
    seeded("relearn", kind).shuffle(rows)
    if kind == "forget":
        picked = []
        for lvl, k in ((1, 7), (2, 7), (3, 6)):
            picked += [r for r in rows if r["level"] == lvl][:k]
    else:
        picked = rows[:20]
    return [{"ids": r["prompt_ids"] + r["completion_ids"], "start": len(r["prompt_ids"])} for r in picked]


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
        return H.eval_calc(model, tok, tasks["rows"], tasks["retain"], a.max_new)
    return H.eval_book(model, tok, Path(a.data), Path(a.attack_data))


def train_curve(a, tok, kind: str, items: list[dict], tasks) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    dev = device()
    # full fine-tune: fp32 master weights (bf16 AdamW updates at lr ~1e-5 mostly round away); LoRA: bf16 base
    base = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True,
                                                dtype=torch.float32 if a.full else torch.bfloat16).to(dev)
    base.config.use_cache = True
    if a.full:
        model = base
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = sorted({int(s) for s in a.steps.split(",")})
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
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)), lab[:, 1:].to(dev).reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
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
    a = ap.parse_args()

    from transformers import AutoTokenizer

    out = Path(a.out or f"results/relearn/{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tasks = {}
    if a.domain == "calc":
        rows = read_jsonl(Path(a.attack_data) / "_inputs/math_calc_test.jsonl", allow_test=True)
        seeded("abl", "test").shuffle(rows)  # same 100 problems as the abliteration attack
        tasks = {"rows": rows[:a.n_calc], "retain": read_jsonl(Path(a.attack_data) / "_inputs/math_retain_test.jsonl", allow_test=True)[:40]}
    report = {"name": a.name, "model": a.model, "domain": a.domain, "args": vars(a), "curves": {}}
    for kind in a.kinds.split(","):
        items = (calc_items(tok, Path(a.selfgen_dev), Path(a.data), kind) if a.domain == "calc"
                 else book_items(tok, Path(a.data), kind, a.book_extra))
        log(f"{a.name} {kind}: {len(items)} training items")
        report["curves"][kind] = train_curve(a, tok, kind, items, tasks)
        (out / "relearn.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
