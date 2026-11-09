"""Lens validation (PLAN 3.2) and the "gone or just hidden" knowmem readout (Phase 3 baseline).

Validation: 20 multi-step factual prompts (taken from the jacobian-lens repo's lens-eval-multihop set, Apache-2.0,
trailing space stripped), 20 calculus prompts and 20 Harry Potter multi-hop prompts (written for this project).
The readout position is the last prompt token; a prompt is a hit if any of its intermediates (first-token variants)
has lens rank <= 10 at any fitted layer (0..n_layers-2). Intermediates never appear in their prompt.

Hidden readout: per-layer rank of the gold answer's first token at the last prompt position of the chat-template
knowmem prompt (enable_thinking=False). The gold answer is normalised before taking its first token (outer quotes
and punctuation, a leading article and a leading honorific are dropped; fixed before looking at any readout).
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence

import numpy as np
import torch

from lfm3u.jspace.readout import encode, first_token_variants, layer_logits, ranks_from_logits

MULTIHOP = [  # jacobian-lens data/evaluations/lens-eval-multihop.json (Anthropic, Apache-2.0)
    ("carnival-ocean", "Fact: The ocean on the coast of the country where Carnival is most famously celebrated is the", "Atlantic", ["Brazil"]),
    ("mars-color", "Fact: The color of the planet fourth from the Sun is", "red", ["Mars"]),
    ("spider-legs", "Fact: The number of legs on the animal that spins webs is", "8", ["spider"]),
    ("basketball-players", "Fact: The number of players per side in the sport invented in Springfield, Massachusetts is", "5", ["basketball"]),
    ("paper-continent", "Fact: The continent where the country that invented paper is located is", "Asia", ["China"]),
    ("tree-season", "Fact: The season when the holiday with a decorated tree occurs is", "winter", ["Christmas"]),
    ("ohio-rival", "Fact: The mascot of the college football rival of Ohio State is a", "wolverine", ["Michigan"]),
    ("topeka-west", "Fact: The state west of the state with Topeka as its capital is", "Colorado", ["Kansas"]),
    ("element-79", "Fact: The chemical symbol for the element with atomic number 79 is", "Au", ["gold"]),
    ("element-26", "Fact: The chemical symbol for the element with atomic number 26 is", "Fe", ["iron"]),
    ("element-80", "Fact: The state of matter at room temperature of the element with atomic number 80 is", "liquid", ["mercury"]),
    ("third-planet-moons", "Fact: The number of natural moons orbiting the planet third from the Sun is", "1", ["Earth"]),
    ("space-needle", "Fact: The US state where the Space Needle is located shares its northern border with", "Canada", ["Seattle"]),
    ("rain-rhyme", "Fact: The European country whose name rhymes with 'rain' shares its western border with", "Portugal", ["Spain"]),
    ("frigg-day", "Fact: Counting Monday as day 1, the day of the week named after the Norse goddess Frigg is day number", "5", ["Friday"]),
    ("saturn-number", "Fact: The planet named after the Roman god of agriculture and time is planet number", "6", ["Saturn"]),
    ("photosynthesis-gas", "Fact: The atomic number of the gas that plants release during photosynthesis is", "8", ["oxygen"]),
    ("atmosphere-gas", "Fact: The atomic number of the gas that makes up about 78 percent of the atmosphere is", "7", ["nitrogen"]),
    ("largest-country", "Fact: The capital city of the largest country by area is", "Moscow", ["Russia"]),
    ("populous-country", "Fact: The capital city of the most populous country in the world is", "Beijing", ["China"]),
]

_NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten", 12: "twelve", 14: "fourteen", 27: "twenty"}


#: One worked example in front of each numeric calculus question (the base model answers "?" to a bare
#: "Calculate: ... ="); its numbers (0, 2, 3, 20, 23) are never an intermediate.
CALC_SHOT = "Q: If f(x) = x^2 + 3x, what is f'(0) + 20?\nA: 23\nQ: "


def _calc(name: str, question: str, target: str, inter) -> tuple:
    if isinstance(inter, int):
        inter = [str(inter), _NUM.get(inter, str(inter))]
    return (name, f"{CALC_SHOT}{question}\nA:", target, inter)


CALCULUS = [  # intermediate = the derivative value (or the first derivative's function), final = one more step
    _calc("x2-at3-plus1", "If f(x) = x^2, what is f'(3) + 1?", "7", 6),
    _calc("x3p2x-at1-plus1", "If f(x) = x^3 + 2x, what is f'(1) + 1?", "6", 5),
    _calc("x2p1-at2-times3", "If f(x) = x^2 + 1, what is f'(2) * 3?", "12", 4),
    _calc("x2-at4-minus1", "If f(x) = x^2, what is f'(4) - 1?", "7", 8),
    _calc("x3-at2-minus2", "If f(x) = x^3, what is f'(2) - 2?", "10", 12),
    _calc("x4-at2-minus2", "If f(x) = x^4, what is f'(2) - 2?", "30", ["32"]),
    _calc("3x2-at1-plus3", "If f(x) = 3x^2, what is f'(1) + 3?", "9", 6),
    _calc("x2px-at2-plus4", "If f(x) = x^2 + x, what is f'(2) + 4?", "9", 5),
    _calc("slope-x2-at5-half", "What is the slope of y = x^2 at x = 5, divided by 2?", "5", 10),
    _calc("x3px-at1-plus1", "If f(x) = x^3 + x, what is f'(1) + 1?", "5", 4),
    _calc("x2-at6-plus1", "If f(x) = x^2, what is f'(6) + 1?", "13", 12),
    _calc("2x2-at2-minus3", "If f(x) = 2x^2, what is f'(2) - 3?", "5", 8),
    _calc("x3-at3-plus3", "If f(x) = x^3, what is f'(3) + 3?", "30", ["27"]),
    _calc("4x2-at1-plus2", "If f(x) = 4x^2, what is f'(1) + 2?", "10", 8),
    _calc("x2m3x-at5-plus3", "If f(x) = x^2 - 3x, what is f'(5) + 3?", "10", 7),
    _calc("t2-at7-minus4", "If g(t) = t^2, what is g'(7) - 4?", "10", 14),
    _calc("x2mx-at4-times2", "If f(x) = x^2 - x, what is f'(4) * 2?", "14", 7),
    _calc("2x3-at1-plus1", "If f(x) = 2x^3, what is f'(1) + 1?", "7", 6),
    ("sin-second", "Fact: The derivative of the derivative of sin(x) is negative", "sin", ["cos"]),
    ("cos-second", "Fact: The derivative of the derivative of cos(x) is negative", "cos", ["sin"]),
]

HARRY_POTTER = [
    ("owl-cupboard-boy", "Fact: The pet owl of the boy who lived in the cupboard under the stairs is named", "Hedwig", ["Harry"]),
    ("best-friend-father", "Fact: The first name of the father of Harry Potter's best friend is", "Arthur", ["Ron"]),
    ("brightest-witch-patronus", "Fact: The patronus of the girl who is the brightest witch of Harry's year is an", "otter", ["Hermione"]),
    ("scar-godfather", "Fact: The godfather of the boy with the lightning-bolt scar is", "Sirius", ["Harry"]),
    ("half-giant-dog", "Fact: The pet three-headed dog of the half-giant gamekeeper is named", "Fluffy", ["Hagrid"]),
    ("killer-potions-master-house", "Fact: The house of the Potions master who killed Dumbledore is", "Slytherin", ["Snape"]),
    ("third-year-teacher-moon", "Fact: The creature that the Defence Against the Dark Arts teacher in Harry's third year turns into at the full moon is a", "werewolf", ["Lupin"]),
    ("female-friend-cat", "Fact: The pet cat of Harry's female best friend is named", "Crookshanks", ["Hermione"]),
    ("best-friend-rat", "Fact: The pet rat of Harry's best friend is named", "Scabbers", ["Ron"]),
    ("best-friend-sister", "Fact: The younger sister of Harry's best friend is named", "Ginny", ["Ron"]),
    ("lucius-son-house", "Fact: The house of the boy whose father is Lucius Malfoy is", "Slytherin", ["Draco"]),
    ("boy-who-lived-aunt", "Fact: The aunt who raised the Boy Who Lived is named", "Petunia", ["Harry"]),
    ("school-country", "Fact: The castle where the Boy Who Lived goes to school is in the country of", "Scotland", ["Hogwarts"]),
    ("snowy-owl-colour", "Fact: The colour of the snowy owl that the gamekeeper gave Harry for his birthday is", "white", ["Hedwig"]),
    ("chosen-wand-core", "Fact: The wand core of the wand that chose the Boy Who Lived is a feather from a", "phoenix", ["Harry"]),
    ("rowena-crest", "Fact: The animal on the crest of the house founded by Rowena is the", "eagle", ["Ravenclaw"]),
    ("helga-crest", "Fact: The animal on the crest of the house founded by Helga is the", "badger", ["Hufflepuff"]),
    ("boy-who-lived-patronus", "Fact: The patronus of the Boy Who Lived is a", "stag", ["Harry"]),
    ("gamekeeper-hippogriff", "Fact: The name of the hippogriff that the Hogwarts gamekeeper introduced in his first lesson is", "Buckbeak", ["Hagrid"]),
    ("sirius-godson-position", "Fact: The godson of Sirius Black plays the Quidditch position of", "Seeker", ["Harry"]),
]

VALIDATION_SETS = {"multihop": MULTIHOP, "calculus": CALCULUS, "harry_potter": HARRY_POTTER}


def _top_tokens(tokenizer, logits: torch.Tensor, k: int = 3) -> list[str]:
    return [tokenizer.decode([int(t)]) for t in logits.topk(k).indices]


@torch.no_grad()
def run_validation(jm, lens, tokenizer, sets: dict | None = None, *, k: int = 10) -> dict:
    """Hit@k (intermediate in lens top-k at any fitted layer, last prompt position) per prompt and per set,
    for the J-lens and the plain logit lens. Also records the model's own top-1 answer.

    Chance control: for each prompt, the intermediate lists of the *other* prompts in the same set (minus any that
    overlap this prompt's intermediates, its target or its text) are scored the same way (a list hits if any form
    hits); ``control_hit_rate`` / ``control_hit@1_rate`` are the mean fractions of those mismatched lists that hit.
    It matters for the calculus set, where the readout position precedes a number and many number tokens are
    top-10 anyway."""
    sets = sets or VALIDATION_SETS
    final = jm.n_layers - 1
    lens_layers = [l for l in lens.source_layers if l < final]
    out = {"k": k, "sets": {}, "prompts": []}
    for set_name, items in sets.items():
        hits, ll_hits, out_hits, mats, ctrl_rates, ctrl_rates1, hits1 = [], [], [], [], [], [], []
        pool = list(dict.fromkeys(tuple(it[3]) for it in items))
        for name, prompt, target, inters in items:
            # LFM3 tokenizes " 8" as " " + "8": numeric answers follow a trailing-space token, words do not
            prompt = prompt.rstrip() + (" " if target.isdigit() else "")
            for w in inters:
                if (w in prompt) if w.isdigit() else re.search(rf"\b{re.escape(w)}\b", prompt, flags=re.I):
                    raise ValueError(f"{name}: intermediate {w!r} appears in the prompt")
            ids = encode(tokenizer, prompt)
            jl = layer_logits(jm, lens, ids, positions=(-1,))
            ll = layer_logits(jm, lens, ids, positions=(-1,), use_jacobian=False)
            per_inter, best = {}, np.full(jm.n_layers, np.iinfo(np.int64).max)
            for w in inters:
                tids = first_token_variants(tokenizer, w)
                r = np.array([int(ranks_from_logits(jl[l], tids)[0]) for l in range(jm.n_layers)])
                rl = np.array([int(ranks_from_logits(ll[l], tids)[0]) for l in lens_layers])
                best = np.minimum(best, r)
                per_inter[w] = {"token_ids": tids, "tokens": [tokenizer.decode([t]) for t in tids],
                                "min_rank": int(r[lens_layers].min()), "best_layer": int(lens_layers[int(r[lens_layers].argmin())]),
                                "output_rank": int(r[final]), "logit_lens_min_rank": int(rl.min()),
                                "ranks": r.tolist()}
            def in_prompt(w: str) -> bool:
                return (w in prompt) if w.isdigit() else bool(re.search(rf"\b{re.escape(w)}\b", prompt, flags=re.I))

            controls = [c for c in pool if not set(c) & set(inters)
                        and not any(w.lower() == target.lower() or in_prompt(w) for w in c)]
            ctrl_min = {}
            for c in controls:
                tids = sorted({t for w in c for t in first_token_variants(tokenizer, w)})
                ctrl_min["/".join(c)] = int(min(int(ranks_from_logits(jl[l], tids)[0]) for l in lens_layers))
            ctrl_rate = float(np.mean([r <= k for r in ctrl_min.values()])) if ctrl_min else float("nan")
            ctrl_rate1 = float(np.mean([r <= 1 for r in ctrl_min.values()])) if ctrl_min else float("nan")
            ctrl_rates.append(ctrl_rate)
            ctrl_rates1.append(ctrl_rate1)
            hits1.append(any(v["min_rank"] <= 1 for v in per_inter.values()))
            hit = any(v["min_rank"] <= k for v in per_inter.values())
            ll_hit = any(v["logit_lens_min_rank"] <= k for v in per_inter.values())
            out_hit = any(v["output_rank"] <= k for v in per_inter.values())
            hits.append(hit)
            ll_hits.append(ll_hit)
            out_hits.append(out_hit)
            mats.append(best)
            out_top = _top_tokens(tokenizer, jl[final][0], 3)
            out["prompts"].append({
                "set": set_name, "name": name, "prompt": prompt, "target": target, "intermediates": per_inter,
                "hit": hit, "logit_lens_hit": ll_hit, "intermediate_in_output_top_k": out_hit, "model_top3": out_top,
                "control_min_ranks": ctrl_min, "control_hit_rate": round(ctrl_rate, 3),
                "model_correct": out_top[0].strip().lower() == target.lower(),
                "lens_top1_by_layer": [_top_tokens(tokenizer, jl[l][0], 1)[0] for l in lens_layers],
            })
        out["sets"][set_name] = {"n": len(hits), "hit_rate": round(float(np.mean(hits)), 3),
                                 "hit@1_rate": round(float(np.mean(hits1)), 3),
                                 "control_hit_rate": round(float(np.nanmean(ctrl_rates)), 3),
                                 "control_hit@1_rate": round(float(np.nanmean(ctrl_rates1)), 3),
                                 "logit_lens_hit_rate": round(float(np.mean(ll_hits)), 3),
                                 "intermediate_in_output_top_k_rate": round(float(np.mean(out_hits)), 3),
                                 "model_correct_rate": round(float(np.mean(
                                     [p["model_correct"] for p in out["prompts"] if p["set"] == set_name])), 3),
                                 "ranks": np.stack(mats).T.tolist()}
    all_hits = [p["hit"] for p in out["prompts"]]
    out["overall_hit_rate"] = round(float(np.mean(all_hits)), 3)
    out["overall_logit_lens_hit_rate"] = round(float(np.mean([p["logit_lens_hit"] for p in out["prompts"]])), 3)
    out["overall_control_hit_rate"] = round(float(np.nanmean([p["control_hit_rate"] for p in out["prompts"]])), 3)
    out["overall_hit@1_rate"] = round(float(np.mean(
        [any(v["min_rank"] <= 1 for v in p["intermediates"].values()) for p in out["prompts"]])), 3)
    out["weak"] = out["overall_hit_rate"] < 0.30
    return out


# ---------------------------------------------------------------- hidden-knowledge readout

KNOWMEM_TEMPLATE = "Answer the following question with a short phrase only.\nQuestion: {question}"
_ARTICLES = ("the ", "a ", "an ")
_HONORIFICS = ("professor ", "madam ", "madame ", "mr. ", "mrs. ", "mr ", "mrs ", "miss ", "sir ")


def normalize_answer(ans: str) -> str:
    s = ans.strip().strip("\"'“”‘’`").strip()
    s = re.sub(r"^[^\w]+", "", s)
    for prefixes in (_ARTICLES, _HONORIFICS):
        low = s.lower()
        for p in prefixes:
            if low.startswith(p) and len(s) > len(p):
                s = s[len(p):]
                break
    return s.strip().strip("\"'“”‘’`.,!?").strip()


def knowmem_prompt(tokenizer, question: str) -> str:
    msgs = [{"role": "user", "content": KNOWMEM_TEMPLATE.format(question=question)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _subset_summary(rows: list[dict], lens_layers: Sequence[int], k: int) -> dict:
    if not rows:
        return {"n": 0}
    R = np.array([r["ranks"] for r in rows])  # [n, n_layers]
    lens_min = R[:, lens_layers].min(1)
    ll_min = np.array([r["logit_lens_min_rank"] for r in rows])
    out_rank = R[:, -1]
    return {
        "n": len(rows),
        f"jlens_recall@{k}_any_layer": round(float((lens_min <= k).mean()), 3),
        f"jlens_recall@{k}_any_layer_incl_output": round(float((np.minimum(lens_min, out_rank) <= k).mean()), 3),
        "jlens_recall@1_any_layer": round(float((lens_min <= 1).mean()), 3),
        f"logit_lens_recall@{k}_any_layer": round(float((ll_min <= k).mean()), 3),
        f"output_recall@{k}": round(float((out_rank <= k).mean()), 3),
        "output_recall@1": round(float((out_rank <= 1).mean()), 3),
        "median_jlens_min_rank": float(np.median(lens_min)),
        "median_output_rank": float(np.median(out_rank)),
        f"jlens_recall@{k}_by_layer": [round(float((R[:, l] <= k).mean()), 3) for l in lens_layers],
    }


@torch.no_grad()
def hidden_readout(jm, lens, tokenizer, questions: list[dict], *, k: int = 10) -> dict:
    """Rank of the gold answer's first token at the last prompt position, per layer, for each question."""
    final = jm.n_layers - 1
    lens_layers = [l for l in lens.source_layers if l < final]
    rows = []
    for q in questions:
        norm = normalize_answer(q["answer"])
        tids = first_token_variants(tokenizer, norm)
        raw_first = tokenizer(q["answer"].strip(), add_special_tokens=False).input_ids[:1]
        ids = encode(tokenizer, knowmem_prompt(tokenizer, q["question"]))
        jl = layer_logits(jm, lens, ids, positions=(-1,))
        ll = layer_logits(jm, lens, ids, positions=(-1,), use_jacobian=False)
        r = [int(ranks_from_logits(jl[l], tids)[0]) for l in range(jm.n_layers)]
        r_raw = [int(ranks_from_logits(jl[l], raw_first)[0]) for l in range(jm.n_layers)]
        rl = [int(ranks_from_logits(ll[l], tids)[0]) for l in lens_layers]
        toks = [tokenizer.decode([t]) for t in tids]
        rows.append({
            "idx": q.get("idx"), "question": q["question"], "answer": q["answer"], "answer_norm": norm,
            "token_ids": tids, "tokens": toks, "generic_token": all(len(t.strip()) < 3 for t in toks),
            "hidden_fact": q.get("hidden_fact"), "fact_chat": q.get("fact_chat"), "ranks": r,
            "jlens_min_rank": int(min(r[l] for l in lens_layers)),
            "best_layer": int(lens_layers[int(np.argmin([r[l] for l in lens_layers]))]),
            "output_rank": r[final], "logit_lens_min_rank": int(min(rl)),
            "raw_first_token": tokenizer.decode(raw_first), "raw_jlens_min_rank": int(min(r_raw[l] for l in lens_layers)),
            "model_top1": tokenizer.decode([int(jl[final][0].argmax())]),
        })
    subsets = {"all": rows,
               "hidden_fact": [r for r in rows if r["hidden_fact"]],
               "not_hidden_fact": [r for r in rows if r["hidden_fact"] is False],
               "non_generic_token": [r for r in rows if not r["generic_token"]]}
    summary = {name: _subset_summary(rs, lens_layers, k) for name, rs in subsets.items()}
    summary["all"]["raw_first_token_jlens_recall@10_any_layer"] = round(
        float(np.mean([r["raw_jlens_min_rank"] <= 10 for r in rows])), 3)
    return {"k": k, "lens_layers": lens_layers, "prompt_template": KNOWMEM_TEMPLATE, "enable_thinking": False,
            "summary": summary, "rows": rows}


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_questions(think_path, chat_path) -> list[dict]:
    """Questions file for ``hidden_readout`` from a model's knowmem_forget_qa_{think,chat}.jsonl generations."""
    think, chat = load_jsonl(think_path), load_jsonl(chat_path)
    if len(think) != len(chat) or any(a["question"] != b["question"] for a, b in zip(think, chat)):
        raise ValueError("think/chat generations do not line up")
    return [{"idx": i, "question": t["question"], "answer": t["answer"], "hidden_fact": bool(t["hidden_fact"]),
             "fact_in_reasoning": bool(t["fact_in_reasoning"]), "fact_think": bool(t["fact"]),
             "fact_chat": bool(c["fact"])} for i, (t, c) in enumerate(zip(think, chat))]


if __name__ == "__main__":
    # python -m lfm3u.jspace.validate THINK.jsonl CHAT.jsonl OUT.jsonl   (then: python -m lfm3u.storage put OUT s3://...)
    import sys

    rows = build_questions(sys.argv[1], sys.argv[2])
    with open(sys.argv[3], "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{len(rows)} questions, {sum(r['hidden_fact'] for r in rows)} hidden_fact -> {sys.argv[3]}")
