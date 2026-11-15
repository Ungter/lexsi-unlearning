"""Print the key baseline metrics of results/<name>/baseline.json side by side (aya branch helper).

  python scripts/aya_table.py base target_A15 retain_A15 ...
"""
import json
import sys

COLS = [("calculus_direct", "overall", "correct", "calc_probe"), ("math_gen", "algebra", "correct", "algebra"),
        ("math_gen", "calculus", "correct", "mg_calc"), ("knowmem_forget_qa_icl", None, "rougeL_recall", "km_forget"),
        ("knowmem_forget_qa_icl", None, "fact", "km_f_fact"), ("knowmem_retain_qa_icl", None, "rougeL_recall", "km_retain"),
        ("verbmem_forget_raw", None, "rougeL_f1", "verbmem"), ("mmlu", None, "acc", "mmlu"), ("ppl_fineweb", None, "ppl", "ppl"),
        ("ifeval_direct", None, "prompt_strict", "ifeval")]

print("model".ljust(14) + "".join(c[3].rjust(11) for c in COLS))
for m in sys.argv[1:]:
    try:
        r = json.load(open(f"results/{m}/baseline.json", encoding="utf-8"))["results"]
    except Exception as e:  # noqa: BLE001
        print(m, "ERR", e)
        continue
    row = []
    for task, sub, met, _ in COLS:
        v = r.get(task, {})
        if sub:
            v = v.get(sub, {})
        x = v.get(met) if isinstance(v, dict) else None
        row.append(("-" if x is None else f"{x:.3f}").rjust(11))
    print(m.ljust(14) + "".join(row))
