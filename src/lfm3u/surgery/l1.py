"""L1 (PLAN 7.4): an agent-chosen unit list in the format scripts/07_unlearn.py --mask consumes.

--mask takes a localizer scores.pt and runs lfm3u.unlearn.localize.select on it (greedy by forget-specific score per
parameter, positive scores only, up to --mask-frac of all parameters). `write_mask` therefore writes a scores.pt with
the same layout in which the chosen units have a positive forget score and every other unit has zero, and retain
scores are zero: select() then returns exactly the chosen units (verified here), and --mask-random draws its control
from the other units of the same tensors, as for the localizer mask.

  python scripts/07_unlearn.py ... --mask results/surgery/l1/<target>/scores.pt --mask-frac 0.005
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def write_mask(target: str, units: dict[str, list[int]], out_dir: str | Path, *, note: str, frac: float = 0.005,
               source_scores: str | Path | None = None, extra: dict | None = None) -> dict:
    """units = {writer module name (as in the localizer, without ".weight"): [unit indices]}."""
    import torch

    from lfm3u.unlearn import localize

    src = Path(source_scores or ROOT / f"results/localize/{target}/scores.pt")
    sc = torch.load(src, weights_only=False)
    hd = sc["meta"]["head_dim"]
    unknown = sorted(set(units) - set(sc["modules"]))
    if unknown:
        raise KeyError(f"not writer modules: {unknown[:3]}")
    modules = {}
    for n, e in sc["modules"].items():
        w = localize.unit_width(e, hd)
        f = torch.zeros_like(e["forget"])
        for i in units.get(n, []):
            f[i * w:(i + 1) * w] = 1.0 / w          # unit score 1 for every chosen unit
        modules[n] = {"layer": e["layer"], "kind": e["kind"], "d_out": e["d_out"], "forget": f, "retain": torch.zeros_like(e["retain"])}
    new = {"meta": dict(sc["meta"]) | {"agent_chosen": True, "note": note, "derived_from": str(src)}, "modules": modules}
    sel = localize.select(new, frac)
    want = {n: sorted(set(v)) for n, v in units.items() if v}
    if sel["units"] != {n: want[n] for n in sel["units"]} or set(sel["units"]) != set(want):
        cost = sum(modules[n]["d_out"] * localize.unit_width(modules[n], hd) * len(v) for n, v in want.items())
        raise ValueError(f"select() does not reproduce the list: cost {cost} vs budget {sel['budget_params']} at frac {frac}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(new, out / "scores.pt")
    summary = {k: v for k, v in sel.items() if k != "units"} | {"target": target, "note": note, "scores": str((out / "scores.pt").relative_to(ROOT).as_posix()),
                                                                 "mask_frac_to_pass": frac} | (extra or {})
    (out / "units.json").write_text(json.dumps({"summary": summary, "units": sel["units"]}, indent=1), encoding="utf-8")
    return summary
