"""Guard for weight surgery (PLAN 7.2, with the Nov-10 corrections).

Rules
  protected   edits to embed_tokens / lm_head (tied) and embedding_norm (the final norm) are rejected
  edit size   a low-rank edit costs rank * (rows + cols); a sparse edit costs its changed elements; the sum over the
              active edits must stay <= max_frac (0.1%) of all parameters. (PLAN counted changed elements only, but a
              rank-one edit of one w2 touches all 15.7M of its elements.) save_patch reports both this cost and the
              number of elements that differ in the bf16 checkpoint.
  forwards    run_forward / dev_score are charged one call each (<= 256 prompts for run_forward); the session budget is
              10, 25 or 50 and is fixed when the run is created
  closing     save_patch closes the run: no more edits or budgeted calls (the final dev eval of the saved checkpoint is
              the audit, not feedback)

State lives in <run_dir>/state.json and survives across processes; deltas are files in <run_dir>/deltas/ and are
re-applied on load. edits.jsonl is rewritten from the state (one line per edit: tensor, operation, cost, rationale,
dev score before/after); events.jsonl is the append-only trail of every guarded action.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

PROTECTED = ("embed_tokens", "lm_head", "embedding_norm")
BUDGETS = (10, 25, 50)
MAX_PROMPTS = 256
MAX_FRAC = 0.001


class GuardError(RuntimeError):
    pass


class BudgetExceeded(GuardError):
    pass


def edit_cost(kind: str, shape: tuple, *, rank: int = 0, n_cols: int = 0, n_elements: int = 0) -> int:
    """Guard cost of one edit. lowrank: rank * (rows + cols). cols: rows * changed columns. flat: changed elements."""
    if kind == "lowrank":
        if len(shape) != 2 or rank < 1:
            raise GuardError("a low-rank edit needs a 2-D tensor and rank >= 1")
        return rank * (shape[0] + shape[1])
    if kind == "cols":
        return shape[0] * n_cols
    if kind == "flat":
        return n_elements
    raise GuardError(f"unknown edit kind {kind}")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Guard:
    def __init__(self, run_dir: str | Path, *, n_params: int | None = None, call_budget: int | None = None,
                 target: str | None = None, meta: dict | None = None, max_frac: float = MAX_FRAC):
        self.dir = Path(run_dir)
        self.path = self.dir / "state.json"
        if self.path.is_file():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if call_budget is not None and call_budget != self.state["call_budget"]:
                raise GuardError(f"run already has a forward budget of {self.state['call_budget']}; it cannot be changed")
            if n_params is not None and n_params != self.state["n_params"]:
                raise GuardError("parameter count differs from the run's model")
            return
        if n_params is None or call_budget is None:
            raise GuardError(f"no run at {self.dir}; create it with a target and a forward budget")
        if call_budget not in BUDGETS:
            raise GuardError(f"forward budget must be one of {BUDGETS}")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "deltas").mkdir(exist_ok=True)
        self.state = {"run": self.dir.name, "target": target, "created": _now(),
                      "n_params": int(n_params), "max_frac": max_frac, "edit_budget": int(max_frac * n_params),
                      "call_budget": int(call_budget), "calls_used": 0, "calls": [], "edits": [], "scores": [],
                      "closed": False, "meta": meta or {}}
        self._save()
        self._event({"event": "create", "target": target, "call_budget": call_budget, "edit_budget": self.state["edit_budget"]})

    # ------------------------------------------------------------ persistence
    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        tmp.replace(self.path)
        with (self.dir / "edits.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for e in self.state["edits"]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def _event(self, rec: dict) -> None:
        rec = {"time": _now(), **rec}
        with (self.dir / "events.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _open(self) -> None:
        if self.state["closed"]:
            raise GuardError("run is closed (a patch was saved); start a new run to edit further")

    # ------------------------------------------------------------ forward budget
    @property
    def calls_left(self) -> int:
        return self.state["call_budget"] - self.state["calls_used"]

    def charge(self, kind: str, n_prompts: int, note: str = "") -> int:
        """Charge one budgeted forward call (run_forward or dev_score). Raises before any model work is done."""
        self._open()
        if kind == "run_forward" and n_prompts > MAX_PROMPTS:
            raise GuardError(f"run_forward takes at most {MAX_PROMPTS} prompts per call (got {n_prompts})")
        if self.calls_left <= 0:
            self._event({"event": "rejected_call", "kind": kind, "reason": "forward budget used up"})
            raise BudgetExceeded(f"forward budget of {self.state['call_budget']} calls is used up")
        self.state["calls_used"] += 1
        rec = {"n": self.state["calls_used"], "time": _now(), "kind": kind, "n_prompts": n_prompts, "note": note,
               "active_edits": [e["id"] for e in self.active()]}
        self.state["calls"].append(rec)
        self._save()
        self._event({"event": "call", **rec})
        return self.state["calls_used"]

    # ------------------------------------------------------------ edits
    def active(self) -> list[dict]:
        return [e for e in self.state["edits"] if e["status"] == "active"]

    def used(self) -> int:
        return sum(e["cost"] for e in self.active())

    def check_param(self, name: str) -> None:
        if any(p in name for p in PROTECTED):
            self._event({"event": "rejected_edit", "param": name, "reason": "protected tensor"})
            raise GuardError(f"{name} is protected (tied embedding / lm_head / embedding_norm)")

    def admit(self, *, param: str, shape: tuple, kind: str, cost: int, op: str, args: dict, rationale: str) -> dict:
        """Register an edit if it passes the rules; returns its record (the caller then writes the delta file)."""
        self._open()
        self.check_param(param)
        if not rationale or len(rationale.strip()) < 10:
            raise GuardError("every edit needs a rationale (what was tested and why this edit)")
        budget = self.state["edit_budget"]
        if self.used() + cost > budget:
            self._event({"event": "rejected_edit", "param": param, "op": op, "cost": cost, "used": self.used(),
                         "reason": "edit budget"})
            raise BudgetExceeded(f"edit costs {cost}, {self.used()} of {budget} already used (0.1% of parameters)")
        eid = f"e{len(self.state['edits']) + 1:03d}"
        last = self.state["scores"][-1] if self.state["scores"] else None
        rec = {"id": eid, "status": "active", "time": _now(), "param": param,
               "shape": list(shape), "op": op, "kind": kind, "cost": int(cost), "args": args, "rationale": rationale.strip(),
               "file": f"deltas/{eid}.pt", "dev_before": last, "dev_after": None,
               "calls_used_at_edit": self.state["calls_used"]}
        self.state["edits"].append(rec)
        self._save()
        self._event({"event": "edit", "id": eid, "param": param, "op": op, "cost": cost, "used": self.used(), "budget": budget})
        return rec

    def drop(self, edit_id: str) -> None:
        """Remove a just-admitted record whose delta could not be written."""
        self.state["edits"] = [e for e in self.state["edits"] if e["id"] != edit_id]
        self._save()
        self._event({"event": "drop", "id": edit_id})

    def revert(self, edit_id: str, reason: str = "") -> dict:
        self._open()
        for e in self.state["edits"]:
            if e["id"] == edit_id:
                if e["status"] != "active":
                    raise GuardError(f"{edit_id} is already {e['status']}")
                e["status"] = "reverted"
                e["reverted"] = {"time": _now(), "reason": reason}
                self._save()
                self._event({"event": "revert", "id": edit_id, "reason": reason})
                return e
        raise GuardError(f"no edit {edit_id}")

    def record_score(self, score: dict) -> None:
        """Store a dev score; it becomes dev_after of every active edit made since the previous score."""
        score = {"call": self.state["calls_used"], "active_edits": [e["id"] for e in self.active()], **score}
        for e in self.state["edits"]:
            if e["dev_after"] is None and e["status"] == "active":
                e["dev_after"] = score
        self.state["scores"].append(score)
        self._save()
        self._event({"event": "score", "call": score["call"], "active_edits": score["active_edits"]})

    def close(self, patch: dict) -> None:
        self.state["closed"] = True
        self.state["patch"] = patch
        self._save()
        self._event({"event": "close", "patch": patch.get("dir")})

    def summary(self) -> dict:
        s = self.state
        return {"run": s["run"], "target": s["target"], "calls_used": s["calls_used"], "call_budget": s["call_budget"],
                "edit_cost_used": self.used(), "edit_budget": s["edit_budget"], "closed": s["closed"],
                "active_edits": [{k: e[k] for k in ("id", "param", "op", "cost")} for e in self.active()],
                "reverted": [e["id"] for e in s["edits"] if e["status"] == "reverted"]}
