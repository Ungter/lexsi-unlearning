"""Batched generation for Tiny Aya Global: vLLM backend with a transformers fallback.

Prompts are passed as token ids so the chat template's BOS is not duplicated. Tiny Aya has no thinking mode, so the
`thinking` arguments kept from the LFM3 code are accepted and ignored (lfm3u.arch). Generation stops at
<|END_RESPONSE|> / <|END_OF_TURN_TOKEN|>; context is 8,192 tokens (prompt + output).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from lfm3u import arch

MODEL_ID = arch.MODEL_ID


@dataclass
class Sampling:
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: tuple[str, ...] = field(default_factory=tuple)
    seed: int | None = 0  # None = unseeded (vLLM draws per-request randomness; needed for best-of-N)
    allowed_token_ids: tuple[int, ...] = field(default_factory=tuple)  # restrict sampling (e.g. MMLU letters)

    def as_dict(self) -> dict:
        return asdict(self)


def math_thinking(max_tokens: int) -> Sampling:
    """Sampled preset for math (kept from LFM3's thinking preset; Tiny Aya has no thinking mode)."""
    return Sampling(max_tokens, temperature=1.0, top_p=0.95, top_k=20)


def general(max_tokens: int) -> Sampling:
    """Model-card preset for chat/general (Tiny Aya card: temperature 0.1, top_p 0.95)."""
    return Sampling(max_tokens, temperature=0.1, top_p=0.95)


class Generator:
    def __init__(self, model: str = MODEL_ID, backend: str = "auto", max_model_len: int = arch.MAX_CONTEXT,
                 gpu_mem: float = 0.88):
        from transformers import AutoTokenizer

        max_model_len = min(max_model_len, arch.MAX_CONTEXT)
        self.model = model
        self.max_model_len = max_model_len
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=arch.TRUST_REMOTE_CODE)
        self.backend = None
        self.backend_error = None
        if backend in ("auto", "vllm"):
            try:
                from vllm import LLM

                self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                               seed=0, trust_remote_code=arch.TRUST_REMOTE_CODE, generation_config="vllm")
                self.backend = "vllm"
            except Exception as e:  # noqa: BLE001
                if backend == "vllm":
                    raise
                self.backend_error = f"{type(e).__name__}: {e}"
                print(f"[infer] vLLM unavailable ({self.backend_error}); falling back to transformers", flush=True)
        if self.backend is None:
            import torch
            from transformers import AutoModelForCausalLM

            self.hf = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=arch.TRUST_REMOTE_CODE, dtype=torch.bfloat16,
                                                           device_map="cuda").eval()
            self.backend = "hf"
        self.eos_ids = set(arch.eos_ids(self.tok))
        self.strip = tuple(s for s in (*arch.SPECIAL_STRIP, self.tok.eos_token, self.tok.pad_token) if s)

    # ------------------------------------------------------------ prompts
    def chat_ids(self, user: str, thinking: bool = False, system: str | None = None, prefill: str = "") -> list[int]:
        return arch.chat_ids(self.tok, user, thinking=thinking, system=system, prefill=prefill)

    def raw_ids(self, text: str) -> list[int]:
        return arch.raw_ids(self.tok, text)

    def _clean(self, text: str) -> str:
        for s in self.strip:
            text = text.replace(s, "")
        return text.strip()

    def _budget(self, prompts: list[list[int]], max_tokens: int) -> list[int]:
        """Per-prompt max new tokens so prompt + output fits the 8K context."""
        return [max(1, min(max_tokens, self.max_model_len - len(p))) for p in prompts]

    # ------------------------------------------------------------ generation
    def generate(self, prompts: list[list[int]], s: Sampling) -> tuple[list[dict], dict]:
        t0 = time.time()
        too_long = sum(len(p) >= self.max_model_len for p in prompts)
        if too_long:
            print(f"[infer] {too_long} prompts >= {self.max_model_len} tokens; truncating them from the left", flush=True)
            prompts = [p[-(self.max_model_len - 1):] if len(p) >= self.max_model_len else p for p in prompts]
        outs = self._vllm(prompts, s) if self.backend == "vllm" else self._hf(prompts, s)
        dt = time.time() - t0
        ntok = sum(o["n_tokens"] for o in outs)
        print(f"[infer] {len(outs)} prompts, {ntok} tokens in {dt:.1f}s ({ntok / max(dt, 1e-9):.0f} tok/s)", flush=True)
        return outs, {"seconds": round(dt, 2), "tokens": ntok, "tok_per_s": round(ntok / max(dt, 1e-9), 1)}

    def _vllm(self, prompts, s: Sampling):
        from vllm import SamplingParams

        def params(max_tokens: int):
            kw = dict(max_tokens=max_tokens, temperature=s.temperature, top_p=s.top_p, repetition_penalty=s.repetition_penalty,
                      seed=s.seed, skip_special_tokens=False, stop_token_ids=sorted(self.eos_ids))
            if s.top_k > 0:
                kw["top_k"] = s.top_k
            if s.stop:
                kw["stop"] = list(s.stop)
            if s.allowed_token_ids:
                kw["allowed_token_ids"] = list(s.allowed_token_ids)
            return SamplingParams(**kw)

        budgets = self._budget(prompts, s.max_tokens)
        sp = params(s.max_tokens) if len(set(budgets)) == 1 and budgets[0] == s.max_tokens else [params(b) for b in budgets]
        res = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
        return [{"text": self._clean(r.outputs[0].text), "n_tokens": len(r.outputs[0].token_ids),
                 "token_ids": list(r.outputs[0].token_ids), "finish_reason": r.outputs[0].finish_reason} for r in res]

    def prompt_logprobs(self, prompts: list[list[int]]) -> list[list[float]]:
        """Log-probability of every prompt token after the first (teacher-forced), for perplexity."""
        t0 = time.time()
        prompts = [p[:self.max_model_len - 1] for p in prompts]
        if self.backend == "vllm":
            from vllm import SamplingParams

            res = self.llm.generate([{"prompt_token_ids": p} for p in prompts],
                                    SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0), use_tqdm=False)
            out = [[lp[t].logprob for lp, t in zip(r.prompt_logprobs[1:], p[1:])] for r, p in zip(res, prompts)]
        else:
            import torch

            out = []
            with torch.no_grad():
                for p in prompts:
                    ids = torch.tensor([p], device="cuda")
                    logits = self.hf(input_ids=ids).logits[0, :-1].float()
                    out.append(torch.log_softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0].tolist())
        print(f"[infer] prompt logprobs for {len(prompts)} prompts in {time.time() - t0:.1f}s", flush=True)
        return out

    def _hf(self, prompts, s: Sampling, bs: int = 16):
        import torch

        if s.allowed_token_ids and s.max_tokens == 1:  # one restricted step: argmax over the allowed ids
            allowed = torch.tensor(s.allowed_token_ids, device="cuda")
            res = []
            with torch.no_grad():
                for p in prompts:
                    logits = self.hf(input_ids=torch.tensor([p], device="cuda")).logits[0, -1]
                    tid = int(allowed[logits[allowed].argmax()])
                    res.append({"text": self.tok.decode([tid]), "n_tokens": 1, "token_ids": [tid], "finish_reason": "length"})
            return res

        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        out: list[dict | None] = [None] * len(prompts)
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            L = max(len(prompts[i]) for i in idx)
            inp = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                inp[j, L - len(prompts[i]):] = torch.tensor(prompts[i])
                att[j, L - len(prompts[i]):] = 1
            max_new = max(1, min(s.max_tokens, self.max_model_len - L))
            gk = dict(max_new_tokens=max_new, pad_token_id=pad, eos_token_id=sorted(self.eos_ids),
                      repetition_penalty=s.repetition_penalty, do_sample=s.temperature > 0)
            if s.temperature > 0:
                gk.update(temperature=s.temperature, top_p=s.top_p, top_k=s.top_k or None)
            else:
                gk.update(temperature=None, top_p=None, top_k=None)
            if s.stop:
                gk.update(stop_strings=list(s.stop), tokenizer=self.tok)
            torch.manual_seed(s.seed or 0)
            with torch.no_grad():
                g = self.hf.generate(input_ids=inp.cuda(), attention_mask=att.cuda(), **gk)
            for j, i in enumerate(idx):
                new = g[j, L:].tolist()
                n = next((k for k, t in enumerate(new) if t in self.eos_ids or t == pad), len(new))
                text = self.tok.decode(new[:n], skip_special_tokens=False)
                for st in s.stop:
                    text = text.split(st)[0]
                out[i] = {"text": self._clean(text), "n_tokens": n, "token_ids": new[:n + 1] if n < len(new) else new[:n],
                          "finish_reason": "length" if n >= max_new else "stop"}
        return out
