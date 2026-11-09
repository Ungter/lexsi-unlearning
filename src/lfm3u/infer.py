"""Batched generation for LFM3: vLLM backend with a transformers fallback.

Prompts are passed as token ids so the chat template's BOS is not duplicated. Thinking is toggled with the
chat template's enable_thinking flag; direct mode renders an empty <think></think> block.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

MODEL_ID = "LiquidAI/LFM3-1.7B"
_STRIP = ("<|im_end|>", "<|endoftext|>", "<|startoftext|>", "<|im_start|>")


@dataclass
class Sampling:
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop: tuple[str, ...] = field(default_factory=tuple)
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def math_thinking(max_tokens: int) -> Sampling:
    """Model-card preset for math/STEM."""
    return Sampling(max_tokens, temperature=1.0, top_p=0.95, top_k=20)


def general(max_tokens: int) -> Sampling:
    """Model-card preset for chat/general."""
    return Sampling(max_tokens, temperature=0.1, top_k=50, repetition_penalty=1.1)


class Generator:
    def __init__(self, model: str = MODEL_ID, backend: str = "auto", max_model_len: int = 32768, gpu_mem: float = 0.88):
        from transformers import AutoTokenizer

        self.model = model
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.backend = None
        self.backend_error = None
        if backend in ("auto", "vllm"):
            try:
                from vllm import LLM

                self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                               seed=0, trust_remote_code=True, generation_config="vllm")
                self.backend = "vllm"
            except Exception as e:  # noqa: BLE001
                if backend == "vllm":
                    raise
                self.backend_error = f"{type(e).__name__}: {e}"
                print(f"[infer] vLLM unavailable ({self.backend_error}); falling back to transformers", flush=True)
        if self.backend is None:
            import torch
            from transformers import AutoModelForCausalLM

            self.hf = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()
            self.backend = "hf"
        eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        self.eos_ids = {i for i in eos if isinstance(i, int) and i >= 0}
        self.strip = tuple(s for s in (*_STRIP, self.tok.eos_token, self.tok.pad_token) if s)

    # ------------------------------------------------------------ prompts
    def chat_ids(self, user: str, thinking: bool = False, system: str | None = None) -> list[int]:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def raw_ids(self, text: str) -> list[int]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        bos = self.tok.bos_token_id
        return ([bos] if bos is not None else []) + ids

    def _clean(self, text: str) -> str:
        for s in self.strip:
            text = text.replace(s, "")
        return text.strip()

    # ------------------------------------------------------------ generation
    def generate(self, prompts: list[list[int]], s: Sampling) -> tuple[list[dict], dict]:
        t0 = time.time()
        outs = self._vllm(prompts, s) if self.backend == "vllm" else self._hf(prompts, s)
        dt = time.time() - t0
        ntok = sum(o["n_tokens"] for o in outs)
        print(f"[infer] {len(outs)} prompts, {ntok} tokens in {dt:.1f}s ({ntok / max(dt, 1e-9):.0f} tok/s)", flush=True)
        return outs, {"seconds": round(dt, 2), "tokens": ntok, "tok_per_s": round(ntok / max(dt, 1e-9), 1)}

    def _vllm(self, prompts, s: Sampling):
        from vllm import SamplingParams

        kw = dict(max_tokens=s.max_tokens, temperature=s.temperature, top_p=s.top_p, repetition_penalty=s.repetition_penalty,
                  seed=s.seed, skip_special_tokens=False)
        if s.top_k > 0:
            kw["top_k"] = s.top_k
        if s.stop:
            kw["stop"] = list(s.stop)
        res = self.llm.generate([{"prompt_token_ids": p} for p in prompts], SamplingParams(**kw), use_tqdm=False)
        return [{"text": self._clean(r.outputs[0].text), "n_tokens": len(r.outputs[0].token_ids),
                 "finish_reason": r.outputs[0].finish_reason} for r in res]

    def _hf(self, prompts, s: Sampling, bs: int = 32):
        import torch

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
            gk = dict(max_new_tokens=s.max_tokens, pad_token_id=pad, eos_token_id=sorted(self.eos_ids),
                      repetition_penalty=s.repetition_penalty, do_sample=s.temperature > 0)
            if s.temperature > 0:
                gk.update(temperature=s.temperature, top_p=s.top_p, top_k=s.top_k or None)
            else:
                gk.update(temperature=None, top_p=None, top_k=None)
            if s.stop:
                gk.update(stop_strings=list(s.stop), tokenizer=self.tok)
            torch.manual_seed(s.seed)
            with torch.no_grad():
                g = self.hf.generate(input_ids=inp.cuda(), attention_mask=att.cuda(), **gk)
            for j, i in enumerate(idx):
                new = g[j, L:].tolist()
                n = next((k for k, t in enumerate(new) if t in self.eos_ids or t == pad), len(new))
                text = self.tok.decode(new[:n], skip_special_tokens=False)
                for st in s.stop:
                    text = text.split(st)[0]
                out[i] = {"text": self._clean(text), "n_tokens": n, "finish_reason": "length" if n >= s.max_tokens else "stop"}
        return out
