"""Model-family facts for the aya branch: CohereLabs/tiny-aya-global (Cohere2ForCausalLM), kept in one place.

Tiny Aya (arXiv 2603.11510, Table 2): 3.35B parameters, 36 layers, d_model 2048, SwiGLU FFN 11008, 16 query / 4 KV heads
(head_dim 128), 262k vocabulary with tied embeddings, LayerNorm without bias, no biases anywhere, 8K context.
Blocks are PARALLEL: h = x + attn(ln(x)) + mlp(ln(x)) with one input_layernorm per layer, so there is no mid-block
residual and no separate FFN norm. Attention interleaves sliding-window (4096, RoPE) and full (NoPE) layers 3:1.
There is no thinking mode: the chat template has no <think> block or enable_thinking flag; `thinking` arguments are
accepted everywhere for API compatibility with the LFM3 code and ignored.

LFM3 -> Tiny Aya name map: feed_forward.w1 -> mlp.gate_proj, feed_forward.w3 -> mlp.up_proj, feed_forward.w2 ->
mlp.down_proj, self_attn.out_proj -> self_attn.o_proj, operator_norm/ffn_norm -> input_layernorm (shared),
embedding_norm -> model.norm, conv.* -> (none).
"""
from __future__ import annotations

MODEL_ID = "CohereLabs/tiny-aya-global"
FAMILY = "tiny-aya"
N_LAYERS = 36
D_MODEL = 2048
D_FFN = 11008
N_HEADS = 16
N_KV_HEADS = 4
HEAD_DIM = 128
MAX_CONTEXT = 8192
TRUST_REMOTE_CODE = False  # Cohere2 is native in transformers and vLLM

# parameter-name fragments inside model.model.layers.{i}.
Q_PROJ, K_PROJ, V_PROJ, O_PROJ = "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"
GATE_PROJ, UP_PROJ, DOWN_PROJ = "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
ATTN_WRITER, MLP_WRITER = O_PROJ, DOWN_PROJ  # the two matrices that write into the residual stream
WRITERS = (ATTN_WRITER, MLP_WRITER)
READERS = (Q_PROJ, K_PROJ, V_PROJ, GATE_PROJ, UP_PROJ)  # read ln(x); all share input_layernorm
LAYER_NORM = "input_layernorm"
FINAL_NORM = "model.norm"
EMBED = "model.embed_tokens"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Math unlearning target. Base Tiny Aya failed the Phase-1 gate on the calculus generator (math_gen calculus L1 0.54,
# L2 0.21; results/base/baseline.json, run base-20261114-031109-gate-dd79) and passed on algebra (L1 0.93, L2 0.44), so
# the math target is ALGEBRA on this branch. Code keeps the LFM3 "calc" names for domains/tasks/files (calc, calc_*,
# --domain calc) but they now mean MATH_TARGET; MATH_RETAIN are the other math_gen topics.
MATH_TOPICS = ("arithmetic", "algebra", "precalculus", "calculus")
MATH_TARGET = "algebra"
MATH_RETAIN = tuple(t for t in MATH_TOPICS if t != MATH_TARGET)

END_RESPONSE, END_TURN = "<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"
STOP_STRINGS = (END_RESPONSE, END_TURN)
SPECIAL_STRIP = ("<|START_RESPONSE|>", END_RESPONSE, END_TURN, "<|START_OF_TURN_TOKEN|>", "<|CHATBOT_TOKEN|>",
                 "<|USER_TOKEN|>", "<|SYSTEM_TOKEN|>", "<BOS_TOKEN>", "<EOS_TOKEN>", "<PAD>")


def layers(model):
    """The decoder layer list (works on the CausalLM wrapper or the inner model)."""
    inner = getattr(model, "model", model)
    return inner.layers


def n_layers(model) -> int:
    return len(layers(model))


def layer_types(config) -> list[str]:
    """'sliding_attention' / 'full_attention' per layer. Every layer is attention (no conv)."""
    lt = getattr(config, "layer_types", None)
    n = getattr(config, "num_hidden_layers", N_LAYERS)
    if lt:
        return list(lt)
    pat = getattr(config, "sliding_window_pattern", 4) or 4
    return ["full_attention" if (i + 1) % pat == 0 else "sliding_attention" for i in range(n)]


def final_norm(model):
    return getattr(model, "model", model).norm


def embed(model):
    return getattr(model, "model", model).embed_tokens


def param(layer: int, frag: str) -> str:
    """Full parameter name, e.g. param(4, DOWN_PROJ) -> 'model.layers.4.mlp.down_proj.weight'."""
    return f"model.layers.{layer}.{frag}.weight"


def module(model, layer: int, frag: str):
    m = layers(model)[layer]
    for part in frag.split("."):
        m = getattr(m, part)
    return m


# ------------------------------------------------------------------ prompts
def chat_text(tok, user: str, thinking: bool = False, system: str | None = None, prefill: str = "") -> str:
    """Rendered chat prompt ending with <|START_RESPONSE|> (+ prefill). `thinking` is ignored (no thinking mode)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + prefill


def chat_ids(tok, user: str, thinking: bool = False, system: str | None = None, prefill: str = "") -> list[int]:
    return tok(chat_text(tok, user, thinking, system, prefill), add_special_tokens=False)["input_ids"]


def answer_suffix() -> str:
    """What follows an assistant answer in the template; part of the supervised tokens when training on chat pairs."""
    return END_RESPONSE + END_TURN


def chat_pair_ids(tok, user: str, answer: str, system: str | None = None) -> tuple[list[int], int]:
    """(ids, start): prompt + answer + end tokens; labels should cover ids[start:]."""
    p = chat_ids(tok, user, system=system)
    a = tok(answer + answer_suffix(), add_special_tokens=False)["input_ids"]
    return p + a, len(p)


def raw_ids(tok, text: str) -> list[int]:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    bos = tok.bos_token_id
    return ([bos] if bos is not None else []) + ids


def eos_ids(tok) -> list[int]:
    ids = {tok.eos_token_id, *(tok.convert_tokens_to_ids(t) for t in STOP_STRINGS)}
    unk = tok.unk_token_id
    return sorted(i for i in ids if isinstance(i, int) and i >= 0 and i != unk)


def clean(text: str) -> str:
    for s in SPECIAL_STRIP:
        text = text.replace(s, "")
    return text.strip()
