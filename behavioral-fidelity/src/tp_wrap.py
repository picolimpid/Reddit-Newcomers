"""Run a (possibly frozen) vLLM script unchanged, with multi-GPU workarounds for this box (2026-11-07).

On this machine TP=2 hangs twice: NCCL P2P init deadlocks (fix: NCCL_P2P_DISABLE=1, set here) and, with P2P off, the
first forward pass hangs in vLLM's custom all-reduce (fix: disable_custom_all_reduce=True, patched into vllm.LLM).
Neither changes the numbers a model computes beyond floating-point reduction order.
GEN_SUFFIX (optional, e.g. a newline for GLM-4-0414, whose first generated token is "\n" with p ~ 1, see DBG-glm4-bf16c.log)
is appended to every chat-template string rendered with add_generation_prompt=True, so the Yes/No readout reads the
token after it. Same suffix in every condition; a model-specific deviation that the paper reports.

    python tp_wrap.py panel_agents.py --site ffxiv2019 --model ... --tp 2"""
import os
import runpy
import sys

os.environ.setdefault("NCCL_P2P_DISABLE", "1")

import vllm  # noqa: E402

_init = vllm.LLM.__init__


def _patched(self, *a, **kw):
    kw.setdefault("disable_custom_all_reduce", True)
    return _init(self, *a, **kw)


vllm.LLM.__init__ = _patched

SUFFIX = os.environ.get("GEN_SUFFIX", "").encode().decode("unicode_escape")
if SUFFIX:
    from transformers import PreTrainedTokenizerBase
    _tmpl = PreTrainedTokenizerBase.apply_chat_template

    def _tmpl_patched(self, *a, **kw):
        out = _tmpl(self, *a, **kw)
        if kw.get("add_generation_prompt") and kw.get("tokenize") is False and isinstance(out, str):
            out += SUFFIX
        return out
    PreTrainedTokenizerBase.apply_chat_template = _tmpl_patched
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
