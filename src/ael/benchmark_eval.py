"""Scoring + runners for the WordNet QA benchmark.

Both systems produce a free-text answer per question; we score with the
same rules for fair comparison. Scoring rules per category:

  define     -- 1 if generated text contains any noun-lemma of the definition
                (we strip stopwords/articles, look for the key content words)
  hypernym   -- 1 if generated text contains any accepted hypernym lemma
  hyponym    -- 1 if generated text contains any accepted hyponym lemma
  is_a       -- 1 if generated text starts with yes/no matching expected
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .benchmark import QABenchmarkItem


STOPWORDS = {
    "a", "an", "the", "is", "are", "of", "to", "in", "for", "with", "or",
    "and", "by", "that", "this", "as", "any", "some", "be", "from", "at",
    "on", "having", "used", "one", "kind", "type", "form", "very",
    "having", "act", "thing", "person", "place",
}


def _normalize(text: str) -> str:
    return text.lower().replace("_", " ").strip()


def _content_words(text: str) -> set[str]:
    text = _normalize(text)
    tokens = re.findall(r"[a-z][a-z\-]+", text)
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def score_item(item: QABenchmarkItem, generated: str) -> bool:
    g = _normalize(generated)
    if item.category == "is_a":
        # First yes/no token wins.
        m = re.search(r"\b(yes|no)\b", g)
        if not m:
            return False
        return m.group(1) == item.answer.lower()

    g_words = _content_words(g)

    if item.category == "define":
        # Score: at least one informative content word from the definition
        # appears in the generation.
        target = _content_words(item.answer)
        # Filter target to "head noun"-ish content words (length >= 4, not too generic).
        target = {t for t in target if len(t) > 3}
        return len(g_words & target) >= 1

    # hypernym / hyponym: any accepted lemma anywhere in the generation.
    for acc in item.accepted:
        acc_n = _normalize(acc)
        if acc_n and acc_n in g:
            return True
    return False


# ---------------------------------------------------------------------------
# GPT-2 runner
# ---------------------------------------------------------------------------


_GPT2_MODEL = None
_GPT2_TOK = None
_HF_MODELS: dict = {}


def _get_gpt2(model_name: str = "gpt2"):
    global _GPT2_MODEL, _GPT2_TOK
    if _GPT2_MODEL is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _GPT2_TOK = AutoTokenizer.from_pretrained(model_name)
        _GPT2_MODEL = AutoModelForCausalLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            _GPT2_MODEL = _GPT2_MODEL.to("cuda")
        _GPT2_MODEL.eval()
        if _GPT2_TOK.pad_token is None:
            _GPT2_TOK.pad_token = _GPT2_TOK.eos_token
    return _GPT2_MODEL, _GPT2_TOK


def _get_hf(model_name: str):
    """Cache any HF causal-LM. Returns (model, tokenizer)."""
    if model_name not in _HF_MODELS:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _HF_MODELS[model_name] = (model, tok)
    return _HF_MODELS[model_name]


def hf_answer(question: str, model_name: str, max_new: int = 30) -> str:
    """Generic HF causal-LM zero-shot Q/A. Works for GPT-2, Qwen, etc."""
    import torch
    model, tok = _get_hf(model_name)
    prompt = f"Q: {question}\nA:"
    enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    after = full[len(prompt):]
    for stop in ("\nQ:", "\n\n", "\nA:"):
        if stop in after:
            after = after.split(stop)[0]
    return after.strip()


def gpt2_answer(question: str, model_name: str = "gpt2", max_new: int = 30) -> str:
    import torch
    model, tok = _get_gpt2(model_name)
    prompt = f"Q: {question}\nA:"
    enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    # Strip prompt, take everything until next \n or 'Q:' or end.
    after = full[len(prompt):]
    for stop in ("\nQ:", "\n\n", "\nA:"):
        if stop in after:
            after = after.split(stop)[0]
    return after.strip()


# ---------------------------------------------------------------------------
# AEL runner
# ---------------------------------------------------------------------------


def ael_answer(question: str, ael_retrieve_fn=None) -> str:
    from .qa import answer
    return answer(question, ael_retrieve_fn=ael_retrieve_fn).text


# ---------------------------------------------------------------------------
# Top-level evaluator
# ---------------------------------------------------------------------------


def evaluate(items: list[QABenchmarkItem], system_fn, name: str, verbose: bool = False) -> dict:
    correct = {"define": 0, "hypernym": 0, "hyponym": 0, "is_a": 0}
    total = {"define": 0, "hypernym": 0, "hyponym": 0, "is_a": 0}
    samples = {"define": [], "hypernym": [], "hyponym": [], "is_a": []}
    for it in items:
        gen = system_fn(it.question)
        ok = score_item(it, gen)
        correct[it.category] += int(ok)
        total[it.category] += 1
        if len(samples[it.category]) < 3:
            samples[it.category].append((it.question, gen, ok, it.answer[:40]))
    overall = sum(correct.values()) / max(sum(total.values()), 1)
    out = {
        "name": name,
        "overall": overall,
        "per_cat": {c: correct[c] / max(total[c], 1) for c in total},
        "n": sum(total.values()),
        "samples": samples,
    }
    if verbose:
        print(f"\n  {name} overall: {overall:.3f}")
        for c in total:
            print(f"    {c:<10} {correct[c]:>3}/{total[c]:<3}  ({out['per_cat'][c]:.3f})")
    return out
