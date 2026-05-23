"""Build a text training corpus from WordNet noun synsets.

For each synset, emits 1-3 short passages combining:
  - the lemma + definition
  - the hypernym chain (X is a kind of Y is a kind of Z)
  - an example sentence (if available)
  - sibling lemmas
  - hyponym lemmas

This produces a small but dense corpus (~5-10 MB raw text) where every
WordNet relation gets explicit text representation. Useful for testing
the training loop end-to-end before scaling to FineWeb-Edu.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from nltk.corpus import wordnet as wn


def _lemma(s) -> str:
    return s.lemmas()[0].name().replace("_", " ")


def render_synset(s, rng: random.Random) -> list[str]:
    """Return 1-3 natural-language passages for this synset."""
    lemmas = [l.name().replace("_", " ") for l in s.lemmas()]
    lemma = lemmas[0]
    defn = s.definition().strip()
    hyps = s.hypernyms()
    hyponyms = s.hyponyms()
    examples = [e.strip() for e in s.examples() if e and e.strip()]

    out: list[str] = []

    # Definition passage.
    out.append(f"A {lemma} is {defn}.")
    if hyps:
        parent = _lemma(hyps[0])
        out.append(f"A {lemma} is a kind of {parent}.")
    if len(hyps) >= 1 and len(hyps[0].hypernyms()) >= 1:
        parent = _lemma(hyps[0])
        grandparent = _lemma(hyps[0].hypernyms()[0])
        out.append(f"A {lemma} is a kind of {parent}, which is a kind of {grandparent}.")
    if hyponyms:
        names = [_lemma(h) for h in hyponyms[:4]]
        if len(names) == 1:
            out.append(f"A {names[0]} is a kind of {lemma}.")
        else:
            joined = ", ".join(names[:-1]) + f", and {names[-1]}"
            out.append(f"Kinds of {lemma} include {joined}.")
    if examples:
        ex = rng.choice(examples)
        out.append(f"Example: {ex}")
    if len(lemmas) > 1:
        synonyms = ", ".join(lemmas[1:5])
        out.append(f"Other words for {lemma}: {synonyms}.")
    return out


def build_corpus(
    out_path: Path,
    pos: str = "n",
    max_synsets: int | None = None,
    seed: int = 0,
) -> dict:
    """Write a plain-text corpus, one passage per line."""
    rng = random.Random(seed)
    synsets = list(wn.all_synsets(pos))
    if max_synsets:
        synsets = synsets[:max_synsets]
    n_passages = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for s in synsets:
            for p in render_synset(s, rng):
                f.write(p + "\n")
                n_passages += 1
    stats = {
        "synsets": len(synsets),
        "passages": n_passages,
        "bytes": out_path.stat().st_size,
        "out": str(out_path),
    }
    return stats


def tokenize_corpus(
    corpus_path: Path,
    tokenizer_path: Path,
    out_path: Path,
) -> dict:
    """Tokenize a plain-text corpus with the fant3 v2 tokenizer and save as
    a flat numpy uint16 array (vocab is 32k so uint16 is sufficient)."""
    import numpy as np
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tok.get_vocab_size()
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    ids: list[int] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            enc = tok.encode(line)
            ids.extend(enc.ids)
            # End-of-passage marker (use vocab-1 if no explicit eos).
            # Tokenizer v2 may have a specific eos; check.
            # For now we don't insert one; chunks are sampled randomly.

    arr = np.array(ids, dtype=dtype)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return {
        "tokens": int(arr.size),
        "vocab_size": vocab_size,
        "dtype": str(arr.dtype),
        "out": str(out_path),
    }
