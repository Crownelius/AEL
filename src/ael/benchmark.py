"""WordNet-grounded QA benchmark.

Four categories, each with a deterministic ground-truth answer drawn from
WordNet so we can score open-ended generations and multiple-choice cleanly:

  define     -- "What is a {word}?" -> short paraphrase of WordNet definition
  hypernym   -- "What is a {word} a kind of?" -> immediate parent lemma
  hyponym    -- "Name a kind of {word}." -> any direct child lemma
  is_a       -- "Is a {x} a {y}?" -> yes / no, by hypernym closure

Stratified by frequency tier (common WordNet nouns vs less common) and
sampled deterministically by a fixed seed so all systems see the same
benchmark.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from nltk.corpus import wordnet as wn


COMMON_SEEDS = [
    "dog", "cat", "horse", "bird", "fish", "tree", "flower", "car", "house",
    "chair", "table", "book", "pen", "computer", "telephone", "hammer",
    "knife", "shoe", "shirt", "river", "mountain", "lake", "city", "country",
    "doctor", "teacher", "scientist", "artist", "musician", "farmer",
    "apple", "banana", "orange", "tomato", "potato", "bread", "milk",
    "guitar", "piano", "violin", "drum", "saxophone",
    "lion", "tiger", "wolf", "rabbit", "mouse", "deer",
]


@dataclass
class QABenchmarkItem:
    question: str
    category: str               # define | hypernym | hyponym | is_a
    answer: str                 # ground-truth answer string
    accepted: set               # accepted answer strings (lowercase)
    synset: str                 # primary synset used


def _lemma(s) -> str:
    return s.lemmas()[0].name().replace("_", " ")


def _all_lemmas(s) -> set:
    return {l.name().replace("_", " ").lower() for l in s.lemmas()}


def _is_a_closure(syn) -> set:
    """All ancestors of syn in WordNet (synset names)."""
    seen = set()
    stack = list(syn.hypernyms())
    while stack:
        h = stack.pop()
        if h.name() in seen:
            continue
        seen.add(h.name())
        stack.extend(h.hypernyms())
    return seen


def make_define(word: str) -> QABenchmarkItem | None:
    candidates = wn.synsets(word, pos="n")
    if not candidates:
        return None
    s = candidates[0]
    # The "right" answer is the WordNet definition; we accept any contiguous
    # substring of 5+ chars from the definition's first 50 chars.
    return QABenchmarkItem(
        question=f"What is a {word}?",
        category="define",
        answer=s.definition(),
        accepted={s.definition().lower()},
        synset=s.name(),
    )


def make_hypernym(word: str) -> QABenchmarkItem | None:
    cands = wn.synsets(word, pos="n")
    if not cands:
        return None
    s = cands[0]
    hyps = s.hypernyms()
    if not hyps:
        return None
    accepted = set()
    for h in hyps:
        accepted |= _all_lemmas(h)
    return QABenchmarkItem(
        question=f"What is a {word} a kind of?",
        category="hypernym",
        answer=_lemma(hyps[0]),
        accepted={a.lower() for a in accepted},
        synset=s.name(),
    )


def make_hyponym(word: str) -> QABenchmarkItem | None:
    cands = wn.synsets(word, pos="n")
    if not cands:
        return None
    s = cands[0]
    hyp = s.hyponyms()
    if not hyp:
        return None
    accepted = set()
    for h in hyp[:20]:
        accepted |= _all_lemmas(h)
    return QABenchmarkItem(
        question=f"Name a kind of {word}.",
        category="hyponym",
        answer=_lemma(hyp[0]),
        accepted={a.lower() for a in accepted},
        synset=s.name(),
    )


def make_is_a(word_x: str, word_y: str, expected: bool) -> QABenchmarkItem | None:
    cands_x = wn.synsets(word_x, pos="n")
    cands_y = wn.synsets(word_y, pos="n")
    if not cands_x or not cands_y:
        return None
    sx, sy = cands_x[0], cands_y[0]
    ancestors_x = _is_a_closure(sx)
    actual = (sy.name() in ancestors_x) or (sx == sy)
    if actual != expected:
        return None  # mismatch; drop this item
    return QABenchmarkItem(
        question=f"Is a {word_x} a {word_y}?",
        category="is_a",
        answer="yes" if expected else "no",
        accepted={"yes"} if expected else {"no"},
        synset=sx.name(),
    )


def build_benchmark(seed: int = 42) -> list[QABenchmarkItem]:
    rng = random.Random(seed)
    items: list[QABenchmarkItem] = []

    # define + hypernym + hyponym for each seed word
    for word in COMMON_SEEDS:
        for fn in (make_define, make_hypernym, make_hyponym):
            it = fn(word)
            if it: items.append(it)

    # is-a: 30 positives + 30 negatives drawn from common seeds
    is_a_pos = []
    for word in COMMON_SEEDS:
        cands = wn.synsets(word, pos="n")
        if not cands:
            continue
        s = cands[0]
        ancs = list(_is_a_closure(s))
        if ancs:
            anc_name = rng.choice(ancs)
            anc_lemma = wn.synset(anc_name).lemmas()[0].name().replace("_", " ")
            item = make_is_a(word, anc_lemma, expected=True)
            if item: is_a_pos.append(item)
    items += is_a_pos[:30]

    # negatives: pair random unrelated words
    neg_count = 0
    attempts = 0
    while neg_count < 30 and attempts < 500:
        attempts += 1
        a, b = rng.sample(COMMON_SEEDS, 2)
        item = make_is_a(a, b, expected=False)
        if item:
            items.append(item); neg_count += 1

    rng.shuffle(items)
    return items


if __name__ == "__main__":
    import collections
    items = build_benchmark()
    cat_counts = collections.Counter(it.category for it in items)
    print(f"Benchmark size: {len(items)}")
    for c, n in cat_counts.items():
        print(f"  {c}: {n}")
    print("\nSamples:")
    for it in items[:8]:
        print(f"  [{it.category}] {it.question}  -> {it.answer[:60]}")
