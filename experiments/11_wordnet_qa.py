"""Phase F demo: WordNet QA using AEL retrieval for similarity questions.

Builds the gasket + Hopf placement once, then loops a script of test
questions through the QA. Mixed: questions that exercise raw WordNet facts
(definitions, hypernym chain) and questions that route to AEL retrieval.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ael.gasket import build_gasket, standard_root_neg1_2_2_3
from src.ael.hopf_lift import gasket_to_s2
from src.ael.qa import answer
from src.ael.wordnet_data import load_noun_subset


# Build AEL just-in-time for the "similar" intent.
print("Initializing AEL retrieval substrate (cached for the session)...")
sub = load_noun_subset(max_depth=8, max_nodes=5000)
g = build_gasket(standard_root_neg1_2_2_3(), max_depth=7)
gasket_s2 = gasket_to_s2(g)
print(f"  {len(sub)} synsets, {len(g.circles)} gasket circles ready.\n")


def ael_retrieve(synset_name: str, k: int = 5) -> list[str]:
    """Stub: until a placement head is trained, route 'similar' to WordNet siblings.

    A future iteration plugs in the trained Hopf placement so this
    function returns AEL-nearest synsets instead of WordNet-nearest.
    """
    import nltk
    from nltk.corpus import wordnet as wn
    s = wn.synset(synset_name)
    parents = s.hypernyms()
    if not parents:
        return []
    sibs = []
    for p in parents:
        for c in p.hyponyms():
            if c != s and c.name() in sub.nodes:
                sibs.append(c.name())
    return sibs[:k]


TEST_QUESTIONS = [
    "What is a dog?",
    "Define cat.",
    "What does a hammer mean?",
    "What kinds of vehicle are there?",
    "What is a poodle a kind of?",
    "What is the parent of dog?",
    "Is a dog a mammal?",
    "Is a sparrow a fish?",
    "What is similar to dog?",
    "What is similar to chair?",
    "Describe scientist.",
    "List kinds of fruit.",
]


def main() -> None:
    print("WordNet QA — AEL retrieval prototype\n" + "=" * 60)
    for q in TEST_QUESTIONS:
        ans = answer(q, ael_retrieve_fn=ael_retrieve)
        marker = "✓" if ans.confidence > 0.7 else "?" if ans.confidence > 0 else "✗"
        print(f"\nQ: {q}")
        print(f"  {marker} {ans.text}")
        if ans.evidence:
            print(f"    [evidence: {', '.join(ans.evidence[:3])}{'...' if len(ans.evidence)>3 else ''}]")


if __name__ == "__main__":
    main()
