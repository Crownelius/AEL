"""WordNet noun hierarchy loader.

For iteration 1 we want a tractable subset of WordNet nouns with:
  - a single root (entity.n.01)
  - bounded depth from root
  - hypernym / hyponym / sibling relations as ground truth

This module returns a tree-as-DAG (WordNet is technically a DAG; we keep all
parents but break ties for tree layout by picking the first hypernym).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import nltk
from nltk.corpus import wordnet as wn


def _ensure_wordnet() -> None:
    try:
        wn.synset("entity.n.01")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


@dataclass
class WnNode:
    name: str           # synset name, e.g. "dog.n.01"
    depth: int          # distance from entity.n.01 via primary-parent chain
    parent: str | None  # primary hypernym (first one)
    children: list[str] = field(default_factory=list)


@dataclass
class WnSubset:
    """A truncated subtree of WordNet rooted at a given synset."""

    root: str
    nodes: dict[str, WnNode]

    def __len__(self) -> int:
        return len(self.nodes)

    def siblings(self, name: str) -> list[str]:
        """Other children of this node's primary parent."""
        n = self.nodes[name]
        if n.parent is None:
            return []
        return [c for c in self.nodes[n.parent].children if c != name]

    def hypernyms(self, name: str) -> list[str]:
        """Ancestor chain (parent, grandparent, ...) within the subset."""
        out = []
        cur = self.nodes[name].parent
        while cur is not None and cur in self.nodes:
            out.append(cur)
            cur = self.nodes[cur].parent
        return out

    def hyponyms(self, name: str) -> list[str]:
        return list(self.nodes[name].children)


def load_noun_subset(
    root_name: str = "entity.n.01",
    max_depth: int = 6,
    max_nodes: int = 5000,
) -> WnSubset:
    """BFS-load WordNet nouns from `root_name` to bounded depth/size."""
    _ensure_wordnet()

    root_synset = wn.synset(root_name)
    nodes: dict[str, WnNode] = {}
    nodes[root_synset.name()] = WnNode(name=root_synset.name(), depth=0, parent=None)

    queue: deque[tuple[object, int]] = deque([(root_synset, 0)])
    while queue and len(nodes) < max_nodes:
        synset, d = queue.popleft()
        if d >= max_depth:
            continue
        for child in synset.hyponyms():
            cname = child.name()
            if cname in nodes:
                continue
            # Primary parent = `synset` (the one we expanded from).
            nodes[cname] = WnNode(name=cname, depth=d + 1, parent=synset.name())
            nodes[synset.name()].children.append(cname)
            queue.append((child, d + 1))
            if len(nodes) >= max_nodes:
                break

    return WnSubset(root=root_synset.name(), nodes=nodes)


if __name__ == "__main__":
    sub = load_noun_subset(max_depth=5, max_nodes=2000)
    print(f"Loaded {len(sub)} synsets from {sub.root} (depth<=5)")
    # A spot check.
    sample = "dog.n.01"
    if sample in sub.nodes:
        print(f"\n{sample}")
        print(f"  parent: {sub.nodes[sample].parent}")
        print(f"  depth:  {sub.nodes[sample].depth}")
        print(f"  hypernym chain: {sub.hypernyms(sample)}")
        print(f"  siblings (first 5): {sub.siblings(sample)[:5]}")
        print(f"  hyponyms (first 5): {sub.hyponyms(sample)[:5]}")
