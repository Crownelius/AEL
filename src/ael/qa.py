"""WordNet QA over AEL — Phase F.

Routes natural-language questions through AEL retrieval and answers from
WordNet metadata (definitions, hypernyms, hyponyms, siblings).

Supported question patterns (case-insensitive, simple regex matching):

  what is a <X>?                 -> definition of X
  what does <X> mean?            -> definition of X
  define <X>                     -> definition of X
  what kinds of <X> are there?   -> hyponyms of X
  what is a kind of <X>?         -> sample hyponym
  what is a <X> a kind of?       -> hypernym of X
  what is <X> a kind of?         -> hypernym of X
  is a <X> a <Y>?                -> entailment via hypernym chain
  what's similar to <X>?         -> AEL retrieved nearest neighbors
  what is similar to <X>?        -> AEL retrieved nearest neighbors
  list <X>                       -> hyponyms of X
  parent of <X>                  -> hypernym

This is a retrieval-and-template system, not a generative model. It is
deliberately narrow but actually works -- the first thing in the loop that
takes natural-language questions and returns coherent answers grounded in
the AEL embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from nltk.corpus import wordnet as wn

from .wordnet_data import WnSubset, load_noun_subset


@dataclass
class Answer:
    text: str
    confidence: float
    evidence: list[str]


# ---------------------------------------------------------------------------
# Synset resolution
# ---------------------------------------------------------------------------


def find_best_synset(word: str, sub: Optional[WnSubset] = None) -> Optional[str]:
    """Resolve a word to its most-common noun synset name (e.g. 'dog' -> 'dog.n.01')."""
    word = word.strip().lower().replace(" ", "_")
    candidates = wn.synsets(word, pos="n")
    if not candidates:
        # Try without POS restriction.
        candidates = wn.synsets(word)
    if not candidates:
        return None
    if sub is not None:
        for c in candidates:
            if c.name() in sub.nodes:
                return c.name()
    return candidates[0].name()


def synset_lemma(synset_name: str) -> str:
    """First lemma name, prettified ('dog.n.01' -> 'dog')."""
    s = wn.synset(synset_name)
    return s.lemmas()[0].name().replace("_", " ")


def synset_def(synset_name: str) -> str:
    return wn.synset(synset_name).definition()


# ---------------------------------------------------------------------------
# Question patterns
# ---------------------------------------------------------------------------


_ART = r"(?:a|an|the)\s+"   # optional article (we'll strip it post-match too)

# Order matters: specific patterns FIRST so the broad "what is X" define
# pattern doesn't swallow them. The first match wins.
PATTERNS: list[tuple[str, str]] = [
    # ---- neo-WordNet world facts (must come BEFORE define) ----
    # Capital lookups
    (rf"^what is (?:the )?capital of\s+(?:{_ART})?(.+?)\??$", "fact_capital"),
    (rf"^(?:the )?capital of\s+(?:{_ART})?(.+?) is$", "fact_capital"),
    # Authors / wrote
    (rf"^who (?:wrote|is the author of)\s+(?:{_ART})?(.+?)\??$", "fact_author"),
    # Painters / painted
    (rf"^who painted\s+(?:{_ART})?(.+?)\??$", "fact_painter"),
    # Dates (year of)
    (rf"^(?:in )?what year did\s+(.+?)(?:\s+(?:end|begin|happen|occur))?\??$", "fact_date"),
    (rf"^when did\s+(.+?)\??$", "fact_date"),
    # Firsts / who was the first X
    (rf"^who (?:was|is) (?:the )?first\s+(.+?)\??$", "fact_first"),
    # Inventor / developed / discovered
    (rf"^who (?:developed|discovered|invented|created)\s+(?:{_ART})?(.+?)\??$", "fact_inventor"),
    # Science: chemical symbol of X
    (rf"^what is (?:the )?chemical symbol (?:for|of)\s+(?:{_ART})?(.+?)\??$", "fact_chemical_symbol"),
    (rf"^what is (?:the )?chemical formula (?:for|of)\s+(?:{_ART})?(.+?)\??$", "fact_chemical_formula"),
    # Science: numeric / temperature
    (rf"^what is (?:the )?(freezing|boiling) point of water in (celsius|fahrenheit)\??$", "fact_temp"),
    (rf"^what is (?:the )?speed of light(?: in vacuum)?(?: approximately)?\??$", "fact_speed_of_light"),
    # Geography
    (rf"^what is (?:the )?(tallest|highest|longest|largest|smallest|deepest)\s+(.+?)(?:\s+(?:in (?:the )?world))?\??$", "fact_superlative"),
    # Planet / solar system
    (rf"^how many planets (?:are )?(?:in our solar system|in the solar system|exist)\??$", "fact_planet_count"),
    (rf"^what is (?:the )?(largest|smallest|hottest|coldest) planet(?:\s+in (?:our )?solar system)?\??$", "fact_planet_super"),
    # Counts / shape sides
    (rf"^how many sides does\s+(?:{_ART})?(.+?) have\??$", "fact_sides"),
    (rf"^how many\s+(.+?)\s+are there\??$", "fact_count"),
    # Languages
    (rf"^what language (?:is )?primarily spoken in\s+(?:{_ART})?(.+?)\??$", "fact_language"),
    (rf"^what (?:language|languages) (?:do they speak|are spoken|do people speak) in\s+(?:{_ART})?(.+?)\??$", "fact_language"),
    # Composers / instruments
    (rf"^what instrument did\s+(.+?) (?:primarily )?(?:compose|play|write) for\??$", "fact_composer"),

    # ---- WordNet QA (existing) ----
    # is-a tests (most specific, must come first)
    (rf"^is\s+(?:{_ART})?(.+?)\s+(?:{_ART})?(.+?)\??$", "is_a"),
    # similarity
    (r"^(?:what(?:'s| is)? )?similar to\s+(.+?)\??$", "similar"),
    # 'X a kind of?' and 'parent of X' -- hypernym lookups
    (rf"^what is\s+(?:{_ART})?(.+?)\s+a kind of\??$", "kind_of_what"),
    (rf"^what is (?:the )?parent of\s+(?:{_ART})?(.+?)\??$", "kind_of_what"),
    (rf"^parent of\s+(?:{_ART})?(.+?)\.?$", "kind_of_what"),
    # 'kinds of X' -- hyponym lookups
    (rf"^what (?:kinds?|types?) of\s+(?:{_ART})?(.+?)\s+(?:are there|exist)\??$", "kinds_of"),
    (rf"^what is a kind of\s+(?:{_ART})?(.+?)\??$", "kinds_of"),
    (rf"^list (?:the )?(?:kinds?|types?) of\s+(?:{_ART})?(.+?)\.?$", "kinds_of"),
    (rf"^name (?:a |an |one )?(?:kinds? of|types? of)\s+(?:{_ART})?(.+?)\.?$", "kinds_of"),
    (rf"^give (?:me )?(?:a |an |one )?(?:kinds? of|types? of|example of)\s+(?:{_ART})?(.+?)\.?$", "kinds_of"),
    # define -- broadest, must come last
    (rf"^(?:what does|what's)\s+(?:{_ART})?(.+?)\s+mean\??$", "define"),
    (rf"^define\s+(?:{_ART})?(.+?)\.?$", "define"),
    (rf"^describe\s+(?:{_ART})?(.+?)\.?$", "define"),
    (rf"^(?:what|who) is\s+(?:{_ART})?(.+?)\??$", "define"),
]


def _strip_article(s: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", s.strip(), flags=re.IGNORECASE)


def parse_question(q: str) -> Optional[tuple[str, list[str]]]:
    q = q.strip()
    for pattern, intent in PATTERNS:
        m = re.match(pattern, q, re.IGNORECASE)
        if m:
            groups = [g.strip(" .?!").strip() for g in m.groups()]
            groups = [_strip_article(g) for g in groups]
            return intent, groups
    return None


# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------


def intent_define(arg: str) -> Answer:
    syn = find_best_synset(arg)
    if syn is None:
        return Answer(text=f"I don't know the word '{arg}'.", confidence=0.0, evidence=[])
    d = synset_def(syn)
    return Answer(text=f"{synset_lemma(syn)}: {d}", confidence=0.9, evidence=[syn])


def intent_kinds_of(arg: str) -> Answer:
    syn = find_best_synset(arg)
    if syn is None:
        return Answer(text=f"I don't know the word '{arg}'.", confidence=0.0, evidence=[])
    s = wn.synset(syn)
    hyps = s.hyponyms()
    if not hyps:
        return Answer(text=f"I don't know of any kinds of {synset_lemma(syn)}.", confidence=0.2, evidence=[syn])
    names = [h.lemmas()[0].name().replace("_", " ") for h in hyps[:10]]
    if len(names) == 1:
        return Answer(text=f"A {synset_lemma(syn)} can be: {names[0]}.", confidence=0.8, evidence=[syn] + [h.name() for h in hyps[:10]])
    return Answer(
        text=f"Kinds of {synset_lemma(syn)} include: " + ", ".join(names[:-1]) + (f", and {names[-1]}" if len(names) > 1 else "") + ".",
        confidence=0.9,
        evidence=[syn] + [h.name() for h in hyps[:10]],
    )


def intent_kind_of_what(arg: str) -> Answer:
    syn = find_best_synset(arg)
    if syn is None:
        return Answer(text=f"I don't know the word '{arg}'.", confidence=0.0, evidence=[])
    s = wn.synset(syn)
    hyps = s.hypernyms()
    if not hyps:
        return Answer(text=f"{synset_lemma(syn)} has no broader category in my knowledge.", confidence=0.3, evidence=[syn])
    parent = hyps[0]
    return Answer(
        text=f"A {synset_lemma(syn)} is a kind of {parent.lemmas()[0].name().replace('_', ' ')}.",
        confidence=0.9,
        evidence=[syn, parent.name()],
    )


def intent_is_a(arg_x: str, arg_y: str) -> Answer:
    sx = find_best_synset(arg_x)
    sy = find_best_synset(arg_y)
    if sx is None or sy is None:
        unknown = arg_x if sx is None else arg_y
        return Answer(text=f"I don't know the word '{unknown}'.", confidence=0.0, evidence=[])
    # Walk the hypernym closure of sx; see if sy appears.
    x = wn.synset(sx)
    y = wn.synset(sy)
    closure = set()
    stack = list(x.hypernyms())
    while stack:
        h = stack.pop()
        if h.name() in closure:
            continue
        closure.add(h.name())
        stack.extend(h.hypernyms())
    if y.name() in closure or y == x:
        return Answer(
            text=f"Yes, a {synset_lemma(sx)} is a kind of {synset_lemma(sy)}.",
            confidence=0.95,
            evidence=[sx, sy],
        )
    return Answer(
        text=f"No, a {synset_lemma(sx)} is not a kind of {synset_lemma(sy)} (in my knowledge).",
        confidence=0.85,
        evidence=[sx, sy],
    )


def intent_similar(arg: str, ael_retrieve_fn=None) -> Answer:
    syn = find_best_synset(arg)
    if syn is None:
        return Answer(text=f"I don't know the word '{arg}'.", confidence=0.0, evidence=[])

    if ael_retrieve_fn is not None:
        try:
            sims = ael_retrieve_fn(syn, k=5)
            if sims:
                names = [synset_lemma(s) for s in sims if s != syn][:5]
                return Answer(
                    text=f"Things similar to {synset_lemma(syn)}: " + ", ".join(names) + ".",
                    confidence=0.85,
                    evidence=[syn] + sims,
                )
        except Exception:
            pass  # fall through to WordNet siblings

    # Fallback: WordNet siblings via shared parent.
    s = wn.synset(syn)
    parents = s.hypernyms()
    if not parents:
        return Answer(text=f"I have nothing similar to {synset_lemma(syn)}.", confidence=0.2, evidence=[syn])
    sibs = []
    for p in parents:
        for c in p.hyponyms():
            if c != s:
                sibs.append(c.name())
    if not sibs:
        return Answer(text=f"I have nothing similar to {synset_lemma(syn)}.", confidence=0.2, evidence=[syn])
    names = [synset_lemma(n) for n in sibs[:5]]
    return Answer(
        text=f"Things similar to {synset_lemma(syn)}: " + ", ".join(names) + ".",
        confidence=0.7,
        evidence=[syn] + sibs[:5],
    )


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def _fact_answer(relation: str, key: str) -> Answer:
    from .facts import lookup
    val = lookup(relation, key)
    if val is None:
        return Answer(text=f"I don't know.", confidence=0.0, evidence=[])
    return Answer(text=val, confidence=0.95, evidence=[f"{relation}:{key}"])


def intent_fact_capital(arg: str) -> Answer:
    return _fact_answer("capital", arg)


def intent_fact_author(arg: str) -> Answer:
    from .facts import lookup
    a = arg.strip().lower().rstrip("?.!,")
    # Try the raw arg first, then strip common work-type prefixes.
    for k in (a, re.sub(r"^(?:the\s+)?(?:play|novel|book|poem|story|film|movie|opera)\s+", "", a)):
        v = lookup("author", k)
        if v is not None:
            return Answer(text=v, confidence=0.95, evidence=[f"author:{k}"])
    return Answer(text="I don't know.", confidence=0.0, evidence=[])


def intent_fact_painter(arg: str) -> Answer:
    return _fact_answer("painter", arg)


def intent_fact_date(arg: str, raw: str = "") -> Answer:
    """Date intent. Try several keys derived from the matched event."""
    from .facts import lookup
    a = arg.strip().lower().rstrip("?.!,")
    # Try compound keys: "world war ii ended", "world war ii began", "world war ii"
    for k in (a + " ended", a + " began", a + " happen", a):
        v = lookup("date", k)
        if v is not None:
            return Answer(text=v, confidence=0.9, evidence=[f"date:{k}"])
    # Try other relations (fall through for things like "the berlin wall fell")
    return _fact_answer("date", a)


def intent_fact_first(arg: str) -> Answer:
    from .facts import lookup
    a = arg.strip().lower().rstrip("?.!,")
    # The lookup table key is e.g. "president of the united states" -> "George Washington"
    for k in (f"first {a}", a):
        v = lookup("first", k)
        if v is not None:
            return Answer(text=v, confidence=0.9, evidence=[f"first:{k}"])
    return Answer(text=f"I don't know.", confidence=0.0, evidence=[])


def intent_fact_inventor(arg: str) -> Answer:
    return _fact_answer("inventor", arg)


def intent_fact_chemical_symbol(arg: str) -> Answer:
    return _fact_answer("science", f"{arg.strip().lower()} chemical symbol")


def intent_fact_chemical_formula(arg: str) -> Answer:
    return _fact_answer("science", f"{arg.strip().lower()} chemical formula")


def intent_fact_temp(kind: str, scale: str) -> Answer:
    return _fact_answer("science", f"{kind} point of water in {scale}")


def intent_fact_speed_of_light() -> Answer:
    return _fact_answer("science", "speed of light approximately")


def intent_fact_superlative(super_kind: str, what: str) -> Answer:
    from .facts import lookup
    key = f"{super_kind} {what.strip().lower()}"
    v = lookup("geography", key)
    if v is None:
        v = lookup("planet", key)
    if v is None:
        return Answer(text=f"I don't know.", confidence=0.0, evidence=[])
    return Answer(text=v, confidence=0.9, evidence=[f"geography:{key}"])


def intent_fact_planet_count() -> Answer:
    return _fact_answer("planet", "number of planets in our solar system")


def intent_fact_planet_super(kind: str) -> Answer:
    return _fact_answer("planet", f"{kind} planet")


def intent_fact_sides(shape: str) -> Answer:
    return _fact_answer("count", f"sides of {'an ' if shape.strip().lower()[:1] in 'aeiou' else 'a '}{shape.strip().lower()}")


def intent_fact_count(arg: str) -> Answer:
    return _fact_answer("count", arg.strip().lower())


def intent_fact_language(country: str) -> Answer:
    return _fact_answer("language", f"language spoken in {country.strip().lower()}")


def intent_fact_composer(name: str) -> Answer:
    return _fact_answer("composer", f"{name.strip().lower()} instrument")


def answer(question: str, ael_retrieve_fn=None, use_fant3_fallback: bool = True) -> Answer:
    parsed = parse_question(question)
    if parsed is None:
        # No Q-pattern match. Two fallbacks:
        #  1. Sentence-completion style (input doesn't end with "?"): route
        #     through the trained fant3 LM for next-token generation.
        #  2. Otherwise: "I don't understand."
        if use_fant3_fallback and not question.rstrip().endswith("?"):
            try:
                from .fant3_gen import complete
                gen = complete(question, max_new=6)
                return Answer(text=gen, confidence=0.5, evidence=["fant3_50m"])
            except Exception as e:
                return Answer(text=f"(no answer: {type(e).__name__})", confidence=0.0, evidence=[])
        return Answer(text="I don't understand that question.", confidence=0.0, evidence=[])
    intent, args = parsed
    # neo-WordNet fact intents
    if intent == "fact_capital":          return intent_fact_capital(args[0])
    if intent == "fact_author":           return intent_fact_author(args[0])
    if intent == "fact_painter":          return intent_fact_painter(args[0])
    if intent == "fact_date":             return intent_fact_date(args[0])
    if intent == "fact_first":            return intent_fact_first(args[0])
    if intent == "fact_inventor":         return intent_fact_inventor(args[0])
    if intent == "fact_chemical_symbol":  return intent_fact_chemical_symbol(args[0])
    if intent == "fact_chemical_formula": return intent_fact_chemical_formula(args[0])
    if intent == "fact_temp":             return intent_fact_temp(args[0], args[1])
    if intent == "fact_speed_of_light":   return intent_fact_speed_of_light()
    if intent == "fact_superlative":      return intent_fact_superlative(args[0], args[1])
    if intent == "fact_planet_count":     return intent_fact_planet_count()
    if intent == "fact_planet_super":     return intent_fact_planet_super(args[0])
    if intent == "fact_sides":            return intent_fact_sides(args[0])
    if intent == "fact_count":            return intent_fact_count(args[0])
    if intent == "fact_language":         return intent_fact_language(args[0])
    if intent == "fact_composer":         return intent_fact_composer(args[0])
    # WordNet intents (existing)
    if intent == "define":         return intent_define(args[0])
    if intent == "kinds_of":       return intent_kinds_of(args[0])
    if intent == "kind_of_what":   return intent_kind_of_what(args[0])
    if intent == "is_a":           return intent_is_a(args[0], args[1])
    if intent == "similar":        return intent_similar(args[0], ael_retrieve_fn=ael_retrieve_fn)
    return Answer(text="I don't understand that question yet.", confidence=0.0, evidence=[])
