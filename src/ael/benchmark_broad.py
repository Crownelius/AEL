"""Broader benchmark covering WordNet QA + open-domain trivia + LM completion.

Three categories:
  - wn_*       : WordNet-shaped QA (define, hypernym, hyponym, is_a) -- AEL home
  - trivia_*   : world-knowledge facts (capitals, dates, who/what) -- mixed turf
  - complete_* : last-word sentence completion (LAMBADA-style) -- GPT-2 home

Each item has accepted answers (lowercase set) so the same scoring rules apply.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BroadItem:
    question: str
    category: str           # broad bucket: wn / trivia / complete
    subcategory: str        # finer label: define / hypernym / capital / date / completion
    accepted: set           # acceptable answer strings (lowercase)
    notes: str = ""


# ---------------------------------------------------------------------------
# Open-domain trivia
# ---------------------------------------------------------------------------

TRIVIA_ITEMS: list[BroadItem] = [
    # Capitals
    BroadItem("What is the capital of France?",          "trivia", "capital", {"paris"}),
    BroadItem("What is the capital of Japan?",           "trivia", "capital", {"tokyo"}),
    BroadItem("What is the capital of Germany?",         "trivia", "capital", {"berlin"}),
    BroadItem("What is the capital of Italy?",           "trivia", "capital", {"rome"}),
    BroadItem("What is the capital of Spain?",           "trivia", "capital", {"madrid"}),
    BroadItem("What is the capital of Canada?",          "trivia", "capital", {"ottawa"}),
    BroadItem("What is the capital of Australia?",       "trivia", "capital", {"canberra"}),
    BroadItem("What is the capital of Brazil?",          "trivia", "capital", {"brasilia", "brasília"}),
    BroadItem("What is the capital of Egypt?",           "trivia", "capital", {"cairo"}),
    BroadItem("What is the capital of Russia?",          "trivia", "capital", {"moscow"}),
    BroadItem("What is the capital of India?",           "trivia", "capital", {"new delhi", "delhi"}),
    BroadItem("What is the capital of China?",           "trivia", "capital", {"beijing"}),
    BroadItem("What is the capital of Mexico?",          "trivia", "capital", {"mexico city"}),
    BroadItem("What is the capital of South Korea?",     "trivia", "capital", {"seoul"}),
    BroadItem("What is the capital of Argentina?",       "trivia", "capital", {"buenos aires"}),
    # Year facts
    BroadItem("In what year did World War II end?",      "trivia", "date", {"1945"}),
    BroadItem("In what year did World War I begin?",     "trivia", "date", {"1914"}),
    BroadItem("In what year was the United States declaration of independence signed?", "trivia", "date", {"1776"}),
    BroadItem("In what year did the Berlin Wall fall?",  "trivia", "date", {"1989"}),
    BroadItem("In what year did humans first land on the moon?", "trivia", "date", {"1969"}),
    # Who/what trivia
    BroadItem("Who wrote the play Hamlet?",              "trivia", "who", {"shakespeare", "william shakespeare"}),
    BroadItem("Who painted the Mona Lisa?",              "trivia", "who", {"leonardo da vinci", "da vinci", "leonardo"}),
    BroadItem("Who was the first president of the United States?", "trivia", "who", {"washington", "george washington"}),
    BroadItem("Who developed the theory of general relativity?", "trivia", "who", {"einstein", "albert einstein"}),
    BroadItem("Who wrote the novel 1984?",               "trivia", "who", {"orwell", "george orwell"}),
    BroadItem("What is the largest planet in our solar system?", "trivia", "what", {"jupiter"}),
    BroadItem("What is the chemical symbol for gold?",   "trivia", "what", {"au"}),
    BroadItem("What is the chemical symbol for water?",  "trivia", "what", {"h2o", "h₂o"}),
    BroadItem("What is the speed of light in vacuum approximately?", "trivia", "what", {"300000", "299792", "3 x 10", "3*10"}),
    BroadItem("What language is primarily spoken in Brazil?", "trivia", "what", {"portuguese"}),
    BroadItem("How many continents are there?",           "trivia", "what", {"7", "seven"}),
    BroadItem("How many sides does a hexagon have?",     "trivia", "what", {"6", "six"}),
    BroadItem("How many planets are in our solar system?","trivia", "what", {"8", "eight"}),
    BroadItem("What is the tallest mountain in the world?","trivia", "what", {"everest", "mount everest"}),
    BroadItem("What is the longest river in the world?", "trivia", "what", {"nile", "amazon"}),
    BroadItem("What is the largest ocean on Earth?",     "trivia", "what", {"pacific", "pacific ocean"}),
    BroadItem("What is the smallest country in the world?","trivia", "what", {"vatican", "vatican city"}),
    BroadItem("What instrument did Beethoven primarily compose for?","trivia", "what", {"piano"}),
    BroadItem("What is the freezing point of water in Celsius?","trivia", "what", {"0", "zero"}),
    BroadItem("What is the boiling point of water in Celsius?","trivia", "what", {"100", "one hundred"}),
]

# ---------------------------------------------------------------------------
# Sentence completion (LAMBADA-style)
# ---------------------------------------------------------------------------

COMPLETIONS: list[BroadItem] = [
    BroadItem("The sun rises in the east and sets in the",      "complete", "completion", {"west"}),
    BroadItem("Birds have feathers and can usually",            "complete", "completion", {"fly"}),
    BroadItem("Fish breathe through their",                     "complete", "completion", {"gills"}),
    BroadItem("Bees produce a sweet substance called",          "complete", "completion", {"honey"}),
    BroadItem("Cows produce a white liquid called",             "complete", "completion", {"milk"}),
    BroadItem("Bread is made from flour and",                   "complete", "completion", {"water", "yeast"}),
    BroadItem("Snow is made of frozen",                         "complete", "completion", {"water"}),
    BroadItem("Three plus four equals",                         "complete", "completion", {"7", "seven"}),
    BroadItem("Two times three equals",                         "complete", "completion", {"6", "six"}),
    BroadItem("The opposite of hot is",                         "complete", "completion", {"cold"}),
    BroadItem("The opposite of big is",                         "complete", "completion", {"small", "little"}),
    BroadItem("The opposite of happy is",                       "complete", "completion", {"sad", "unhappy"}),
    BroadItem("Cats are commonly afraid of",                    "complete", "completion", {"dogs", "water"}),
    BroadItem("Dogs are often considered humans' best",         "complete", "completion", {"friend", "friends"}),
    BroadItem("A baby cat is called a",                         "complete", "completion", {"kitten"}),
    BroadItem("A baby dog is called a",                         "complete", "completion", {"puppy"}),
    BroadItem("Lions live in groups called",                    "complete", "completion", {"prides", "pride"}),
    BroadItem("People wear shoes on their",                     "complete", "completion", {"feet"}),
    BroadItem("People wear gloves on their",                    "complete", "completion", {"hands"}),
    BroadItem("Spiders have eight",                             "complete", "completion", {"legs"}),
    BroadItem("Insects have six",                               "complete", "completion", {"legs"}),
    BroadItem("Penguins cannot",                                "complete", "completion", {"fly"}),
    BroadItem("Ice is solid",                                   "complete", "completion", {"water"}),
    BroadItem("Steam is gaseous",                               "complete", "completion", {"water"}),
    BroadItem("The capital of the moon is",                     "complete", "trick", {"none", "no", "there is", "doesn't"}),  # trick
]


def build_broad_benchmark(include_wordnet: bool = True) -> list[BroadItem]:
    """Combine WordNet QA + trivia + completion."""
    items: list[BroadItem] = []
    if include_wordnet:
        from .benchmark import build_benchmark
        wn_items = build_benchmark(seed=42)
        for it in wn_items:
            items.append(BroadItem(
                question=it.question,
                category="wn",
                subcategory=it.category,
                accepted=it.accepted,
                notes=it.synset,
            ))
    items.extend(TRIVIA_ITEMS)
    items.extend(COMPLETIONS)
    return items
