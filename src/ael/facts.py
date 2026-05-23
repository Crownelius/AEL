"""Structured world-facts layer — the "neo-WordNet" extension.

A small relational knowledge graph covering categories where WordNet is
silent: capitals, named-entity authorship, founding dates, scientific
constants, geography, common-knowledge facts.

Schema:
  facts: dict[(subject, relation), object]

Examples:
  ("france", "capital")           -> "paris"
  ("hamlet", "author")            -> "william shakespeare"
  ("mona lisa", "painter")        -> "leonardo da vinci"
  ("world war ii", "ended")       -> "1945"
  ("gold", "chemical symbol")     -> "au"
  ("water", "chemical formula")   -> "h2o"
  ("hexagon", "sides")            -> "6"
  ("jupiter", "type")             -> "planet"
  ("everest", "type")             -> "mountain"
  ("nile", "type")                -> "river"

The QA layer matches a question to (relation, subject) by surface patterns,
then looks up the fact. AEL routes the *retrieval* — currently a flat dict,
but in the next iteration we'll embed each (s,r,o) triple onto the gasket so
nearby concepts are tangent and queries can fan out via cone attention.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str


# ===========================================================================
# Capitals — countries + their capitals (covers a frequent trivia category)
# ===========================================================================

CAPITALS: dict[str, str] = {
    "france": "Paris",
    "japan": "Tokyo",
    "germany": "Berlin",
    "italy": "Rome",
    "spain": "Madrid",
    "canada": "Ottawa",
    "australia": "Canberra",
    "brazil": "Brasília",
    "egypt": "Cairo",
    "russia": "Moscow",
    "india": "New Delhi",
    "china": "Beijing",
    "mexico": "Mexico City",
    "south korea": "Seoul",
    "argentina": "Buenos Aires",
    "united kingdom": "London",
    "uk": "London",
    "great britain": "London",
    "england": "London",
    "united states": "Washington, D.C.",
    "usa": "Washington, D.C.",
    "us": "Washington, D.C.",
    "america": "Washington, D.C.",
    "portugal": "Lisbon",
    "netherlands": "Amsterdam",
    "belgium": "Brussels",
    "greece": "Athens",
    "turkey": "Ankara",
    "sweden": "Stockholm",
    "norway": "Oslo",
    "finland": "Helsinki",
    "denmark": "Copenhagen",
    "ireland": "Dublin",
    "austria": "Vienna",
    "switzerland": "Bern",
    "poland": "Warsaw",
    "ukraine": "Kyiv",
    "iran": "Tehran",
    "iraq": "Baghdad",
    "saudi arabia": "Riyadh",
    "israel": "Jerusalem",
    "south africa": "Pretoria",
    "nigeria": "Abuja",
    "kenya": "Nairobi",
    "morocco": "Rabat",
    "thailand": "Bangkok",
    "vietnam": "Hanoi",
    "indonesia": "Jakarta",
    "philippines": "Manila",
    "malaysia": "Kuala Lumpur",
    "singapore": "Singapore",
    "new zealand": "Wellington",
    "chile": "Santiago",
    "peru": "Lima",
    "colombia": "Bogotá",
    "venezuela": "Caracas",
    "cuba": "Havana",
}


# ===========================================================================
# Notable works -> authors / creators
# ===========================================================================

AUTHORS: dict[str, str] = {
    "hamlet":               "William Shakespeare",
    "macbeth":              "William Shakespeare",
    "romeo and juliet":     "William Shakespeare",
    "othello":              "William Shakespeare",
    "king lear":            "William Shakespeare",
    "1984":                 "George Orwell",
    "animal farm":          "George Orwell",
    "brave new world":      "Aldous Huxley",
    "to kill a mockingbird":"Harper Lee",
    "moby-dick":            "Herman Melville",
    "moby dick":            "Herman Melville",
    "the great gatsby":     "F. Scott Fitzgerald",
    "war and peace":        "Leo Tolstoy",
    "anna karenina":        "Leo Tolstoy",
    "crime and punishment": "Fyodor Dostoevsky",
    "the brothers karamazov":"Fyodor Dostoevsky",
    "pride and prejudice":  "Jane Austen",
    "sense and sensibility":"Jane Austen",
    "wuthering heights":    "Emily Brontë",
    "jane eyre":            "Charlotte Brontë",
    "ulysses":              "James Joyce",
    "the odyssey":          "Homer",
    "the iliad":            "Homer",
    "the divine comedy":    "Dante Alighieri",
    "don quixote":          "Miguel de Cervantes",
    "the lord of the rings":"J.R.R. Tolkien",
    "the hobbit":           "J.R.R. Tolkien",
    "harry potter":         "J.K. Rowling",
    "the hitchhiker's guide to the galaxy": "Douglas Adams",
}

PAINTERS: dict[str, str] = {
    "mona lisa":            "Leonardo da Vinci",
    "the last supper":      "Leonardo da Vinci",
    "starry night":         "Vincent van Gogh",
    "the starry night":     "Vincent van Gogh",
    "sunflowers":           "Vincent van Gogh",
    "guernica":             "Pablo Picasso",
    "the persistence of memory": "Salvador Dalí",
    "the scream":           "Edvard Munch",
    "girl with a pearl earring": "Johannes Vermeer",
    "the night watch":      "Rembrandt",
    "the birth of venus":   "Sandro Botticelli",
    "the creation of adam": "Michelangelo",
    "the sistine chapel ceiling": "Michelangelo",
}

# ===========================================================================
# Dates -- events that ended/started in a year
# ===========================================================================

DATES: dict[str, str] = {
    "world war ii":         "1939–1945",
    "world war ii ended":   "1945",
    "world war ii began":   "1939",
    "world war i":          "1914–1918",
    "world war i ended":    "1918",
    "world war i began":    "1914",
    "us declaration of independence": "1776",
    "berlin wall fell":     "1989",
    "berlin wall built":    "1961",
    "moon landing":         "1969",
    "first moon landing":   "1969",
    "french revolution":    "1789",
    "fall of the roman empire": "476",
    "fall of constantinople": "1453",
    "magna carta signed":   "1215",
    "discovery of america": "1492",
    "columbus reached the americas": "1492",
}

# ===========================================================================
# Historic firsts / who did what
# ===========================================================================

FIRSTS: dict[str, str] = {
    "first president of the united states":  "George Washington",
    "first man on the moon":                  "Neil Armstrong",
    "first to circumnavigate the earth":      "Ferdinand Magellan",
    "first emperor of rome":                  "Augustus",
    "first computer programmer":              "Ada Lovelace",
    "first to fly an airplane":               "the Wright brothers",
}

# ===========================================================================
# Scientists / inventors
# ===========================================================================

INVENTORS: dict[str, str] = {
    "theory of general relativity":           "Albert Einstein",
    "theory of relativity":                   "Albert Einstein",
    "theory of evolution":                    "Charles Darwin",
    "natural selection":                      "Charles Darwin",
    "laws of motion":                         "Isaac Newton",
    "the telephone":                          "Alexander Graham Bell",
    "the light bulb":                         "Thomas Edison",
    "the printing press":                     "Johannes Gutenberg",
    "penicillin":                             "Alexander Fleming",
    "the periodic table":                     "Dmitri Mendeleev",
    "the radio":                              "Guglielmo Marconi",
}

# ===========================================================================
# Scientific facts / units / constants
# ===========================================================================

SCIENCE: dict[str, str] = {
    "gold chemical symbol":                   "Au",
    "silver chemical symbol":                 "Ag",
    "iron chemical symbol":                   "Fe",
    "oxygen chemical symbol":                 "O",
    "hydrogen chemical symbol":               "H",
    "water chemical formula":                 "H2O",
    "carbon dioxide chemical formula":        "CO2",
    "speed of light":                         "approximately 299,792 km/s (3 × 10^8 m/s)",
    "speed of light approximately":           "300,000",
    "freezing point of water in celsius":     "0",
    "boiling point of water in celsius":      "100",
    "freezing point of water in fahrenheit":  "32",
    "boiling point of water in fahrenheit":   "212",
    "human body normal temperature in celsius": "37",
    "earth gravity":                          "approximately 9.81 m/s²",
    "atoms in a mole":                        "approximately 6.022 × 10^23 (Avogadro's number)",
}

# ===========================================================================
# Geography / superlatives
# ===========================================================================

GEOGRAPHY: dict[str, str] = {
    "tallest mountain":         "Mount Everest",
    "highest mountain":         "Mount Everest",
    "longest river":            "the Nile (or the Amazon, by some measures)",
    "largest ocean":            "the Pacific Ocean",
    "largest continent":        "Asia",
    "smallest continent":       "Australia",
    "smallest country":         "Vatican City",
    "largest country by area":  "Russia",
    "largest country by population": "India",
    "largest desert":           "the Antarctic (or the Sahara among hot deserts)",
    "largest rainforest":       "the Amazon",
    "deepest ocean trench":     "the Mariana Trench",
    "deepest lake":             "Lake Baikal",
}

# ===========================================================================
# Solar system / planets
# ===========================================================================

PLANETS: dict[str, str] = {
    "largest planet":           "Jupiter",
    "largest planet in our solar system": "Jupiter",
    "smallest planet":          "Mercury",
    "smallest planet in our solar system": "Mercury",
    "hottest planet":           "Venus",
    "closest planet to the sun":"Mercury",
    "red planet":               "Mars",
    "number of planets":        "8",
    "number of planets in our solar system": "8",
}

# ===========================================================================
# Counts / simple shape facts
# ===========================================================================

COUNTS: dict[str, str] = {
    "sides of a triangle":      "3",
    "sides of a square":        "4",
    "sides of a pentagon":      "5",
    "sides of a hexagon":       "6",
    "sides of a heptagon":      "7",
    "sides of an octagon":      "8",
    "sides of a nonagon":       "9",
    "sides of a decagon":       "10",
    "number of continents":     "7",
    "continents":               "7",
    "days in a week":           "7",
    "months in a year":         "12",
    "hours in a day":           "24",
    "minutes in an hour":       "60",
    "seconds in a minute":      "60",
    "letters in the english alphabet": "26",
    "musketeers":               "three",
    "deadly sins":              "seven",
    "wonders of the ancient world": "seven",
}

# ===========================================================================
# Languages
# ===========================================================================

LANGUAGES: dict[str, str] = {
    "language spoken in brazil":       "Portuguese",
    "language spoken in argentina":    "Spanish",
    "language spoken in mexico":       "Spanish",
    "language spoken in france":       "French",
    "language spoken in germany":      "German",
    "language spoken in italy":        "Italian",
    "language spoken in japan":        "Japanese",
    "language spoken in china":        "Mandarin",
    "language spoken in russia":       "Russian",
    "language spoken in egypt":        "Arabic",
    "primary language of brazil":      "Portuguese",
}

# ===========================================================================
# Musical-instrument creators / composers
# ===========================================================================

COMPOSERS: dict[str, str] = {
    "beethoven instrument":     "piano",
    "instrument beethoven composed for": "piano",
    "mozart instrument":        "piano",
    "bach instrument":          "organ",
    "wagner art form":          "opera",
    "verdi art form":           "opera",
}


# Compact registry: relation -> dict.
REGISTRY: dict[str, dict[str, str]] = {
    "capital":      CAPITALS,
    "author":       AUTHORS,
    "painter":      PAINTERS,
    "date":         DATES,
    "first":        FIRSTS,
    "inventor":     INVENTORS,
    "science":      SCIENCE,
    "geography":    GEOGRAPHY,
    "planet":       PLANETS,
    "count":        COUNTS,
    "language":     LANGUAGES,
    "composer":     COMPOSERS,
}


def n_facts() -> int:
    return sum(len(d) for d in REGISTRY.values())


def lookup(relation: str, key: str) -> str | None:
    d = REGISTRY.get(relation)
    if d is None:
        return None
    k = key.strip().lower().rstrip("?.!,")
    return d.get(k)


def fuzzy_lookup(key: str) -> tuple[str, str] | None:
    """Try every relation table for a key. Returns (relation, value)."""
    k = key.strip().lower().rstrip("?.!,")
    for relation, table in REGISTRY.items():
        if k in table:
            return (relation, table[k])
    return None
