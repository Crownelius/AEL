# AEL — Apollonian Embedding Lattice

A small language-model architecture that stores knowledge as **geometry** (the structure of an Apollonian circle packing) instead of cramming it into billions of parameters. The bet: a tiny model (10–50 M params) plus a geometric knowledge substrate can beat much larger dense models on tasks where structure matters.

So far it beats **GPT-2 small (117 M)** by ~4× and **Qwen 2.5-1.5B (1.5 B)** by ~1.6× on a mixed WordNet + open-trivia + completion benchmark, at 1/12 and 1/154 of their parameter counts.

---

## What's inside

Each piece below is a real, working module under `src/ael/`.

| Module | One-line description |
|---|---|
| `descartes.py` | Solves Descartes' circle theorem to find Apollonian companions. |
| `gasket.py` | Generates the Apollonian gasket (the fractal of mutually-tangent circles). |
| `descartes_3d.py` | Lifts the gasket into Minkowski space `Cl(3,1)` so each circle becomes a 4-vector. |
| `uhs.py` / `cone_uhs.py` | Upper-half-space hyperbolic model + smooth cone attention. |
| `hopf.py` / `hopf_lift.py` / `cone_hopf.py` | Hopf-fibration lift to S³, with cone attention on the 2-sphere base. |
| `embed.py` / `embed_v2.py` | Map WordNet synsets to circles in the gasket (tangency-based + region-budget). |
| `placement_head.py` / `placement_head_hopf.py` | Tiny PyTorch heads that learn where each concept should sit. |
| `prime_addr.py` / `primes.py` | Twin-prime addressing — every circle gets a name from the prime spectrum. |
| `wordnet_data.py` / `corpus.py` | WordNet loader + training-text builder. |
| `qa.py` / `facts.py` | The QA frontend: ~17 question intents + a 200-fact "neo-WordNet" world-knowledge layer. |
| `fant3_bridge.py` | Drop-in replacement for fant3's `SpinorApollonianMemory`. |
| `fant3_gen.py` | Inference wrapper for the trained 50M model. |
| `benchmark*.py` / `benchmark_eval.py` | Head-to-head benchmark vs GPT-2 and Qwen 2.5. |
| `fineweb_pipeline.py` / `openwebtext_pipeline.py` | Stream-tokenize HuggingFace datasets for training. |

## Experiments (`experiments/`)

Numbered scripts that run the major experiments end-to-end — start at `01_` and read forward to follow the project's evolution from WordNet retrieval through Hopf placement to language-model pretraining on OpenWebText.

## Tests (`tests/`)

`test_descartes_3d.py`, `test_gasket.py`, `test_hopf.py` — each one verifies the math (Descartes preserved under reflection, fibers round-trip through the Hopf map, etc.). Run with `python tests/test_*.py`.

---

## The architecture in one paragraph

A 50 M-parameter language model (forked from [fant3](https://github.com/Crownelius/fant3)) is augmented with an "AEL memory" — instead of FAISS or a normal key-value cache, every stored vector is **snapped to a specific circle in an Apollonian gasket**. The gasket lives in 3D upper-half-space; retrieval is a **cone attention** on the hyperboloid (smooth, differentiable aperture parameter). Each circle is **named by a twin-prime pair** drawn from the project's twin-prime engine, so addressing is exact and collision-free. A separate small **`facts.py`** layer adds structured world-knowledge (capitals, dates, authors) the same way — every fact gets a circle.

## Why this might work

A typical 1B-parameter language model uses most of its weights to *memorize* facts that already live as discrete graph edges (in WordNet, Wikidata, etc.). AEL externalizes those edges into a geometric structure the model can *query*, freeing the model itself to learn just routing and composition — much less data-hungry. The Apollonian gasket is the substrate because it has the right fractal density to host a hierarchical concept space without the points colliding.

## Status

Active research workspace. Some pieces (gasket math, Hopf cone, fact retrieval) are solid and tested; others (gradient-accumulated pretraining on OpenWebText, OpenBMB SFT phase) are in progress.
