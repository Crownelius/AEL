#!/usr/bin/env python3
"""Interactive chat / one-shot prompt interface for the trained fant3+AEL model.

Each input is routed:
  - If it matches a known Q-pattern (define / hypernym / capital / etc.) →
    answered by AEL retrieval (WordNet + neo-WordNet facts). Works well.
  - Otherwise → fed to the trained fant3 LM for next-token completion.
    Quality depends entirely on how well-trained the checkpoint is. At low
    pretraining budgets the output will be fragmentary.

Three usage modes:

  1. CLI REPL (terminal):
       python pod/chat.py --ckpt pod_ckpt/final.pt

  2. One-shot from CLI:
       python pod/chat.py --ckpt pod_ckpt/final.pt --prompt "What is a dog?"

  3. Programmatic (Kaggle notebook cell):
       import sys; sys.path.insert(0, '/kaggle/working/AEL/pod')
       import chat
       chat.load('/kaggle/working/AEL/pod_ckpt/final.pt')
       print(chat.ask("What is the capital of France?"))
       print(chat.ask("Once upon a time"))
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
AEL_ROOT = THIS.parent.parent
FANT3_ROOT = AEL_ROOT.parent / "fant3"
for p in (str(FANT3_ROOT), str(AEL_ROOT / "src"), str(AEL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


_LOADED_CKPT: str | None = None


def load(ckpt_path: str | Path) -> None:
    """Load (or reload) the trained model from the given checkpoint path."""
    import ael.fant3_gen as fg
    global _LOADED_CKPT
    fg._MODEL = None  # force reload
    fg._ensure_loaded(ckpt_path=Path(ckpt_path))
    _LOADED_CKPT = str(ckpt_path)
    print(f"[chat] loaded checkpoint: {ckpt_path}")


def ask(
    prompt: str,
    max_new: int = 30,
    use_qa_router: bool = True,
    use_fant3_fallback: bool = True,
) -> str:
    """Single-turn response. Routes via qa.py (AEL retrieval) then falls back
    to trained-fant3 generation for free completions."""
    if _LOADED_CKPT is None:
        return "[chat] no model loaded — call chat.load(path/to/final.pt) first"

    if use_qa_router:
        from ael.qa import answer
        ans = answer(prompt, use_fant3_fallback=use_fant3_fallback)
        # qa.py routes Q-shaped questions to retrieval and "open completion"
        # style inputs to fant3 generation through use_fant3_fallback.
        return ans.text

    # Pure fant3 generation, no QA router.
    from ael.fant3_gen import complete
    return complete(prompt, max_new=max_new)


def repl(ckpt_path: str | Path) -> None:
    """Terminal REPL. Ctrl-D / 'exit' / 'quit' to stop."""
    load(ckpt_path)
    print("[chat] interactive mode — type 'exit' or Ctrl-D to quit")
    print("[chat] Q-shaped inputs route through AEL retrieval; open completions go to fant3 LM")
    print()
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] bye")
            return
        if not line:
            continue
        if line.lower() in ("exit", "quit", ":q"):
            print("[chat] bye")
            return
        try:
            response = ask(line)
            print(f"bot> {response}\n")
        except Exception as e:
            print(f"[chat] error: {type(e).__name__}: {e}\n")


def main():
    p = argparse.ArgumentParser()
    default_ckpt = AEL_ROOT / "pod_ckpt" / "final.pt"
    p.add_argument("--ckpt",  default=str(default_ckpt))
    p.add_argument("--prompt", default=None, help="single-shot; omit for REPL")
    p.add_argument("--max-new", type=int, default=30)
    p.add_argument("--no-qa-router", action="store_true",
                   help="bypass AEL routing; pure fant3 generation only")
    args = p.parse_args()

    if not Path(args.ckpt).exists():
        print(f"[chat] no checkpoint at {args.ckpt}", file=sys.stderr)
        return 1

    if args.prompt is not None:
        load(args.ckpt)
        print(ask(args.prompt, max_new=args.max_new,
                  use_qa_router=not args.no_qa_router))
        return 0

    repl(args.ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
