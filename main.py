"""Command-line entry point for the QTMD multi-agent dialogue system."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a QTMD multi-agent dialogue from the command line."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Should the public be given more freedom to roam?",
        help="Debate question given to all three agents.",
    )
    parser.add_argument(
        "--use_R",
        type=int,
        choices=(0, 1),
        default=1,
        help="Enable (1) or disable (0) QTMD response rules.",
    )
    parser.add_argument(
        "--rule_mode",
        choices=("light", "struct"),
        default="light",
        help="Rule template used when --use_R=1.",
    )
    parser.add_argument(
        "--adaptive",
        type=int,
        choices=(0, 1),
        default=1,
        help="Enable (1) or disable (0) adaptive QTMD weights.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of dialogue rounds to run.",
    )
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # RAG paths in the original project are relative to the repository root.
    os.chdir(PROJECT_ROOT)
    os.environ.setdefault(
        "HF_HOME",
        str(PROJECT_ROOT / ".cache" / "huggingface"),
    )

    # Keep the heavyweight ML imports out of `--help` and argument validation.
    from dialogue_runner import run_dialogue

    results = list(
        run_dialogue(
            query=args.query,
            use_R=bool(args.use_R),
            rule_mode=args.rule_mode,
            adaptive_weight=bool(args.adaptive),
            rounds=args.rounds,
        )
    )

    print(f"\nDialogue complete: {len(results)} agent turns.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
