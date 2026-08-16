"""End-to-end Phase 3 pipeline: refresh data -> Model 3 base scores ->
qualitative LLM review (BBC headlines + official injury/status notes) ->
ILP optimizer -> print the recommended squad, plus the LLM's rationale.

Run with:
    python -m fpl_bot.scripts.draft_squad_model4 --backend ollama
    python -m fpl_bot.scripts.draft_squad_model4 --backend anthropic
"""

import argparse
import io
import sys

from fpl_bot.ingest import ingest_bootstrap
from fpl_bot.optimizer.squad_optimizer import SquadResult, build_squad_result
from fpl_bot.qualitative.synthesizer import run_qualitative_review
from fpl_bot.scripts._display import lookup_team_names, print_result


def run(backend: str = "ollama", refresh: bool = True) -> SquadResult:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    if refresh:
        ingest_bootstrap.run()

    review = run_qualitative_review(backend)
    result = build_squad_result(review["scores"])

    print(f"MODEL 4 -- qualitative agent squad (backend: {backend})")
    print_result(result, lookup_team_names())

    print()
    print("QUALITATIVE NOTES")
    print("-" * 70)
    if review["overall_notes"]:
        print(review["overall_notes"])
    for note in review["notes"]:
        print(f"  {note.web_name}: {note.adjustment:+.0%}  -- {note.rationale}")
    if not review["notes"]:
        print("  (no news-driven adjustments this run)")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ollama", "anthropic"], default="ollama")
    parser.add_argument("--no-refresh", action="store_true")
    args = parser.parse_args()
    run(backend=args.backend, refresh=not args.no_refresh)
