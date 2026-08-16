"""End-to-end Phase 1 pipeline: refresh data -> Model 3 scores -> ILP optimizer
-> print the recommended initial 15-man squad, starting XI, captain and bench.

Run with: python -m fpl_bot.scripts.draft_squad
"""

import io
import sys

from fpl_bot.ingest import ingest_bootstrap
from fpl_bot.models.basic_stats import compute_scores
from fpl_bot.optimizer.squad_optimizer import SquadResult, build_squad_result
from fpl_bot.scripts._display import lookup_team_names, print_result


def run(refresh: bool = True) -> SquadResult:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    if refresh:
        ingest_bootstrap.run()

    scores = compute_scores()
    result = build_squad_result(scores)

    print("MODEL 3 -- basic stats squad")
    print_result(result, lookup_team_names())
    return result


if __name__ == "__main__":
    run()
