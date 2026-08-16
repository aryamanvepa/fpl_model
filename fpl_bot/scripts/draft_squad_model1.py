"""End-to-end Phase 4 pipeline: refresh data -> evolved genome's scoring
-> ILP optimizer -> print the recommended squad for Model 1's live team.

Requires a trained genome (run `python -m fpl_bot.scripts.train_model1`
first if fpl_bot/data/model1_genome.pkl doesn't exist yet).

Run with: python -m fpl_bot.scripts.draft_squad_model1
"""

import io
import sys

from fpl_bot.ingest import ingest_bootstrap
from fpl_bot.models.evolutionary.live_apply import predict_current_squad_scores
from fpl_bot.optimizer.squad_optimizer import SquadResult, build_squad_result
from fpl_bot.scripts._display import lookup_team_names, print_result


def run(refresh: bool = True) -> SquadResult:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    if refresh:
        ingest_bootstrap.run()

    scores = predict_current_squad_scores()
    result = build_squad_result(scores)

    print("MODEL 1 -- evolutionary strategy squad")
    print_result(result, lookup_team_names())
    return result


if __name__ == "__main__":
    run()
