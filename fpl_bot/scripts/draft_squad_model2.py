"""End-to-end Phase 2 pipeline: refresh data -> Model 2 statistical predictor
-> ILP optimizer -> print the recommended squad for the next gameweek.

Requires a trained model (run `python -m fpl_bot.models.statistical_predictor`
first if fpl_bot/data/model2_predictor.pkl doesn't exist yet).

Run with: python -m fpl_bot.scripts.draft_squad_model2
"""

import io
import sys

from fpl_bot.ingest import ingest_bootstrap
from fpl_bot.models.statistical_predictor import predict_next_gameweek
from fpl_bot.optimizer.squad_optimizer import SquadResult, build_squad_result
from fpl_bot.scripts._display import lookup_team_names, print_result


def run(refresh: bool = True) -> SquadResult:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    if refresh:
        ingest_bootstrap.run()

    scores = predict_next_gameweek()
    result = build_squad_result(scores)

    print("MODEL 2 -- statistical predictor squad")
    print_result(result, lookup_team_names())
    return result


if __name__ == "__main__":
    run()
