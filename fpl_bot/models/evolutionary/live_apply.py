"""Apply a trained genome (from train_model1.py) to the current live player
pool, producing a PlayerScore list the same as every other model -- so it
plugs into the same ILP optimizer as Models 2/3/4 for Team A's live squad.
"""

import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from fpl_bot.models.basic_stats import PlayerScore
from fpl_bot.models.evolutionary.genome import FEATURES, Genome
from fpl_bot.models.statistical_predictor import _availability_multiplier, build_live_feature_rows

GENOME_PATH = Path(__file__).parent.parent.parent / "data" / "model1_genome.pkl"


def load_best_genome() -> Genome:
    with open(GENOME_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["genome"]


def _normalize_rows(rows: list[dict]) -> list[dict]:
    """Same per-position min-max normalization used during training, applied
    to this one live snapshot instead of a historical panel."""
    by_position: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_position.setdefault(r["position"], []).append(i)

    normalized = [dict(r) for r in rows]
    for position, idxs in by_position.items():
        for feature in FEATURES:
            values = [rows[i][feature] for i in idxs]
            lo, hi = min(values), max(values)
            for i in idxs:
                normalized[i][f"n_{feature}"] = 0.5 if hi == lo else (rows[i][feature] - lo) / (hi - lo)
    return normalized


def predict_current_squad_scores(genome: Genome | None = None) -> list[PlayerScore]:
    genome = genome or load_best_genome()

    # crude position-average fallback, consistent in spirit with Model 2's cold start
    pos_avg_points = {"GK": 2.5, "DEF": 2.5, "MID": 2.5, "FWD": 2.5}
    pos_avg_minutes = {"GK": 60.0, "DEF": 60.0, "MID": 50.0, "FWD": 45.0}

    rows, meta = build_live_feature_rows(pos_avg_points, pos_avg_minutes)
    rows = _normalize_rows(rows)

    results = []
    for row, (pid, web_name, team_id, element_type, now_cost, status, chance_of_playing_next_round) in zip(rows, meta):
        raw_score = genome.score(row["position"], row)
        availability = _availability_multiplier(status, chance_of_playing_next_round)
        results.append(
            PlayerScore(
                player_id=pid,
                web_name=web_name,
                team_id=team_id,
                element_type=element_type,
                now_cost=now_cost,
                score=round(raw_score * availability, 4),
            )
        )

    results.sort(key=lambda p: p.score, reverse=True)
    return results
