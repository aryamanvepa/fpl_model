"""Feature prep for Model 1's genome scoring.

Reuses Model 2's leakage-free pre-gameweek features (rolling form, team/
opponent scoring rates, price, home/away) and adds per-gameweek,
per-position min-max normalization so a genome's weights are comparable
across features of very different natural scale.
"""

import warnings

warnings.filterwarnings("ignore")

from fpl_bot.models.statistical_predictor import build_features, load_historical_df

NUMERIC_FEATURES = [
    "was_home",
    "price",
    "rolling_points_3",
    "rolling_minutes_3",
    "team_goals_for_rate",
    "team_goals_against_rate",
    "opp_goals_for_rate",
    "opp_goals_against_rate",
    "future_opp_goals_for_rate",
    "future_opp_goals_against_rate",
    "future_fixtures_count",
]


def _normalize_group(group):
    for col in NUMERIC_FEATURES:
        lo, hi = group[col].min(), group[col].max()
        group[f"n_{col}"] = 0.5 if hi == lo else (group[col] - lo) / (hi - lo)
    return group


def build_normalized_features(seasons: list[str]):
    raw = load_historical_df()
    raw = raw[raw["season"].isin(seasons)]
    feats = build_features(raw)
    feats = feats.groupby(["season", "round", "position"], group_keys=False).apply(_normalize_group)
    return feats
