"""Model 2 -- statistical predictor of expected FPL points per player per
gameweek, trained on 3 seasons of historical data.

Every feature is computed using only information available *before* the
gameweek being predicted (rolling/expanding stats, shifted by one game) so
there's no leakage from the outcome we're trying to predict.
"""

import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # noisy, non-actionable numexpr/bottleneck build warnings on this machine

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from fpl_bot import db

MODEL_PATH = Path(__file__).parent.parent / "data" / "model2_predictor.pkl"
FEATURE_COLUMNS = [
    "position",
    "was_home",
    "price",
    "rolling_points_3",
    "rolling_minutes_3",
    "team_goals_for_rate",
    "team_goals_against_rate",
    "opp_goals_for_rate",
    "opp_goals_against_rate",
]
TEST_SEASON = "2025-26"


def load_historical_df() -> pd.DataFrame:
    conn = db.get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM historical_gw", conn)
    finally:
        conn.close()
    # drop the short-lived 2024-25 "Assistant Manager" pseudo-position: not part
    # of the current GK/DEF/MID/FWD game structure, and too rare (322 rows) to model
    df = df[df["position"] != "AM"]

    # collapse genuine double-gameweeks (same element_id+round, two fixtures) into
    # one row per (season, element_id, round): FPL sums points/minutes across both
    # fixtures that gameweek, and every downstream step assumes one row per player-round
    df = df.sort_values(["season", "element_id", "round", "kickoff_time"])
    agg = df.groupby(["season", "element_id", "round"], as_index=False).agg(
        name=("name", "first"),
        position=("position", "first"),
        team=("team", "first"),
        opponent_team=("opponent_team", "first"),
        was_home=("was_home", "first"),
        kickoff_time=("kickoff_time", "first"),
        minutes=("minutes", "sum"),
        total_points=("total_points", "sum"),
        value=("value", "last"),
        team_h_score=("team_h_score", "first"),
        team_a_score=("team_a_score", "first"),
    )
    df = agg.sort_values(["season", "name", "round"]).reset_index(drop=True)
    return df


def _team_game_rates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team, round) with that team's rolling scoring
    rates, using only games strictly before `round`."""
    games = df.drop_duplicates(subset=["season", "team", "round"]).copy()
    games["goals_for"] = np.where(games["was_home"] == 1, games["team_h_score"], games["team_a_score"])
    games["goals_against"] = np.where(games["was_home"] == 1, games["team_a_score"], games["team_h_score"])
    games = games.sort_values(["season", "team", "round"])

    grp = games.groupby(["season", "team"])
    # shift(1) so the current game's own result isn't included in its own rate
    games["team_goals_for_rate"] = grp["goals_for"].transform(lambda s: s.shift(1).expanding().mean())
    games["team_goals_against_rate"] = grp["goals_against"].transform(lambda s: s.shift(1).expanding().mean())

    league_avg_for = games["goals_for"].mean()
    league_avg_against = games["goals_against"].mean()
    games["team_goals_for_rate"] = games["team_goals_for_rate"].fillna(league_avg_for)
    games["team_goals_against_rate"] = games["team_goals_against_rate"].fillna(league_avg_against)

    return games[["season", "team", "round", "team_goals_for_rate", "team_goals_against_rate"]]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    team_rates = _team_game_rates(df)

    df = df.merge(team_rates, on=["season", "team", "round"], how="left")
    opp_rates = team_rates.rename(
        columns={
            "team": "opponent_team",
            "team_goals_for_rate": "opp_goals_for_rate",
            "team_goals_against_rate": "opp_goals_against_rate",
        }
    )
    df = df.merge(opp_rates, on=["season", "opponent_team", "round"], how="left")

    league_avg_for = team_rates["team_goals_for_rate"].mean()
    league_avg_against = team_rates["team_goals_against_rate"].mean()
    df["opp_goals_for_rate"] = df["opp_goals_for_rate"].fillna(league_avg_for)
    df["opp_goals_against_rate"] = df["opp_goals_against_rate"].fillna(league_avg_against)

    # grouped by element_id (the season-scoped numeric player id), not name --
    # names collide often enough (duplicate real names, or literal duplicate
    # source rows) to corrupt a per-player rolling average if used as the key
    df = df.sort_values(["season", "element_id", "round"])
    grp = df.groupby(["season", "element_id"])
    df["rolling_points_3"] = grp["total_points"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["rolling_minutes_3"] = grp["minutes"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    # cold start for a player's first game(s) in a season: position-average rather than 0,
    # since 0 would look like "nailed-on bench player" rather than "no data yet"
    pos_avg_points = df.groupby("position")["total_points"].transform("mean")
    pos_avg_minutes = df.groupby("position")["minutes"].transform("mean")
    df["rolling_points_3"] = df["rolling_points_3"].fillna(pos_avg_points)
    df["rolling_minutes_3"] = df["rolling_minutes_3"].fillna(pos_avg_minutes)

    df["price"] = df["value"] / 10.0
    return df


def train_and_backtest() -> dict:
    raw = load_historical_df()
    feats = build_features(raw)

    train = feats[feats["season"] != TEST_SEASON].copy()
    test = feats[feats["season"] == TEST_SEASON].copy()

    X_train = train[FEATURE_COLUMNS].copy()
    X_train["position"] = X_train["position"].astype("category")
    y_train = train["total_points"]

    X_test = test[FEATURE_COLUMNS].copy()
    X_test["position"] = pd.Categorical(X_test["position"], categories=X_train["position"].cat.categories)
    y_test = test["total_points"]

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        categorical_features=["position"],
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)
    model_mae = mean_absolute_error(y_test, preds)

    naive_baseline = test["rolling_points_3"]
    naive_mae = mean_absolute_error(y_test, naive_baseline)

    pos_avg_points = train.groupby("position")["total_points"].mean().to_dict()
    pos_avg_minutes = train.groupby("position")["minutes"].mean().to_dict()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "position_categories": list(X_train["position"].cat.categories),
                "pos_avg_points": pos_avg_points,
                "pos_avg_minutes": pos_avg_minutes,
            },
            f,
        )

    return {
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "train_rows": len(train),
        "test_rows": len(test),
    }


def get_backtest_arrays() -> pd.DataFrame:
    """Re-predicts on the held-out test season using the already-trained,
    saved model (no retraining) -- for dashboard scatter/feature-importance
    plots. Returns one row per player-gameweek: name, position, actual
    points, predicted points, naive-baseline prediction."""
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    position_categories = bundle["position_categories"]

    raw = load_historical_df()
    feats = build_features(raw)
    test = feats[feats["season"] == TEST_SEASON].copy()

    X_test = test[FEATURE_COLUMNS].copy()
    X_test["position"] = pd.Categorical(X_test["position"], categories=position_categories)
    preds = np.clip(model.predict(X_test), 0, None)

    return pd.DataFrame(
        {
            "name": test["name"].values,
            "position": test["position"].values,
            "round": test["round"].values,
            "actual": test["total_points"].values,
            "predicted": preds,
            "naive_baseline": test["rolling_points_3"].values,
        }
    )


def load_model_and_test_matrix():
    """For dashboard diagnostics (e.g. permutation importance) that need the
    raw model + feature matrix, not just predictions."""
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    position_categories = bundle["position_categories"]

    raw = load_historical_df()
    feats = build_features(raw)
    test = feats[feats["season"] == TEST_SEASON].copy()

    X_test = test[FEATURE_COLUMNS].copy()
    X_test["position"] = pd.Categorical(X_test["position"], categories=position_categories)
    y_test = test["total_points"]
    return model, X_test, y_test


ELEMENT_TYPE_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
LAST_COMPLETED_SEASON = "2025-26"
LAST_SEASON_GAMES = 38


def _full_season_team_rates(season: str) -> tuple[dict[str, tuple[float, float]], float, float]:
    conn = db.get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM historical_gw WHERE season = ?", conn, params=[season])
    finally:
        conn.close()

    games = df.drop_duplicates(subset=["team", "round"]).copy()
    games["goals_for"] = np.where(games["was_home"] == 1, games["team_h_score"], games["team_a_score"])
    games["goals_against"] = np.where(games["was_home"] == 1, games["team_a_score"], games["team_h_score"])
    by_team = games.groupby("team")[["goals_for", "goals_against"]].mean()

    rates = {team: (row["goals_for"], row["goals_against"]) for team, row in by_team.iterrows()}
    league_avg_for = by_team["goals_for"].mean()
    league_avg_against = by_team["goals_against"].mean()
    return rates, league_avg_for, league_avg_against


def _availability_multiplier(status: str, chance_of_playing_next_round) -> float:
    from fpl_bot.models.basic_stats import STATUS_MULTIPLIER

    mult = STATUS_MULTIPLIER.get(status, 1.0)
    if chance_of_playing_next_round is not None:
        mult = min(mult, chance_of_playing_next_round / 100.0)
    return mult


def build_live_feature_rows(pos_avg_points: dict, pos_avg_minutes: dict) -> tuple[list[dict], list[tuple]]:
    """Pre-gameweek feature rows for every current player ahead of the next
    (as-yet-unplayed) gameweek, using last season's rates as the cold-start
    prior when there's no in-season history yet. Shared by Model 2 (feeds a
    trained ML model) and Model 1 (feeds an evolved genome's weights) --
    same inputs, different scoring function on top.

    Returns (rows, meta) where meta is a parallel list of
    (player_id, web_name, team_id, element_type, now_cost, status, chance_of_playing_next_round).
    """
    team_rates, league_avg_for, league_avg_against = _full_season_team_rates(LAST_COMPLETED_SEASON)

    conn = db.get_connection()
    try:
        next_gw = conn.execute("SELECT id FROM gameweeks WHERE is_next = 1").fetchone()
        if next_gw is None:
            next_gw = conn.execute(
                "SELECT id FROM gameweeks WHERE finished = 0 ORDER BY id LIMIT 1"
            ).fetchone()
        gw_id = next_gw[0]

        fixtures = conn.execute(
            "SELECT team_h, team_a FROM fixtures WHERE event = ?", (gw_id,)
        ).fetchall()
        teams = {row[0]: row[1] for row in conn.execute("SELECT id, name FROM teams")}

        players = conn.execute(
            """
            SELECT id, web_name, team_id, element_type, now_cost, points_per_game,
                   minutes, status, chance_of_playing_next_round
            FROM players
            """
        ).fetchall()
    finally:
        conn.close()

    # team_id -> (opponent_team_id, was_home)
    fixture_map: dict[int, tuple[int, int]] = {}
    for team_h, team_a in fixtures:
        fixture_map[team_h] = (team_a, 1)
        fixture_map[team_a] = (team_h, 0)

    rows = []
    meta = []
    for (pid, web_name, team_id, element_type, now_cost, points_per_game,
         minutes, status, chance_of_playing_next_round) in players:
        if element_type not in ELEMENT_TYPE_TO_POSITION or team_id not in fixture_map:
            continue  # no GW1 fixture (e.g. blank gameweek) -- nothing to predict

        position = ELEMENT_TYPE_TO_POSITION[element_type]
        opp_team_id, was_home = fixture_map[team_id]
        team_name = teams.get(team_id)
        opp_name = teams.get(opp_team_id)

        team_for, team_against = team_rates.get(team_name, (league_avg_for, league_avg_against))
        opp_for, opp_against = team_rates.get(opp_name, (league_avg_for, league_avg_against))

        rolling_points_3 = points_per_game if points_per_game else pos_avg_points.get(position, 2.0)
        rolling_minutes_3 = (minutes / LAST_SEASON_GAMES) if minutes else pos_avg_minutes.get(position, 45.0)

        rows.append(
            {
                "position": position,
                "was_home": was_home,
                "price": now_cost / 10.0,
                "rolling_points_3": rolling_points_3,
                "rolling_minutes_3": rolling_minutes_3,
                "team_goals_for_rate": team_for,
                "team_goals_against_rate": team_against,
                "opp_goals_for_rate": opp_for,
                "opp_goals_against_rate": opp_against,
            }
        )
        meta.append((pid, web_name, team_id, element_type, now_cost, status, chance_of_playing_next_round))

    return rows, meta


def predict_next_gameweek():
    """Predict expected points for every current player for the next
    gameweek using the trained gradient-boosted model."""
    from fpl_bot.models.basic_stats import PlayerScore

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    position_categories = bundle["position_categories"]
    pos_avg_points = bundle["pos_avg_points"]
    pos_avg_minutes = bundle["pos_avg_minutes"]

    rows, meta = build_live_feature_rows(pos_avg_points, pos_avg_minutes)

    X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    X["position"] = pd.Categorical(X["position"], categories=position_categories)
    preds = np.clip(model.predict(X), 0, None)

    results = []
    for (pid, web_name, team_id, element_type, now_cost, status, chance_of_playing_next_round), pred in zip(meta, preds):
        availability = _availability_multiplier(status, chance_of_playing_next_round)
        results.append(
            PlayerScore(
                player_id=pid,
                web_name=web_name,
                team_id=team_id,
                element_type=element_type,
                now_cost=now_cost,
                score=round(float(pred) * availability, 4),
            )
        )

    results.sort(key=lambda p: p.score, reverse=True)
    return results


if __name__ == "__main__":
    results = train_and_backtest()
    print(f"Trained on {results['train_rows']} rows (2023-24 + 2024-25), "
          f"backtested on {results['test_rows']} rows ({TEST_SEASON}).")
    print(f"Model MAE:  {results['model_mae']:.3f} points/player/gameweek")
    print(f"Naive MAE:  {results['naive_mae']:.3f} points/player/gameweek (rolling 3-game average baseline)")
    improvement = (1 - results["model_mae"] / results["naive_mae"]) * 100
    print(f"Improvement over naive baseline: {improvement:.1f}%")
