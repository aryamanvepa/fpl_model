"""The "environment" a genome plays against: replay one historical season
gameweek by gameweek using real results, letting the genome's policy pick an
initial squad, then decide transfers and captaincy each week using only
information available before that gameweek. Total points (minus hit
penalties) is the fitness score -- this is the whole game the genetic
algorithm is trying to get better at, generation over generation.

Players are tracked by element_id (the season-scoped numeric FPL id), not
name -- names collide often enough (two different players sharing a name,
or literal duplicate source rows) to corrupt squad tracking if used as the key.

Known simplifications (v1): at most one transfer evaluated per week (not two),
free transfers cap at 2 (not the newer 5-stack rule), no bench autosubs, no
chips (wildcard/free hit/bench boost/triple captain). All real, worth
revisiting, but this is enough to prove the evolutionary approach works at
all before adding more moving parts.
"""

import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np

from fpl_bot.models.basic_stats import PlayerScore
from fpl_bot.models.evolutionary.genome import FEATURES, POSITIONS, Genome
from fpl_bot.optimizer.squad_optimizer import pick_squad

BUDGET = 1000
POSITION_TO_ELEMENT_TYPE = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
STARTING_XI_MINS = {"DEF": 3, "MID": 2, "FWD": 1}
STARTING_XI_MAXS = {"DEF": 5, "MID": 5, "FWD": 3}


def _score_round(genome: Genome, round_df):
    """Adds a 'genome_score' column, vectorized per position group."""
    round_df = round_df.copy()
    scores = np.zeros(len(round_df))
    feature_cols = [f"n_{f}" for f in FEATURES]
    for pos in POSITIONS:
        mask = (round_df["position"] == pos).values
        if not mask.any():
            continue
        w = np.array([genome.weights[pos][f] for f in FEATURES])
        X = round_df.loc[mask, feature_cols].values
        scores[mask] = X @ w
    round_df["genome_score"] = scores
    return round_df


def _initial_squad(genome: Genome, round0_df) -> tuple[list[int], int]:
    scored = _score_round(genome, round0_df)
    records = scored.reset_index(drop=True).to_dict("records")
    player_scores = [
        PlayerScore(
            player_id=r["element_id"],
            web_name=r["name"],
            team_id=r["team"],
            element_type=POSITION_TO_ELEMENT_TYPE[r["position"]],
            now_cost=int(r["value"]),
            score=r["genome_score"],
        )
        for r in records
    ]
    squad = pick_squad(player_scores, budget=BUDGET)
    ids = [p.player_id for p in squad]
    value = sum(p.now_cost for p in squad)
    return ids, value


def _pick_starting_xi(squad_ids: list[int], round_df) -> tuple[list[int], int | None]:
    present = [i for i in squad_ids if i in round_df.index]
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for i in present:
        by_pos[round_df.loc[i, "position"]].append(i)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda i: round_df.loc[i, "genome_score"], reverse=True)

    if not by_pos["GK"]:
        return [], None

    starters = [by_pos["GK"][0]]
    used = {"DEF": 0, "MID": 0, "FWD": 0}
    for pos, mn in STARTING_XI_MINS.items():
        take = by_pos[pos][:mn]
        starters += take
        used[pos] = len(take)

    remaining = []
    for pos in ["DEF", "MID", "FWD"]:
        remaining += [(i, pos) for i in by_pos[pos][used[pos]:]]
    remaining.sort(key=lambda x: round_df.loc[x[0], "genome_score"], reverse=True)

    slots_left = 11 - len(starters)
    for i, pos in remaining:
        if slots_left <= 0:
            break
        if used[pos] < STARTING_XI_MAXS[pos]:
            starters.append(i)
            used[pos] += 1
            slots_left -= 1

    captain = max(starters, key=lambda i: round_df.loc[i, "genome_score"])
    return starters, captain


def _best_single_transfer(genome: Genome, squad_ids: list[int], squad_value: int, round_df):
    candidates_by_pos = {}
    for pos in POSITIONS:
        sub = round_df[round_df["position"] == pos].sort_values("genome_score", ascending=False)
        candidates_by_pos[pos] = list(
            zip(sub.index.tolist(), sub["genome_score"].tolist(), sub["value"].tolist(), sub["team"].tolist())
        )

    squad_set = set(squad_ids)
    club_counts = Counter(round_df.loc[i, "team"] for i in squad_ids if i in round_df.index)

    best = None  # (gain, out_id, in_id, in_value, in_team, out_value)
    for out_id in squad_ids:
        if out_id not in round_df.index:
            continue
        out_row = round_df.loc[out_id]
        pos = out_row["position"]
        budget_after_sale = squad_value - out_row["value"]
        out_team = out_row["team"]

        for in_id, in_score, in_value, in_team in candidates_by_pos[pos]:
            if in_id in squad_set:
                continue
            if budget_after_sale + in_value > BUDGET:
                continue
            if in_team != out_team and club_counts.get(in_team, 0) >= 3:
                continue
            gain = in_score - out_row["genome_score"]
            if best is None or gain > best[0]:
                best = (gain, out_id, in_id, in_value, in_team, out_row["value"])
            break  # sorted descending -> first feasible is this slot's best option

    return best


def _apply_weekly_transfer(genome: Genome, squad_ids, squad_value, round_df, free_transfers):
    best = _best_single_transfer(genome, squad_ids, squad_value, round_df)
    if best is None:
        return squad_ids, squad_value, min(free_transfers + 1, 2), 0

    gain, out_id, in_id, in_value, in_team, out_value = best

    if free_transfers >= 1 and gain > genome.transfer_threshold:
        new_squad = [in_id if i == out_id else i for i in squad_ids]
        return new_squad, squad_value - out_value + in_value, free_transfers - 1, 0
    if gain > genome.hit_threshold:
        new_squad = [in_id if i == out_id else i for i in squad_ids]
        return new_squad, squad_value - out_value + in_value, free_transfers, 1

    return squad_ids, squad_value, min(free_transfers + 1, 2), 0


def simulate_season(genome: Genome, feats_season) -> dict:
    rounds = sorted(feats_season["round"].unique())
    round0 = feats_season[feats_season["round"] == rounds[0]]
    squad_ids, squad_value = _initial_squad(genome, round0)

    free_transfers = 1
    total_points = 0
    hits_taken = 0
    history = []

    for rnd in rounds:
        round_df = feats_season[feats_season["round"] == rnd].set_index("element_id")
        round_df = _score_round(genome, round_df)

        if rnd != rounds[0]:
            squad_ids, squad_value, free_transfers, hit = _apply_weekly_transfer(
                genome, squad_ids, squad_value, round_df, free_transfers
            )
            hits_taken += hit

        starters, captain = _pick_starting_xi(squad_ids, round_df)
        week_points = 0
        for i in starters:
            pts = round_df.loc[i, "total_points"]
            if i == captain:
                pts *= 2
            week_points += pts
        total_points += week_points
        history.append(week_points)

    return {
        "total_points": total_points - 4 * hits_taken,
        "raw_points": total_points,
        "hits_taken": hits_taken,
        "history": history,
    }
