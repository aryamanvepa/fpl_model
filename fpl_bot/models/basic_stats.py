"""Model 3 -- transparent, no-ML scoring formula (points-per-million, form,
underlying attacking/defensive output), weighted per position.

This is deliberately simple: it's the fast-to-ship baseline the other models
get backtested against, not the final word.
"""

from dataclasses import dataclass

from fpl_bot import db

# Position ids from the `positions` table: 1=GK, 2=DEF, 3=MID, 4=FWD
POSITION_WEIGHTS = {
    1: {"pts_rate": 0.30, "value": 0.20, "ict": 0.10, "defense": 0.40, "attack": 0.00},
    2: {"pts_rate": 0.25, "value": 0.20, "ict": 0.15, "defense": 0.30, "attack": 0.10},
    3: {"pts_rate": 0.25, "value": 0.20, "ict": 0.20, "defense": 0.05, "attack": 0.30},
    4: {"pts_rate": 0.25, "value": 0.20, "ict": 0.15, "defense": 0.00, "attack": 0.40},
}

# Unavailable/major-doubt statuses from the FPL API: a=available, d=doubtful,
# i=injured, s=suspended, u=unavailable (left club), n=not eligible this GW.
STATUS_MULTIPLIER = {"a": 1.0, "d": 0.75, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}


@dataclass
class PlayerScore:
    player_id: int
    web_name: str
    team_id: int
    element_type: int
    now_cost: int
    score: float


def _min_max_normalize(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _availability_multiplier(status: str, chance_of_playing_next_round) -> float:
    mult = STATUS_MULTIPLIER.get(status, 1.0)
    if chance_of_playing_next_round is not None:
        mult = min(mult, chance_of_playing_next_round / 100.0)
    return mult


def compute_scores(weights_override: dict | None = None) -> list[PlayerScore]:
    """weights_override lets a caller (e.g. the dashboard's live sliders)
    supply its own {element_type: {feature: weight}} dict instead of the
    default POSITION_WEIGHTS, without duplicating the scoring logic."""
    all_weights = weights_override or POSITION_WEIGHTS
    conn = db.get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, web_name, team_id, element_type, now_cost, status,
                   chance_of_playing_next_round, total_points, points_per_game,
                   form, ict_index, expected_goal_involvements, expected_goals_conceded
            FROM players
            """
        ).fetchall()
    finally:
        conn.close()

    by_position: dict[int, list[tuple]] = {}
    for row in rows:
        by_position.setdefault(row[3], []).append(row)

    results: list[PlayerScore] = []
    for element_type, players in by_position.items():
        weights = all_weights.get(element_type, all_weights[3])

        pts_rate = {r[0]: (r[8] or 0.0) for r in players}
        value = {r[0]: (r[7] / (r[4] / 10)) if r[4] else 0.0 for r in players}
        ict = {r[0]: (r[10] or 0.0) for r in players}
        attack = {r[0]: (r[11] or 0.0) for r in players}
        defense = {r[0]: -(r[12] or 0.0) for r in players}

        n_pts_rate = _min_max_normalize(pts_rate)
        n_value = _min_max_normalize(value)
        n_ict = _min_max_normalize(ict)
        n_attack = _min_max_normalize(attack)
        n_defense = _min_max_normalize(defense)

        for r in players:
            pid = r[0]
            raw_score = (
                weights["pts_rate"] * n_pts_rate[pid]
                + weights["value"] * n_value[pid]
                + weights["ict"] * n_ict[pid]
                + weights["attack"] * n_attack[pid]
                + weights["defense"] * n_defense[pid]
            )
            availability = _availability_multiplier(r[5], r[6])
            results.append(
                PlayerScore(
                    player_id=pid,
                    web_name=r[1],
                    team_id=r[2],
                    element_type=element_type,
                    now_cost=r[4],
                    score=round(raw_score * availability, 4),
                )
            )

    results.sort(key=lambda p: p.score, reverse=True)
    return results


if __name__ == "__main__":
    for p in compute_scores()[:20]:
        print(p)
