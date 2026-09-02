"""Turns a squad (however it was obtained -- tracked team_state, or a
freshly parsed screenshot) plus a model's current scores into ranked,
realistic transfer suggestions. Pure local computation, no API calls.

Deliberately suggests single swaps ranked by score gain, not a full
re-optimization diff: FPL only gives 1-2 free transfers before -4 penalties,
so "here's the optimal 15 from scratch" isn't an actionable recommendation
the way "swap X for Y" is.
"""

import difflib

from fpl_bot import db
from fpl_bot.ensemble.combine import percentile_ranks
from fpl_bot.models.basic_stats import PlayerScore

BUDGET = 1000


def load_players_meta() -> dict[int, dict]:
    conn = db.get_connection()
    try:
        rows = conn.execute("SELECT id, web_name, now_cost, element_type, team_id FROM players").fetchall()
    finally:
        conn.close()
    return {r[0]: {"web_name": r[1], "now_cost": r[2], "element_type": r[3], "team_id": r[4]} for r in rows}


def resolve_names_to_ids(names: list[str], players_meta: dict[int, dict] | None = None) -> tuple[dict[str, int], list[str]]:
    """Fuzzy-matches free-text names (as read off a screenshot) against the
    current players table. Returns (name -> player_id, [names that didn't
    resolve confidently enough to trust automatically])."""
    players_meta = players_meta or load_players_meta()
    by_name = {meta["web_name"]: pid for pid, meta in players_meta.items()}
    all_names = list(by_name.keys())

    resolved: dict[str, int] = {}
    unresolved: list[str] = []
    for name in names:
        if name in by_name:
            resolved[name] = by_name[name]
            continue
        matches = difflib.get_close_matches(name, all_names, n=1, cutoff=0.75)
        if matches:
            resolved[name] = by_name[matches[0]]
        else:
            unresolved.append(name)
    return resolved, unresolved


def best_single_swaps(
    squad_ids: list[int],
    scores: list[PlayerScore],
    players_meta: dict[int, dict] | None = None,
    top_n: int = 3,
    budget: int = BUDGET,
) -> list[dict]:
    """Ranked list of the best single (out, in) swaps -- same position,
    affordable within the squad's current total value.

    Ranked and reported by **percentile-rank gain** (0-1 scale, this model's
    worst pick to its best), not raw score difference: the four models'
    scores live on incompatible scales (Model 2 = predicted points ~0-8,
    Model 1 = a genome-weighted linear score ~-3 to 3, Model 3 = a 0-1
    composite), so a raw "+3.27" from one model and "+0.74" from another
    aren't comparable -- same reason the ensemble combines ranks, not scores.
    `raw_gain` is also included per swap, in case that model's native unit
    (e.g. Model 2's is literally predicted points) is independently useful.

    Each item: {"out_id", "out_name", "in_id", "in_name", "gain", "raw_gain", ...}.
    """
    players_meta = players_meta or load_players_meta()
    raw_score_map = {p.player_id: p.score for p in scores}
    rank_map = percentile_ranks(scores)
    squad_set = set(squad_ids)
    total_value = sum(players_meta[i]["now_cost"] for i in squad_ids if i in players_meta)

    swaps = []
    for out_id in squad_ids:
        if out_id not in players_meta:
            continue
        out_meta = players_meta[out_id]
        out_rank = rank_map.get(out_id, 0.0)
        budget_after_sale = total_value - out_meta["now_cost"]

        candidates = [
            (pid, m) for pid, m in players_meta.items()
            if pid not in squad_set
            and m["element_type"] == out_meta["element_type"]
            and (budget_after_sale + m["now_cost"]) <= budget
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda x: -rank_map.get(x[0], -1))
        in_id, in_meta = candidates[0]
        rank_gain = rank_map.get(in_id, 0.0) - out_rank
        raw_gain = raw_score_map.get(in_id, 0.0) - raw_score_map.get(out_id, 0.0)
        swaps.append(
            {
                "out_id": out_id, "out_name": out_meta["web_name"], "out_cost": out_meta["now_cost"],
                "in_id": in_id, "in_name": in_meta["web_name"], "in_cost": in_meta["now_cost"],
                "gain": rank_gain, "raw_gain": raw_gain,
            }
        )

    swaps.sort(key=lambda s: -s["gain"])
    return swaps[:top_n]
