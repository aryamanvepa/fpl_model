"""Last-N-gameweek form carried over from last completed season.

Why this exists: early in a new season the live `players` table holds only a
handful of matches. At GW3 that's ~2 games -- far too small a sample to rank
600 players on, and it makes whoever happened to return early look elite.
Blending in a real chunk of last season (default: its last 30 gameweeks,
which skips the noisy opening stretch and weights the run-in that best
reflects a settled role) gives a much sturdier prior.

Players are matched across seasons by "first_name second_name", the only
bridge available -- the historical dataset's per-season numeric ids are not
stable across seasons. Roughly three-quarters of the current pool matches;
new signings and promoted-club players simply have no history, and callers
get None for them rather than a fabricated value.
"""

from fpl_bot import db

LAST_COMPLETED_SEASON = "2025-26"
SEASON_LENGTH = 38
DEFAULT_LOOKBACK_GWS = 30


def last_n_gw_form(
    season: str = LAST_COMPLETED_SEASON,
    lookback_gws: int = DEFAULT_LOOKBACK_GWS,
) -> dict[int, dict]:
    """Per current player_id, their form over the last `lookback_gws`
    gameweeks of `season`.

    Returns {player_id: {
        "points_per_game": float,   # over appearances in the window
        "minutes_per_game": float,
        "total_points": int,
        "games": int,               # gameweeks they actually featured in
        "starts_rate": float,       # share of window gameweeks with 60+ minutes
    }}
    Players with no matched history are absent from the dict.
    """
    first_round = max(1, SEASON_LENGTH - lookback_gws + 1)

    conn = db.get_connection()
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hist_name ON historical_gw(season, name)")
        rows = conn.execute(
            """
            SELECT p.id,
                   SUM(h.total_points)                              AS pts,
                   SUM(h.minutes)                                   AS mins,
                   COUNT(*)                                         AS appearances,
                   SUM(CASE WHEN h.minutes >= 60 THEN 1 ELSE 0 END) AS starts
            FROM players p
            JOIN historical_gw h
              ON h.name = p.first_name || ' ' || p.second_name
            WHERE h.season = ? AND h.round >= ?
            GROUP BY p.id
            """,
            (season, first_round),
        ).fetchall()
    finally:
        conn.close()

    window_size = SEASON_LENGTH - first_round + 1
    out = {}
    for pid, pts, mins, appearances, starts in rows:
        if not appearances:
            continue
        out[pid] = {
            "points_per_game": (pts or 0) / appearances,
            "minutes_per_game": (mins or 0) / appearances,
            "total_points": pts or 0,
            "games": appearances,
            "starts_rate": (starts or 0) / window_size,
        }
    return out


def blended_form(
    lookback_gws: int = DEFAULT_LOOKBACK_GWS,
    full_confidence_games: int = 6,
) -> dict[int, dict]:
    """Blends this season's actual form with last season's late-season form,
    weighted by how much of this season has actually been played.

    At GW1 the blend is entirely historical; by `full_confidence_games` played
    it is entirely current-season. In between it shifts linearly, so a player's
    score moves smoothly from prior to evidence rather than lurching the moment
    a new season starts.

    Returns {player_id: {"points_per_game", "minutes_per_game", "history_weight"}}
    for every current player -- those with no matched history fall back to
    this season's numbers alone (history_weight 0).
    """
    hist = last_n_gw_form(lookback_gws=lookback_gws)

    conn = db.get_connection()
    try:
        players = conn.execute(
            "SELECT id, points_per_game, minutes, starts FROM players"
        ).fetchall()
    finally:
        conn.close()

    out = {}
    for pid, ppg_now, minutes_now, starts_now in players:
        games_now = starts_now or 0
        ppg_now = ppg_now or 0.0
        mpg_now = (minutes_now / games_now) if games_now else 0.0

        h = hist.get(pid)
        if h is None:
            out[pid] = {"points_per_game": ppg_now, "minutes_per_game": mpg_now, "history_weight": 0.0}
            continue

        # confidence in this season's sample grows with games played
        current_weight = min(1.0, games_now / full_confidence_games)
        history_weight = 1.0 - current_weight
        out[pid] = {
            "points_per_game": current_weight * ppg_now + history_weight * h["points_per_game"],
            "minutes_per_game": current_weight * mpg_now + history_weight * h["minutes_per_game"],
            "history_weight": history_weight,
        }
    return out
