"""Forward-looking fixture difficulty: what each team faces over the next N
gameweeks, not just the single next one.

FPL's own difficulty rating (FDR) is 1-5, where 1 = easiest and 5 = hardest.
Everything here converts that to an "ease" score on a 0-1 scale where HIGHER
IS BETTER, so it can be weighted alongside the other positive signals in the
models without sign-flipping at every call site.

Also handles the two cases a naive "average the next 5" would get wrong:
blank gameweeks (a team has no fixture that week -- 0 points guaranteed) and
double gameweeks (two fixtures -- two scoring chances).
"""

from fpl_bot import db

DEFAULT_HORIZON = 5
FDR_MIN, FDR_MAX = 1, 5


def _fdr_to_ease(fdr: float) -> float:
    """1 (easiest) -> 1.0, 5 (hardest) -> 0.0."""
    return (FDR_MAX - fdr) / (FDR_MAX - FDR_MIN)


def next_gameweek(conn=None) -> int:
    own_conn = conn is None
    conn = conn or db.get_connection()
    try:
        row = conn.execute("SELECT id FROM gameweeks WHERE is_next = 1").fetchone()
        if row is None:
            row = conn.execute("SELECT id FROM gameweeks WHERE finished = 0 ORDER BY id LIMIT 1").fetchone()
        return row[0] if row else 1
    finally:
        if own_conn:
            conn.close()


def team_fixture_outlook(horizon: int = DEFAULT_HORIZON) -> dict[int, dict]:
    """Per team_id, the fixture picture over the next `horizon` gameweeks.

    Returns {team_id: {
        "ease": float,            # 0-1, higher = easier run (mean over fixtures played)
        "n_fixtures": int,        # <horizon means blanks, >horizon means doubles
        "fixtures_per_gw": float, # n_fixtures / horizon -- captures blank/double effect
        "opportunity": float,     # ease * fixtures_per_gw -- the combined signal
        "per_gw": [ {gw, opponent_id, is_home, fdr, ease}, ... ],
    }}
    """
    conn = db.get_connection()
    try:
        gw = next_gameweek(conn)
        team_ids = [r[0] for r in conn.execute("SELECT id FROM teams")]
        rows = conn.execute(
            """
            SELECT event, team_h, team_a, team_h_difficulty, team_a_difficulty
            FROM fixtures
            WHERE event >= ? AND event < ? AND finished = 0
            """,
            (gw, gw + horizon),
        ).fetchall()
    finally:
        conn.close()

    outlook = {tid: {"per_gw": []} for tid in team_ids}
    for event, team_h, team_a, fdr_h, fdr_a in rows:
        if team_h in outlook and fdr_h is not None:
            outlook[team_h]["per_gw"].append(
                {"gw": event, "opponent_id": team_a, "is_home": True, "fdr": fdr_h, "ease": _fdr_to_ease(fdr_h)}
            )
        if team_a in outlook and fdr_a is not None:
            outlook[team_a]["per_gw"].append(
                {"gw": event, "opponent_id": team_h, "is_home": False, "fdr": fdr_a, "ease": _fdr_to_ease(fdr_a)}
            )

    for tid, data in outlook.items():
        fixtures = sorted(data["per_gw"], key=lambda f: f["gw"])
        data["per_gw"] = fixtures
        data["n_fixtures"] = len(fixtures)
        # a team with no fixtures in the window gets neutral ease but zero opportunity --
        # "unknown difficulty" and "guaranteed no points" are different things
        data["ease"] = sum(f["ease"] for f in fixtures) / len(fixtures) if fixtures else 0.5
        data["fixtures_per_gw"] = len(fixtures) / horizon
        data["opportunity"] = data["ease"] * data["fixtures_per_gw"]

    return outlook


def player_fixture_features(horizon: int = DEFAULT_HORIZON) -> dict[int, dict]:
    """Same outlook, keyed by player_id instead of team_id -- the shape the
    models actually consume."""
    outlook = team_fixture_outlook(horizon)
    conn = db.get_connection()
    try:
        players = conn.execute("SELECT id, team_id FROM players").fetchall()
    finally:
        conn.close()

    neutral = {"ease": 0.5, "n_fixtures": 0, "fixtures_per_gw": 0.0, "opportunity": 0.0}
    return {
        pid: {k: v for k, v in outlook.get(team_id, neutral).items() if k != "per_gw"}
        for pid, team_id in players
    }


def fixture_heatmap_grid(horizon: int = DEFAULT_HORIZON) -> tuple[list[str], list[int], list[list[float]], list[list[str]]]:
    """The classic FPL "fixture ticker": teams x next N gameweeks, coloured by
    how easy each fixture is. Returns (team_names, gw_numbers, ease_matrix,
    label_matrix) ready to hand to a heatmap chart.

    Blank gameweeks come back as None in the ease matrix so they render as a
    visible gap rather than being silently treated as average difficulty.
    """
    outlook = team_fixture_outlook(horizon)
    conn = db.get_connection()
    try:
        teams = {r[0]: r[1] for r in conn.execute("SELECT id, short_name FROM teams")}
    finally:
        conn.close()

    gw_start = next_gameweek()
    gws = list(range(gw_start, gw_start + horizon))

    # easiest run first, so the top of the chart is where the opportunities are
    ordered_team_ids = sorted(teams, key=lambda t: -outlook.get(t, {}).get("opportunity", 0))

    ease_matrix, label_matrix = [], []
    for tid in ordered_team_ids:
        by_gw: dict[int, list[dict]] = {}
        for f in outlook.get(tid, {}).get("per_gw", []):
            by_gw.setdefault(f["gw"], []).append(f)

        ease_row, label_row = [], []
        for gw in gws:
            fixtures = by_gw.get(gw, [])
            if not fixtures:
                ease_row.append(None)
                label_row.append("-")  # blank gameweek
                continue
            ease_row.append(sum(f["ease"] for f in fixtures) / len(fixtures))
            label_row.append(
                ", ".join(
                    f"{teams.get(f['opponent_id'], '?')}{'(H)' if f['is_home'] else '(A)'}" for f in fixtures
                )
            )
        ease_matrix.append(ease_row)
        label_matrix.append(label_row)

    return [teams[t] for t in ordered_team_ids], gws, ease_matrix, label_matrix
