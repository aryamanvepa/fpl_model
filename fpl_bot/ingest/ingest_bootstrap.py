"""Pull bootstrap-static + fixtures from the FPL API and load them into SQLite.

Safe to re-run: every table is refreshed (delete + reinsert) on each call, so
this doubles as the daily refresh job feeding the scheduler.
"""

from fpl_bot import api_client, db


def _to_float(value):
    if value in (None, ""):
        return None
    return float(value)


def _clear_all(conn) -> None:
    """Delete in child-to-parent order so foreign-key constraints don't trip."""
    conn.execute("DELETE FROM fixtures")
    conn.execute("DELETE FROM players")
    conn.execute("DELETE FROM gameweeks")
    conn.execute("DELETE FROM positions")
    conn.execute("DELETE FROM teams")


def ingest_teams(conn, teams: list[dict]) -> None:
    conn.executemany(
        """
        INSERT INTO teams (
            id, name, short_name,
            strength_overall_home, strength_overall_away,
            strength_attack_home, strength_attack_away,
            strength_defence_home, strength_defence_away
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                t["id"], t["name"], t["short_name"],
                t["strength_overall_home"], t["strength_overall_away"],
                t["strength_attack_home"], t["strength_attack_away"],
                t["strength_defence_home"], t["strength_defence_away"],
            )
            for t in teams
        ],
    )


def ingest_positions(conn, element_types: list[dict]) -> None:
    conn.executemany(
        """
        INSERT INTO positions (id, singular_name, short_name, squad_min_play, squad_max_play, squad_select)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                p["id"], p["singular_name"], p["singular_name_short"],
                p["squad_min_play"], p["squad_max_play"], p["squad_select"],
            )
            for p in element_types
        ],
    )


def ingest_players(conn, elements: list[dict]) -> None:
    conn.executemany(
        """
        INSERT INTO players (
            id, web_name, first_name, second_name, team_id, element_type,
            now_cost, status, news, chance_of_playing_next_round,
            total_points, points_per_game, form, selected_by_percent,
            minutes, starts, goals_scored, assists, clean_sheets, goals_conceded,
            bonus, bps, ict_index,
            expected_goals, expected_assists, expected_goal_involvements, expected_goals_conceded
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                e["id"], e["web_name"], e["first_name"], e["second_name"],
                e["team"], e["element_type"],
                e["now_cost"], e["status"], e["news"], e["chance_of_playing_next_round"],
                e["total_points"], _to_float(e["points_per_game"]), _to_float(e["form"]),
                _to_float(e["selected_by_percent"]),
                e["minutes"], e.get("starts", 0), e["goals_scored"], e["assists"], e["clean_sheets"], e["goals_conceded"],
                e["bonus"], e["bps"], _to_float(e["ict_index"]),
                _to_float(e["expected_goals"]), _to_float(e["expected_assists"]),
                _to_float(e["expected_goal_involvements"]), _to_float(e["expected_goals_conceded"]),
            )
            for e in elements
        ],
    )


def ingest_gameweeks(conn, events: list[dict]) -> None:
    conn.executemany(
        """
        INSERT INTO gameweeks (id, name, deadline_time, finished, is_current, is_next)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                ev["id"], ev["name"], ev["deadline_time"],
                int(ev["finished"]), int(ev["is_current"]), int(ev["is_next"]),
            )
            for ev in events
        ],
    )


def ingest_fixtures(conn, fixtures: list[dict]) -> None:
    conn.executemany(
        """
        INSERT INTO fixtures (id, event, kickoff_time, team_h, team_a, team_h_difficulty, team_a_difficulty, finished)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                f["id"], f["event"], f["kickoff_time"],
                f["team_h"], f["team_a"],
                f["team_h_difficulty"], f["team_a_difficulty"],
                int(f["finished"]),
            )
            for f in fixtures
        ],
    )


def run() -> None:
    db.init_db()
    bootstrap = api_client.get_bootstrap_static()
    fixtures = api_client.get_fixtures()

    conn = db.get_connection()
    try:
        _clear_all(conn)
        ingest_teams(conn, bootstrap["teams"])
        ingest_positions(conn, bootstrap["element_types"])
        ingest_players(conn, bootstrap["elements"])
        ingest_gameweeks(conn, bootstrap["events"])
        ingest_fixtures(conn, fixtures)
        conn.commit()
    finally:
        conn.close()

    print(f"Ingested {len(bootstrap['teams'])} teams, {len(bootstrap['elements'])} players, "
          f"{len(bootstrap['events'])} gameweeks, {len(fixtures)} fixtures.")


if __name__ == "__main__":
    run()
