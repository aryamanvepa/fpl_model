"""One-off (rarely re-run) ingest of historical per-gameweek player data from
the vaastav/Fantasy-Premier-League public GitHub dataset. This is the
training/backtesting data for Model 2 -- separate from the live daily refresh
in ingest_bootstrap.py.
"""

import csv
import io

import requests

from fpl_bot import db

RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASONS = ["2023-24", "2024-25", "2025-26"]


def _fetch_csv(url: str) -> list[dict]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


def _team_id_map(season: str) -> dict[str, str]:
    rows = _fetch_csv(f"{RAW_BASE}/{season}/teams.csv")
    return {row["id"]: row["name"] for row in rows}


def _bool(value: str) -> int:
    return 1 if value == "True" else 0


def _int(value: str, default=None):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def ingest_season(conn, season: str) -> int:
    team_names = _team_id_map(season)
    gw_rows = _fetch_csv(f"{RAW_BASE}/{season}/gws/merged_gw.csv")

    records = []
    seen_rows = set()  # dedupe exact-duplicate rows the source data occasionally has
    for row in gw_rows:
        opponent_name = team_names.get(row.get("opponent_team", ""), None)
        round_ = _int(row.get("round") or row.get("GW"))
        minutes = _int(row["minutes"], 0)
        points = _int(row["total_points"], 0)
        element_id = _int(row["element"])

        # legitimate double-gameweek rows (same element+round, different fixture)
        # are kept -- this key only catches exact full duplicates
        dedup_key = (season, element_id, round_, minutes, points, row.get("kickoff_time"))
        if dedup_key in seen_rows:
            continue
        seen_rows.add(dedup_key)

        records.append(
            (
                season,
                element_id,
                row["name"],
                row["position"],
                row["team"],
                opponent_name,
                _bool(row["was_home"]),
                round_,
                row.get("kickoff_time"),
                minutes,
                points,
                _int(row["value"]),
                _int(row["team_h_score"]),
                _int(row["team_a_score"]),
            )
        )

    conn.executemany(
        """
        INSERT INTO historical_gw (
            season, element_id, name, position, team, opponent_team, was_home, round,
            kickoff_time, minutes, total_points, value, team_h_score, team_a_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    return len(records)


def run(seasons: list[str] = SEASONS) -> None:
    db.init_db()
    conn = db.get_connection()
    try:
        conn.execute("DELETE FROM historical_gw WHERE season IN ({})".format(",".join("?" * len(seasons))), seasons)
        total = 0
        for season in seasons:
            n = ingest_season(conn, season)
            total += n
            print(f"  {season}: {n} rows")
        conn.commit()
    finally:
        conn.close()
    print(f"Ingested {total} historical player-gameweek rows across {len(seasons)} seasons.")


if __name__ == "__main__":
    run()
