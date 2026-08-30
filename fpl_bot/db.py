"""SQLite schema and connection helper for the FPL data store."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "fpl.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT NOT NULL,
    strength_overall_home INTEGER,
    strength_overall_away INTEGER,
    strength_attack_home INTEGER,
    strength_attack_away INTEGER,
    strength_defence_home INTEGER,
    strength_defence_away INTEGER
);

CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY,
    singular_name TEXT NOT NULL,
    short_name TEXT NOT NULL,
    squad_min_play INTEGER,
    squad_max_play INTEGER,
    squad_select INTEGER
);

CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    web_name TEXT NOT NULL,
    first_name TEXT,
    second_name TEXT,
    team_id INTEGER REFERENCES teams(id),
    element_type INTEGER REFERENCES positions(id),
    now_cost INTEGER,
    status TEXT,
    news TEXT,
    chance_of_playing_next_round INTEGER,
    total_points INTEGER,
    points_per_game REAL,
    form REAL,
    selected_by_percent REAL,
    minutes INTEGER,
    starts INTEGER,
    goals_scored INTEGER,
    assists INTEGER,
    clean_sheets INTEGER,
    goals_conceded INTEGER,
    bonus INTEGER,
    bps INTEGER,
    ict_index REAL,
    expected_goals REAL,
    expected_assists REAL,
    expected_goal_involvements REAL,
    expected_goals_conceded REAL
);

CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY,
    event INTEGER,
    kickoff_time TEXT,
    team_h INTEGER REFERENCES teams(id),
    team_a INTEGER REFERENCES teams(id),
    team_h_difficulty INTEGER,
    team_a_difficulty INTEGER,
    finished INTEGER
);

CREATE TABLE IF NOT EXISTS gameweeks (
    id INTEGER PRIMARY KEY,
    name TEXT,
    deadline_time TEXT,
    finished INTEGER,
    is_current INTEGER,
    is_next INTEGER
);

-- Historical per-gameweek player rows (multiple past seasons), used to train
-- and backtest Model 2. Team/opponent are stored as names since each
-- season's numeric team ids are not stable across seasons.
CREATE TABLE IF NOT EXISTS historical_gw (
    season TEXT NOT NULL,
    element_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    position TEXT NOT NULL,
    team TEXT NOT NULL,
    opponent_team TEXT,
    was_home INTEGER,
    round INTEGER,
    kickoff_time TEXT,
    minutes INTEGER,
    total_points INTEGER,
    value INTEGER,
    team_h_score INTEGER,
    team_a_score INTEGER
);

CREATE INDEX IF NOT EXISTS idx_historical_gw_season_round ON historical_gw(season, round);
CREATE INDEX IF NOT EXISTS idx_historical_gw_team ON historical_gw(season, team);

-- What each of the 5 teams (model1-4, ensemble) currently holds. This is our
-- own record of "what we last told this team to do" -- once real FPL account
-- credentials are wired in (Phase 7), this gets replaced/reconciled against
-- the actual entry API, but it lets the daily digest diff and propose real
-- transfers today instead of just re-drafting from scratch every run.
CREATE TABLE IF NOT EXISTS team_state (
    team_key TEXT PRIMARY KEY,
    gw INTEGER,
    squad_json TEXT NOT NULL,       -- list of player_ids currently held
    starting_json TEXT NOT NULL,    -- list of player_ids in the starting XI
    captain_id INTEGER,
    vice_captain_id INTEGER,
    updated_at TEXT NOT NULL
);

-- Anything that needs your go-ahead before it's applied (mixed-autonomy rule:
-- captain/bench changes auto-apply, transfers and chips queue here).
CREATE TABLE IF NOT EXISTS pending_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_key TEXT NOT NULL,
    gw INTEGER,
    kind TEXT NOT NULL,             -- 'initial_draft' | 'transfer'
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'approved' | 'rejected'
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

-- One row per gameweek the scheduler has already run a full digest for --
-- lets a daily-triggered check stay idempotent (run once per gameweek
-- cycle, not once per day it happens to be within the pre-deadline window).
CREATE TABLE IF NOT EXISTS digest_runs (
    gw INTEGER PRIMARY KEY,
    run_at TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
