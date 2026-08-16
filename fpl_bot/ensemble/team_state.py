"""What each of the 5 teams currently holds, and the queue of proposed
changes waiting on your approval. This is what lets the daily digest say
"transfer X for Y" instead of just re-printing a fresh squad every day.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from fpl_bot import db
from fpl_bot.optimizer.squad_optimizer import SquadResult


@dataclass
class TeamState:
    team_key: str
    gw: int
    squad_ids: list[int]
    starting_ids: list[int]
    captain_id: int | None
    vice_captain_id: int | None


def get_team_state(team_key: str) -> TeamState | None:
    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT team_key, gw, squad_json, starting_json, captain_id, vice_captain_id "
            "FROM team_state WHERE team_key = ?",
            (team_key,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return TeamState(
        team_key=row[0],
        gw=row[1],
        squad_ids=json.loads(row[2]),
        starting_ids=json.loads(row[3]),
        captain_id=row[4],
        vice_captain_id=row[5],
    )


def save_team_state(team_key: str, gw: int, result: SquadResult) -> None:
    conn = db.get_connection()
    try:
        conn.execute(
            """
            INSERT INTO team_state (team_key, gw, squad_json, starting_json, captain_id, vice_captain_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(team_key) DO UPDATE SET
                gw=excluded.gw, squad_json=excluded.squad_json, starting_json=excluded.starting_json,
                captain_id=excluded.captain_id, vice_captain_id=excluded.vice_captain_id, updated_at=excluded.updated_at
            """,
            (
                team_key,
                gw,
                json.dumps([p.player_id for p in result.squad]),
                json.dumps([p.player_id for p in result.starting_xi]),
                result.captain.player_id if result.captain else None,
                result.vice_captain.player_id if result.vice_captain else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def queue_approval(team_key: str, gw: int, kind: str, payload: dict) -> int:
    conn = db.get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO pending_approvals (team_key, gw, kind, payload_json, status, created_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (team_key, gw, kind, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_pending_approvals(team_key: str | None = None) -> list[dict]:
    conn = db.get_connection()
    try:
        if team_key:
            rows = conn.execute(
                "SELECT id, team_key, gw, kind, payload_json, created_at FROM pending_approvals "
                "WHERE status = 'pending' AND team_key = ? ORDER BY id",
                (team_key,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, team_key, gw, kind, payload_json, created_at FROM pending_approvals "
                "WHERE status = 'pending' ORDER BY id"
            ).fetchall()
    finally:
        conn.close()
    return [
        {"id": r[0], "team_key": r[1], "gw": r[2], "kind": r[3], "payload": json.loads(r[4]), "created_at": r[5]}
        for r in rows
    ]


def resolve_approval(approval_id: int, approved: bool) -> None:
    conn = db.get_connection()
    try:
        conn.execute(
            "UPDATE pending_approvals SET status = ?, resolved_at = ? WHERE id = ?",
            ("approved" if approved else "rejected", datetime.now(timezone.utc).isoformat(), approval_id),
        )
        conn.commit()
    finally:
        conn.close()


def apply_team(team_key: str, gw: int, qualitative_backend: str = "ollama") -> SquadResult:
    """Recompute this team's current best squad and force-apply it as the new
    held state -- what an approval actually does. Recomputing fresh (rather
    than replaying the payload the approval was queued with) means the applied
    squad reflects whatever the latest data says right now, not a stale snapshot."""
    from fpl_bot.ensemble.combine import compute_ensemble_scores
    from fpl_bot.ensemble.models_registry import compute_all_model_scores
    from fpl_bot.optimizer.squad_optimizer import build_squad_result

    model_result = compute_all_model_scores(qualitative_backend=qualitative_backend)
    scores = model_result["scores"]

    if team_key == "ensemble":
        team_scores = compute_ensemble_scores(scores)
    else:
        team_scores = scores.get(team_key)
        if team_scores is None:
            raise RuntimeError(
                f"{team_key} has no scores available right now: {model_result['errors'].get(team_key, 'unknown error')}"
            )

    result = build_squad_result(team_scores)
    save_team_state(team_key, gw, result)
    return result


def diff_squads(old: TeamState | None, new_squad_ids: list[int]) -> tuple[list[int], list[int]]:
    """Returns (transfers_out, transfers_in) player ids. Empty on first-ever draft."""
    if old is None:
        return [], []
    old_set, new_set = set(old.squad_ids), set(new_squad_ids)
    return sorted(old_set - new_set), sorted(new_set - old_set)
