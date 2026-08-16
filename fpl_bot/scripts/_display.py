"""Shared console formatting for squad-draft scripts."""

from fpl_bot import db
from fpl_bot.models.basic_stats import PlayerScore
from fpl_bot.optimizer.squad_optimizer import SquadResult

POSITION_NAMES = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def lookup_team_names() -> dict[int, str]:
    conn = db.get_connection()
    try:
        return {row[0]: row[1] for row in conn.execute("SELECT id, short_name FROM teams")}
    finally:
        conn.close()


def _fmt(p: PlayerScore, teams: dict[int, str]) -> str:
    return f"{p.web_name:<20} {teams.get(p.team_id, '?'):<4} {POSITION_NAMES[p.element_type]:<4} £{p.now_cost / 10:>4.1f}m  score={p.score:.3f}"


def print_result(result: SquadResult, teams: dict[int, str]) -> None:
    print("=" * 70)
    print("STARTING XI")
    print("=" * 70)
    for p in result.starting_xi:
        tag = ""
        if result.captain and p.player_id == result.captain.player_id:
            tag = "  (C)"
        elif result.vice_captain and p.player_id == result.vice_captain.player_id:
            tag = "  (VC)"
        print(_fmt(p, teams) + tag)

    print()
    print("BENCH (order)")
    print("-" * 70)
    for p in result.bench:
        print(_fmt(p, teams))

    print()
    print(f"Total squad cost: £{result.total_cost / 10:.1f}m / £100.0m")
    print("=" * 70)
