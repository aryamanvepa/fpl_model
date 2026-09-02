"""ILP squad optimizer (PuLP) -- picks the score-maximizing 15-man squad
under real FPL constraints, then the best valid starting XI + captain from it.
"""

from dataclasses import dataclass, field

import pulp

from fpl_bot.models.basic_stats import PlayerScore

BUDGET = 1000  # tenths of a million -> £100.0m
SQUAD_REQUIREMENTS = {1: 2, 2: 5, 3: 5, 4: 3}  # GK, DEF, MID, FWD
MAX_PER_CLUB = 3

# valid starting formations: (GK, DEF, MID, FWD), always 11 total
FORMATION_BOUNDS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}


@dataclass
class SquadResult:
    squad: list[PlayerScore]
    starting_xi: list[PlayerScore] = field(default_factory=list)
    bench: list[PlayerScore] = field(default_factory=list)
    captain: PlayerScore | None = None
    vice_captain: PlayerScore | None = None
    total_cost: int = 0


# Deterministic tie-break. When many players share an identical score -- which
# happens constantly, e.g. every player carries the same position-average
# cold-start score in gameweek 1 -- the squad problem is degenerate: a huge
# number of different squads are all exactly optimal, and CBC returns an
# arbitrary one that varies between processes. In a season simulation that
# arbitrary first pick then cascades through every subsequent week, producing
# ~200-point swings in the final score on identical inputs and making any
# model-vs-baseline comparison meaningless. Nudging each player's objective
# coefficient by a minuscule, stable function of their id makes the optimum
# unique and reproducible, while staying far below any real score difference.
TIE_BREAK_EPSILON = 1e-9


def _tie_broken(score: float, player_id: int) -> float:
    return score + player_id * TIE_BREAK_EPSILON


def pick_squad(players: list[PlayerScore], budget: int = BUDGET) -> list[PlayerScore]:
    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = {p.player_id: pulp.LpVariable(f"x_{p.player_id}", cat="Binary") for p in players}

    prob += pulp.lpSum(_tie_broken(p.score, p.player_id) * x[p.player_id] for p in players)
    prob += pulp.lpSum(p.now_cost * x[p.player_id] for p in players) <= budget

    for position, count in SQUAD_REQUIREMENTS.items():
        prob += pulp.lpSum(x[p.player_id] for p in players if p.element_type == position) == count

    teams = {p.team_id for p in players}
    for team_id in teams:
        prob += pulp.lpSum(x[p.player_id] for p in players if p.team_id == team_id) <= MAX_PER_CLUB

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Squad optimizer did not find an optimal solution: {pulp.LpStatus[status]}")

    return [p for p in players if x[p.player_id].value() == 1]


def pick_starting_xi(squad: list[PlayerScore]) -> list[PlayerScore]:
    prob = pulp.LpProblem("fpl_starting_xi", pulp.LpMaximize)
    x = {p.player_id: pulp.LpVariable(f"s_{p.player_id}", cat="Binary") for p in squad}

    prob += pulp.lpSum(_tie_broken(p.score, p.player_id) * x[p.player_id] for p in squad)
    prob += pulp.lpSum(x[p.player_id] for p in squad) == 11

    for position, (lo, hi) in FORMATION_BOUNDS.items():
        count = pulp.lpSum(x[p.player_id] for p in squad if p.element_type == position)
        prob += count >= lo
        prob += count <= hi

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Starting XI optimizer did not find an optimal solution: {pulp.LpStatus[status]}")

    return [p for p in squad if x[p.player_id].value() == 1]


def build_squad_result(players: list[PlayerScore], budget: int = BUDGET) -> SquadResult:
    squad = pick_squad(players, budget)
    starting_xi = pick_starting_xi(squad)
    starting_ids = {p.player_id for p in starting_xi}
    bench = sorted((p for p in squad if p.player_id not in starting_ids), key=lambda p: p.score, reverse=True)

    starters_by_score = sorted(starting_xi, key=lambda p: p.score, reverse=True)
    captain = starters_by_score[0] if starters_by_score else None
    vice_captain = starters_by_score[1] if len(starters_by_score) > 1 else None

    return SquadResult(
        squad=sorted(squad, key=lambda p: (p.element_type, -p.score)),
        starting_xi=sorted(starting_xi, key=lambda p: (p.element_type, -p.score)),
        bench=bench,
        captain=captain,
        vice_captain=vice_captain,
        total_cost=sum(p.now_cost for p in squad),
    )
