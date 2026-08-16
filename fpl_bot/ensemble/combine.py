"""Combine the four models' scores into one ranking.

The four models produce scores in totally different, incompatible units --
Model 2 outputs predicted FPL points (~0-8), Model 1 a genome-weighted
linear combination (~-3 to 3), Models 3/4 a 0-1 composite. Averaging those
directly would let whichever model happens to use the widest numeric range
dominate for no principled reason. Instead each model's scores are converted
to a percentile rank (0=worst, 1=best) *within that model* first, and only
the ranks are combined -- standard practice for ensembling heterogeneous
scoring systems.
"""

from fpl_bot.models.basic_stats import PlayerScore

# Weights reflect what's actually been backtested so far, not vibes:
#   model1 (evolutionary): beat a sensible baseline by +84 pts/season on held-out data
#   model2 (statistical):  beat a naive baseline by ~5% MAE on held-out data
#   -> both get the most weight, roughly balanced against each other
#   model3 (basic stats):  hand-tuned, never formally backtested -- moderate weight
#   model4 (qualitative):  an adjustment layer with no live LLM backtest yet in this
#   environment -- lowest weight until it's actually been run for real and measured
MODEL_WEIGHTS = {"model1": 0.30, "model2": 0.35, "model3": 0.20, "model4": 0.15}


def percentile_ranks(scores: list[PlayerScore]) -> dict[int, float]:
    ordered = sorted(scores, key=lambda p: p.score)
    n = len(ordered)
    if n <= 1:
        return {p.player_id: 0.5 for p in ordered}
    return {p.player_id: i / (n - 1) for i, p in enumerate(ordered)}


def compute_ensemble_scores(model_scores: dict[str, list[PlayerScore]], weights: dict | None = None) -> list[PlayerScore]:
    weights = weights or MODEL_WEIGHTS
    active = {k: w for k, w in weights.items() if model_scores.get(k)}
    if not active:
        raise RuntimeError("No model scores available to build an ensemble from.")

    rank_maps = {k: percentile_ranks(model_scores[k]) for k in active}

    meta_lookup: dict[int, PlayerScore] = {}
    for scores in model_scores.values():
        for p in scores:
            meta_lookup.setdefault(p.player_id, p)

    results = []
    for pid, meta in meta_lookup.items():
        weighted, total_w = 0.0, 0.0
        for k, w in active.items():
            r = rank_maps[k].get(pid)
            if r is not None:
                weighted += w * r
                total_w += w
        if total_w == 0:
            continue
        results.append(
            PlayerScore(pid, meta.web_name, meta.team_id, meta.element_type, meta.now_cost, round(weighted / total_w, 4))
        )

    results.sort(key=lambda p: p.score, reverse=True)
    return results


def model_agreement(model_scores: dict[str, list[PlayerScore]], top_n: int = 40) -> dict[int, list[str]]:
    """player_id -> list of model keys that rank this player in their own top N.
    A quick, readable consensus signal for the digest."""
    agreement: dict[int, list[str]] = {}
    for key, scores in model_scores.items():
        top_ids = {p.player_id for p in sorted(scores, key=lambda p: p.score, reverse=True)[:top_n]}
        for pid in top_ids:
            agreement.setdefault(pid, []).append(key)
    return agreement
