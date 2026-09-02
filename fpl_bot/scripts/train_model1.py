"""Evolve Model 1's strategy on 2023-24 + 2024-25, then backtest the best
genome on held-out 2025-26 against a simple, sensible non-evolved baseline
(a genome that just weights recent form and nothing else) -- both run
through the exact same simulator, so it's an apples-to-apples test of
whether evolution actually found something better than the obvious default.

Run with: python -m fpl_bot.scripts.train_model1
"""

import io
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from fpl_bot.models.evolutionary.features import build_normalized_features
from fpl_bot.models.evolutionary.ga import evolve
from fpl_bot.models.evolutionary.genome import FEATURES, POSITIONS, Genome
from fpl_bot.models.evolutionary.simulator import simulate_season

# Three-way split. The genome is *selected* on the validation season, never on
# training fitness -- with 44 weights to tune, selecting on training score alone
# reliably produces a genome that beats the baseline in-sample and loses to it
# out-of-sample. The test season is touched exactly once, at the end.
TRAIN_SEASONS = ["2023-24"]
VALIDATION_SEASON = "2024-25"
TEST_SEASON = "2025-26"
MODEL_PATH = Path(__file__).parent.parent / "data" / "model1_genome.pkl"


def naive_baseline_genome() -> Genome:
    """A sensible non-evolved default: trust recent form, nothing else."""
    weights = {pos: {f: 0.0 for f in FEATURES} for pos in POSITIONS}
    for pos in POSITIONS:
        weights[pos]["rolling_points_3"] = 1.0
    return Genome(weights=weights, transfer_threshold=0.1, hit_threshold=0.5)


def run(population_size: int = 30, generations: int = 20):
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print(f"Evolving on {TRAIN_SEASONS}, selecting on validation season {VALIDATION_SEASON} "
          f"-- {population_size} genomes x {generations} generations")
    result = evolve(
        TRAIN_SEASONS,
        validation_season=VALIDATION_SEASON,
        population_size=population_size,
        generations=generations,
        seed=42,
    )
    best = result["best_genome"]

    print()
    print(f"Backtesting best genome on held-out {TEST_SEASON}")
    test_feats = build_normalized_features([TEST_SEASON])
    test_df = test_feats[test_feats["season"] == TEST_SEASON]

    evolved_result = simulate_season(best, test_df)
    baseline_result = simulate_season(naive_baseline_genome(), test_df)

    print(f"  Evolved genome:  {evolved_result['total_points']:.0f} pts "
          f"(raw {evolved_result['raw_points']:.0f}, {evolved_result['hits_taken']} hits)")
    print(f"  Naive baseline:  {baseline_result['total_points']:.0f} pts "
          f"(raw {baseline_result['raw_points']:.0f}, {baseline_result['hits_taken']} hits)")
    diff = evolved_result["total_points"] - baseline_result["total_points"]
    print(f"  Evolution {'beat' if diff > 0 else 'lost to'} the naive baseline by {abs(diff):.0f} points over the season.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "genome": best,
                "history": result["history"],
                "genome_snapshots": result["genome_snapshots"],
                "selected_generation": result["selected_generation"],
                "validation_score": result["validation_score"],
                "test_result": evolved_result,
                "baseline_result": baseline_result,
            },
            f,
        )

    return result, evolved_result, baseline_result


if __name__ == "__main__":
    run()
