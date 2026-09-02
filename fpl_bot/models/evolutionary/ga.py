"""The genetic algorithm loop: a population of genomes plays a season each
(the fitness function), the best-scoring ones survive and breed into the
next generation, repeat. Each generation here is exactly one "epoch" of
generational learning -- the direct analogue of the dino-game population
getting better at the game run after run.
"""

import random
import statistics
import warnings

warnings.filterwarnings("ignore")

from fpl_bot.models.evolutionary.features import build_normalized_features
from fpl_bot.models.evolutionary.genome import Genome
from fpl_bot.models.evolutionary.simulator import simulate_season


def fitness(genome: Genome, season_dfs: dict) -> float:
    # cast to plain Python float: totals come out as numpy scalars (pandas
    # arithmetic under the hood), and statistics.pstdev chokes on those
    return float(sum(simulate_season(genome, df)["total_points"] for df in season_dfs.values()))


def _tournament_select(population, fitnesses, k=3):
    idxs = random.sample(range(len(population)), k)
    best = max(idxs, key=lambda i: fitnesses[i])
    return population[best]


def evolve(
    train_seasons: list[str],
    validation_season: str | None = None,
    population_size: int = 30,
    generations: int = 20,
    elite_count: int = 2,
    tournament_size: int = 3,
    mutation_rate: float = 0.15,
    mutation_magnitude: float = 0.3,
    seed: int | None = 42,
    verbose: bool = True,
) -> dict:
    """Evolves a strategy against `train_seasons`.

    If `validation_season` is given, the returned genome is the one that
    scored best on that held-out season -- NOT the one that scored best on
    the training seasons. Selecting on training fitness alone is how a
    genetic algorithm overfits: with 11 features x 4 positions there are 44
    weights to tune, and a population will happily find combinations that
    exploit quirks of the training seasons and then generalize worse than a
    naive baseline. Validation-based selection is the early-stopping
    equivalent for evolution.
    """
    if seed is not None:
        random.seed(seed)

    needed = list(train_seasons) + ([validation_season] if validation_season else [])
    feats = build_normalized_features(needed)
    season_dfs = {s: feats[feats["season"] == s] for s in train_seasons}
    validation_df = feats[feats["season"] == validation_season] if validation_season else None

    population = [Genome.random() for _ in range(population_size)]
    history = []
    genome_snapshots = []  # best genome *as of* each generation -- lets the dashboard replay how strategy evolved

    best_validation = None  # (score, genome, generation)

    for gen in range(generations):
        fitnesses = [fitness(g, season_dfs) for g in population]
        best_fit = max(fitnesses)
        worst_fit = min(fitnesses)
        avg_fit = statistics.mean(fitnesses)
        std_fit = statistics.pstdev(fitnesses)

        ranked = sorted(zip(population, fitnesses), key=lambda pf: pf[1], reverse=True)
        gen_best_genome = ranked[0][0]
        genome_snapshots.append(gen_best_genome.copy())

        record = {"generation": gen, "best": best_fit, "avg": avg_fit, "worst": worst_fit, "std": std_fit}

        if validation_df is not None:
            val_score = float(simulate_season(gen_best_genome, validation_df)["total_points"])
            record["validation"] = val_score
            if best_validation is None or val_score > best_validation[0]:
                best_validation = (val_score, gen_best_genome.copy(), gen)
            if verbose:
                print(f"  generation {gen:>2}: train={best_fit:>6.0f}  avg={avg_fit:>6.0f}  val={val_score:>6.0f}")
        elif verbose:
            print(f"  generation {gen:>2}: best={best_fit:>6.0f}  avg={avg_fit:>6.0f}  worst={worst_fit:>6.0f}")

        history.append(record)

        new_population = [g.copy() for g, _ in ranked[:elite_count]]

        while len(new_population) < population_size:
            parent1 = _tournament_select(population, fitnesses, tournament_size)
            parent2 = _tournament_select(population, fitnesses, tournament_size)
            child = parent1.crossover(parent2)
            child.mutate(rate=mutation_rate, magnitude=mutation_magnitude)
            new_population.append(child)

        population = new_population

    final_fitnesses = [fitness(g, season_dfs) for g in population]

    if best_validation is not None:
        best_genome = best_validation[1]
        if verbose:
            print(f"  selected generation {best_validation[2]}'s genome (best validation score {best_validation[0]:.0f})")
    else:
        best_genome = population[final_fitnesses.index(max(final_fitnesses))]

    return {
        "selected_generation": best_validation[2] if best_validation else generations - 1,
        "validation_score": best_validation[0] if best_validation else None,
        "best_genome": best_genome,
        "history": history,
        "genome_snapshots": genome_snapshots,
        "final_best_fitness": max(final_fitnesses),
    }
