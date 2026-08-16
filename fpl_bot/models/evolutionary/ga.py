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
    population_size: int = 30,
    generations: int = 20,
    elite_count: int = 2,
    tournament_size: int = 3,
    mutation_rate: float = 0.15,
    mutation_magnitude: float = 0.3,
    seed: int | None = 42,
    verbose: bool = True,
) -> dict:
    if seed is not None:
        random.seed(seed)

    feats = build_normalized_features(train_seasons)
    season_dfs = {s: feats[feats["season"] == s] for s in train_seasons}

    population = [Genome.random() for _ in range(population_size)]
    history = []
    genome_snapshots = []  # best genome *as of* each generation -- lets the dashboard replay how strategy evolved

    for gen in range(generations):
        fitnesses = [fitness(g, season_dfs) for g in population]
        best_fit = max(fitnesses)
        worst_fit = min(fitnesses)
        avg_fit = statistics.mean(fitnesses)
        std_fit = statistics.pstdev(fitnesses)
        history.append({"generation": gen, "best": best_fit, "avg": avg_fit, "worst": worst_fit, "std": std_fit})
        if verbose:
            print(f"  generation {gen:>2}: best={best_fit:>6.0f}  avg={avg_fit:>6.0f}  worst={worst_fit:>6.0f}")

        ranked = sorted(zip(population, fitnesses), key=lambda pf: pf[1], reverse=True)
        genome_snapshots.append(ranked[0][0].copy())
        new_population = [g.copy() for g, _ in ranked[:elite_count]]

        while len(new_population) < population_size:
            parent1 = _tournament_select(population, fitnesses, tournament_size)
            parent2 = _tournament_select(population, fitnesses, tournament_size)
            child = parent1.crossover(parent2)
            child.mutate(rate=mutation_rate, magnitude=mutation_magnitude)
            new_population.append(child)

        population = new_population

    final_fitnesses = [fitness(g, season_dfs) for g in population]
    best_genome = population[final_fitnesses.index(max(final_fitnesses))]

    return {
        "best_genome": best_genome,
        "history": history,
        "genome_snapshots": genome_snapshots,
        "final_best_fitness": max(final_fitnesses),
    }
