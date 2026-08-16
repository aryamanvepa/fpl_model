"""A genome is a candidate FPL strategy: how much to weight each signal per
position, and how big a score improvement has to be before it's worth a
transfer (or worth taking a -4 hit for). Evolution mutates and recombines
these to search for a better strategy than any human hand-tuned one.
"""

import random
from dataclasses import dataclass, field

POSITIONS = ["GK", "DEF", "MID", "FWD"]
FEATURES = [
    "was_home",
    "price",
    "rolling_points_3",
    "rolling_minutes_3",
    "team_goals_for_rate",
    "team_goals_against_rate",
    "opp_goals_for_rate",
    "opp_goals_against_rate",
]


@dataclass
class Genome:
    weights: dict  # {position: {feature: float}}
    transfer_threshold: float  # min score gain to spend a free transfer
    hit_threshold: float  # min score gain to spend a -4 hit

    @staticmethod
    def random() -> "Genome":
        weights = {pos: {f: random.uniform(-1, 1) for f in FEATURES} for pos in POSITIONS}
        return Genome(
            weights=weights,
            transfer_threshold=random.uniform(0.05, 0.4),
            hit_threshold=random.uniform(0.3, 1.0),
        )

    def score(self, position: str, normalized_features: dict) -> float:
        w = self.weights[position]
        return sum(w[f] * normalized_features[f"n_{f}"] for f in FEATURES)

    def crossover(self, other: "Genome") -> "Genome":
        child_weights = {}
        for pos in POSITIONS:
            child_weights[pos] = {}
            for f in FEATURES:
                mix = random.random()
                child_weights[pos][f] = mix * self.weights[pos][f] + (1 - mix) * other.weights[pos][f]
        return Genome(
            weights=child_weights,
            transfer_threshold=random.choice([self.transfer_threshold, other.transfer_threshold]),
            hit_threshold=random.choice([self.hit_threshold, other.hit_threshold]),
        )

    def mutate(self, rate: float = 0.1, magnitude: float = 0.3) -> None:
        for pos in POSITIONS:
            for f in FEATURES:
                if random.random() < rate:
                    self.weights[pos][f] = max(-2.0, min(2.0, self.weights[pos][f] + random.gauss(0, magnitude)))
        if random.random() < rate:
            self.transfer_threshold = max(0.01, self.transfer_threshold + random.gauss(0, 0.05))
        if random.random() < rate:
            self.hit_threshold = max(0.05, self.hit_threshold + random.gauss(0, 0.1))

    def copy(self) -> "Genome":
        return Genome(
            weights={pos: dict(fw) for pos, fw in self.weights.items()},
            transfer_threshold=self.transfer_threshold,
            hit_threshold=self.hit_threshold,
        )
