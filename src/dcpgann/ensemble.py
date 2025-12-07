"""Differential evolution and DCGP-based ensemble optimization."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import differential_evolution
from sklearn.metrics import balanced_accuracy_score, f1_score


try:  # Optional dependency; used when available.
    import dcgp  # type: ignore
except Exception:  # pragma: no cover - presence depends on user environment
    dcgp = None


@dataclass
class EnsembleConfig:
    metric: str = "balanced_accuracy"  # or "f1"
    pop_size: int = 15
    max_iter: int = 50
    seed: int = 42
    method: str = "dcgp"  # or "scipy"
    weight_bounds: Tuple[float, float] = (0.0, 1.0)
    dcgp_generations: int = 120
    dcgp_elite: int = 4
    dcgp_lambda: int = 8
    dcgp_sigma: float = 0.1


class EnsembleOptimizer:
    """Optimize ensemble weights for multiple base model logits."""

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _metric(self, y_true: torch.Tensor, y_logits: torch.Tensor) -> float:
        preds = torch.argmax(y_logits, dim=1).numpy()
        y_true_np = y_true.numpy()
        if self.config.metric == "f1":
            return f1_score(y_true_np, preds)
        return balanced_accuracy_score(y_true_np, preds)

    def _objective(self, weights: np.ndarray, logits_stack: torch.Tensor, labels: torch.Tensor) -> float:
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / (weights.sum() + 1e-8)
        ensemble_logits = torch.einsum("mld,m->ld", logits_stack, weights)
        return -self._metric(labels, ensemble_logits)

    def optimize(self, logits_per_model: Sequence[torch.Tensor], labels: torch.Tensor) -> np.ndarray:
        logits_stack = torch.stack(logits_per_model, dim=0)  # (models, samples, classes)
        if self.config.method == "dcgp":
            try:
                return self._optimize_dcgp(logits_stack, labels)
            except ImportError:
                print("dcgp library not found; falling back to scipy differential evolution")
        return self._optimize_scipy(logits_stack, labels)

    def _optimize_scipy(self, logits_stack: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
        bounds = [self.config.weight_bounds for _ in range(logits_stack.size(0))]
        result = differential_evolution(
            func=lambda w: self._objective(w, logits_stack, labels),
            bounds=bounds,
            maxiter=self.config.max_iter,
            popsize=self.config.pop_size,
            seed=self.config.seed,
            recombination=0.7,
            updating="deferred",
        )
        return self._normalize_weights(result.x)

    def _optimize_dcgp(self, logits_stack: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
        if dcgp is None:  # pragma: no cover - environment dependent
            raise ImportError("dcgp is required for DCGP optimization")

        num_models = logits_stack.size(0)
        lower, upper = self.config.weight_bounds
        rng = np.random.default_rng(self.config.seed)

        def fitness_fn(weights: np.ndarray) -> float:
            clipped = np.clip(weights, lower, upper)
            clipped = clipped / (clipped.sum() + 1e-8)
            ensemble_logits = torch.einsum("mld,m->ld", logits_stack, torch.tensor(clipped, dtype=torch.float32))
            return -self._metric(labels, ensemble_logits)

        # Initialize a small population of candidate weight genomes
        population: List[np.ndarray] = [rng.uniform(lower, upper, size=num_models) for _ in range(max(self.config.dcgp_elite, 2))]

        def mutate(candidate: np.ndarray) -> np.ndarray:
            if hasattr(dcgp, "mutate_gaussian"):
                return np.array(dcgp.mutate_gaussian(candidate.tolist(), self.config.dcgp_sigma, rng))  # type: ignore[arg-type]
            if hasattr(dcgp, "mutation") and hasattr(dcgp.mutation, "gaussian"):
                return np.array(dcgp.mutation.gaussian(candidate.tolist(), sigma=self.config.dcgp_sigma, rng=rng))  # type: ignore[attr-defined]
            noise = rng.normal(0.0, self.config.dcgp_sigma, size=candidate.shape)
            return candidate + noise

        for _ in range(self.config.dcgp_generations):
            scores = np.array([fitness_fn(ind) for ind in population])
            elite_idx = list(np.argsort(scores)[: self.config.dcgp_elite])
            elites = [population[i] for i in elite_idx]
            offspring: List[np.ndarray] = []
            for _ in range(self.config.dcgp_lambda):
                parent = elites[rng.integers(0, len(elites))]
                child = mutate(parent)
                offspring.append(child)
            population = elites + offspring

        best = min(population, key=fitness_fn)
        return self._normalize_weights(best)

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.clip(weights, self.config.weight_bounds[0], self.config.weight_bounds[1])
        if math.isclose(weights.sum(), 0.0):
            return np.ones_like(weights) / len(weights)
        return weights / weights.sum()

    def ensemble_logits(self, logits_per_model: Sequence[torch.Tensor], weights: np.ndarray) -> torch.Tensor:
        weights_t = torch.tensor(weights, dtype=torch.float32)
        weights_t = weights_t / (weights_t.sum() + 1e-8)
        logits_stack = torch.stack(logits_per_model, dim=0)
        return torch.einsum("mld,m->ld", logits_stack, weights_t)

    def evaluate(self, logits_per_model: Sequence[torch.Tensor], labels: torch.Tensor) -> float:
        weights = np.ones(len(logits_per_model)) / len(logits_per_model)
        ensemble_logits = self.ensemble_logits(logits_per_model, weights)
        return self._metric(labels, ensemble_logits)
