"""Ensemble optimization for IDC classifier outputs.

The module exposes two complementary optimizers:

* ``scipy`` optimizes non-negative weights over model logits.  It is the most
  transparent baseline and should always be reported.
* ``cgp`` evolves a small Cartesian program over base-model positive-class
  probabilities.  This is intentionally explicit: nodes, connections, function
  ids, constants, and active graph evaluation are visible in the code and in
  saved metadata.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import differential_evolution

from dcpgann.training import classification_metrics


@dataclass(frozen=True)
class EnsembleConfig:
    metric: str = "balanced_accuracy"
    method: str = "cgp"
    seed: int = 42
    weight_bounds: Tuple[float, float] = (0.0, 1.0)
    scipy_pop_size: int = 15
    scipy_max_iter: int = 50
    cgp_nodes: int = 12
    cgp_population: int = 24
    cgp_generations: int = 120
    cgp_elite: int = 4
    cgp_mutation_rate: float = 0.12
    cgp_constant_scale: float = 1.0


@dataclass(frozen=True)
class EnsembleResult:
    method: str
    score: float
    metrics: Dict[str, float]
    weights: List[float]
    metadata: Dict[str, object]


def _metric_value(metrics: Dict[str, float], metric: str) -> float:
    if metric not in metrics:
        raise ValueError(f"Unsupported metric '{metric}'. Available: {sorted(metrics)}")
    return metrics[metric]


def _positive_probabilities(logits_per_model: Sequence[torch.Tensor]) -> np.ndarray:
    probs = [torch.softmax(logits, dim=1)[:, 1].numpy() for logits in logits_per_model]
    return np.stack(probs, axis=1)


def _logits_from_positive_probability(prob_positive: np.ndarray) -> torch.Tensor:
    prob_positive = np.clip(prob_positive, 1e-6, 1.0 - 1e-6)
    probs = np.stack([1.0 - prob_positive, prob_positive], axis=1)
    return torch.tensor(np.log(probs), dtype=torch.float32)


class WeightedLogitOptimizer:
    """Differential evolution over normalized non-negative logit weights."""

    def __init__(self, config: EnsembleConfig):
        self.config = config

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        weights = np.clip(weights, self.config.weight_bounds[0], self.config.weight_bounds[1])
        if math.isclose(float(weights.sum()), 0.0):
            return np.ones_like(weights) / len(weights)
        return weights / weights.sum()

    def ensemble_logits(self, logits_per_model: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
        weights_t = torch.tensor(weights, dtype=torch.float32)
        weights_t = weights_t / (weights_t.sum() + 1e-8)
        logits_stack = torch.stack(logits_per_model, dim=0)
        return torch.einsum("mld,m->ld", logits_stack, weights_t)

    def optimize(self, logits_per_model: Sequence[torch.Tensor], labels: torch.Tensor) -> EnsembleResult:
        bounds = [self.config.weight_bounds for _ in range(len(logits_per_model))]

        def objective(raw_weights: np.ndarray) -> float:
            weights = self._normalize_weights(raw_weights)
            logits = self.ensemble_logits(logits_per_model, weights)
            metrics = classification_metrics(labels, logits)
            return -_metric_value(metrics, self.config.metric)

        result = differential_evolution(
            func=objective,
            bounds=bounds,
            maxiter=self.config.scipy_max_iter,
            popsize=self.config.scipy_pop_size,
            seed=self.config.seed,
            recombination=0.7,
            updating="deferred",
            polish=True,
        )
        weights = self._normalize_weights(result.x)
        logits = self.ensemble_logits(logits_per_model, weights)
        metrics = classification_metrics(labels, logits)
        return EnsembleResult(
            method="scipy_weighted_logits",
            score=_metric_value(metrics, self.config.metric),
            metrics=metrics,
            weights=weights.tolist(),
            metadata={"scipy": {"fun": float(result.fun), "nit": int(result.nit), "success": bool(result.success)}},
        )


_Function = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def _identity(a: np.ndarray, _b: np.ndarray, c: float) -> np.ndarray:
    return a


def _average(a: np.ndarray, b: np.ndarray, _c: float) -> np.ndarray:
    return 0.5 * (a + b)


def _weighted_sum(a: np.ndarray, b: np.ndarray, c: float) -> np.ndarray:
    alpha = 1.0 / (1.0 + np.exp(-c))
    return alpha * a + (1.0 - alpha) * b


def _product(a: np.ndarray, b: np.ndarray, _c: float) -> np.ndarray:
    return a * b


def _maximum(a: np.ndarray, b: np.ndarray, _c: float) -> np.ndarray:
    return np.maximum(a, b)


def _minimum(a: np.ndarray, b: np.ndarray, _c: float) -> np.ndarray:
    return np.minimum(a, b)


def _sigmoid_shift(a: np.ndarray, _b: np.ndarray, c: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a + c)))


FUNCTIONS: Tuple[Tuple[str, _Function], ...] = (
    ("identity", _identity),
    ("average", _average),
    ("weighted_sum", _weighted_sum),
    ("product", _product),
    ("maximum", _maximum),
    ("minimum", _minimum),
    ("sigmoid_shift", _sigmoid_shift),
)


@dataclass
class CGPGenome:
    functions: np.ndarray
    inputs_a: np.ndarray
    inputs_b: np.ndarray
    constants: np.ndarray
    output: int

    def copy(self) -> "CGPGenome":
        return CGPGenome(
            functions=self.functions.copy(),
            inputs_a=self.inputs_a.copy(),
            inputs_b=self.inputs_b.copy(),
            constants=self.constants.copy(),
            output=int(self.output),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "functions": [FUNCTIONS[int(idx)][0] for idx in self.functions],
            "inputs_a": self.inputs_a.astype(int).tolist(),
            "inputs_b": self.inputs_b.astype(int).tolist(),
            "constants": self.constants.astype(float).tolist(),
            "output": int(self.output),
        }


class CartesianProgramOptimizer:
    """Evolve a compact Cartesian program over model probabilities."""

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def _random_genome(self, num_inputs: int) -> CGPGenome:
        functions = self.rng.integers(0, len(FUNCTIONS), size=self.config.cgp_nodes)
        inputs_a = np.zeros(self.config.cgp_nodes, dtype=np.int64)
        inputs_b = np.zeros(self.config.cgp_nodes, dtype=np.int64)
        for node in range(self.config.cgp_nodes):
            max_source = num_inputs + node
            inputs_a[node] = self.rng.integers(0, max_source)
            inputs_b[node] = self.rng.integers(0, max_source)
        constants = self.rng.normal(0.0, self.config.cgp_constant_scale, size=self.config.cgp_nodes)
        output = int(self.rng.integers(0, num_inputs + self.config.cgp_nodes))
        return CGPGenome(functions, inputs_a, inputs_b, constants, output)

    def _mutate(self, genome: CGPGenome, num_inputs: int) -> CGPGenome:
        child = genome.copy()
        for node in range(self.config.cgp_nodes):
            max_source = num_inputs + node
            if self.rng.random() < self.config.cgp_mutation_rate:
                child.functions[node] = self.rng.integers(0, len(FUNCTIONS))
            if self.rng.random() < self.config.cgp_mutation_rate:
                child.inputs_a[node] = self.rng.integers(0, max_source)
            if self.rng.random() < self.config.cgp_mutation_rate:
                child.inputs_b[node] = self.rng.integers(0, max_source)
            if self.rng.random() < self.config.cgp_mutation_rate:
                child.constants[node] += self.rng.normal(0.0, self.config.cgp_constant_scale)
        if self.rng.random() < self.config.cgp_mutation_rate:
            child.output = int(self.rng.integers(0, num_inputs + self.config.cgp_nodes))
        return child

    def _evaluate_program(self, genome: CGPGenome, inputs: np.ndarray) -> np.ndarray:
        values: List[np.ndarray] = [inputs[:, column] for column in range(inputs.shape[1])]
        for node in range(self.config.cgp_nodes):
            name, fn = FUNCTIONS[int(genome.functions[node])]
            del name
            a = values[int(genome.inputs_a[node])]
            b = values[int(genome.inputs_b[node])]
            out = fn(a, b, float(genome.constants[node]))
            values.append(np.clip(out, 1e-6, 1.0 - 1e-6))
        return np.clip(values[int(genome.output)], 1e-6, 1.0 - 1e-6)

    def _score(self, genome: CGPGenome, inputs: np.ndarray, labels: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        prob_positive = self._evaluate_program(genome, inputs)
        logits = _logits_from_positive_probability(prob_positive)
        metrics = classification_metrics(labels, logits)
        return _metric_value(metrics, self.config.metric), metrics

    def optimize(self, logits_per_model: Sequence[torch.Tensor], labels: torch.Tensor) -> EnsembleResult:
        inputs = _positive_probabilities(logits_per_model)
        num_inputs = inputs.shape[1]
        population = [self._random_genome(num_inputs) for _ in range(self.config.cgp_population)]

        best_genome = population[0]
        best_score, best_metrics = self._score(best_genome, inputs, labels)
        history: List[Dict[str, float]] = []

        for generation in range(self.config.cgp_generations):
            scored = []
            for genome in population:
                score, metrics = self._score(genome, inputs, labels)
                scored.append((score, genome, metrics))
            scored.sort(key=lambda item: item[0], reverse=True)
            if scored[0][0] > best_score:
                best_score, best_genome, best_metrics = scored[0][0], scored[0][1].copy(), scored[0][2]
            history.append({"generation": float(generation), "best_score": float(scored[0][0])})

            elites = [item[1].copy() for item in scored[: self.config.cgp_elite]]
            next_population = elites.copy()
            while len(next_population) < self.config.cgp_population:
                parent = elites[int(self.rng.integers(0, len(elites)))]
                next_population.append(self._mutate(parent, num_inputs))
            population = next_population

        return EnsembleResult(
            method="cartesian_program",
            score=float(best_score),
            metrics=best_metrics,
            weights=[],
            metadata={
                "config": asdict(self.config),
                "genome": best_genome.to_dict(),
                "history": history,
                "function_set": [name for name, _ in FUNCTIONS],
            },
        )

    def predict_logits(self, logits_per_model: Sequence[torch.Tensor], genome_payload: Dict[str, object]) -> torch.Tensor:
        inputs = _positive_probabilities(logits_per_model)
        function_names = [name for name, _ in FUNCTIONS]
        genome = CGPGenome(
            functions=np.array([function_names.index(name) for name in genome_payload["functions"]], dtype=np.int64),
            inputs_a=np.array(genome_payload["inputs_a"], dtype=np.int64),
            inputs_b=np.array(genome_payload["inputs_b"], dtype=np.int64),
            constants=np.array(genome_payload["constants"], dtype=np.float64),
            output=int(genome_payload["output"]),
        )
        return _logits_from_positive_probability(self._evaluate_program(genome, inputs))


class EnsembleOptimizer:
    """Facade that preserves the original public API while returning metadata."""

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.weighted = WeightedLogitOptimizer(config)
        self.cgp = CartesianProgramOptimizer(config)
        self.last_result: EnsembleResult | None = None

    def optimize(self, logits_per_model: Sequence[torch.Tensor], labels: torch.Tensor) -> np.ndarray:
        result = self.fit(logits_per_model, labels)
        if result.weights:
            return np.array(result.weights, dtype=np.float64)
        return np.ones(len(logits_per_model), dtype=np.float64) / len(logits_per_model)

    def fit(self, logits_per_model: Sequence[torch.Tensor], labels: torch.Tensor) -> EnsembleResult:
        if self.config.method in {"scipy", "weighted", "weighted_logits"}:
            result = self.weighted.optimize(logits_per_model, labels)
        elif self.config.method in {"cgp", "dcgp"}:
            result = self.cgp.optimize(logits_per_model, labels)
        else:
            raise ValueError("method must be one of: cgp, dcgp, scipy, weighted, weighted_logits")
        self.last_result = result
        return result

    def ensemble_logits(self, logits_per_model: Sequence[torch.Tensor], weights: np.ndarray) -> torch.Tensor:
        return self.weighted.ensemble_logits(logits_per_model, weights)

    def evaluate(self, logits_per_model: Sequence[torch.Tensor], labels: torch.Tensor) -> float:
        weights = np.ones(len(logits_per_model)) / len(logits_per_model)
        logits = self.ensemble_logits(logits_per_model, weights)
        return classification_metrics(labels, logits)[self.config.metric]
