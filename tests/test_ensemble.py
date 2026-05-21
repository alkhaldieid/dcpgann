import torch

from dcpgann.ensemble import EnsembleConfig, EnsembleOptimizer


def test_weighted_optimizer_returns_valid_metrics() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    model_a = torch.tensor([[3.0, 0.1], [2.0, 0.2], [0.2, 2.5], [0.1, 2.7]])
    model_b = torch.tensor([[0.2, 2.0], [2.0, 0.2], [0.3, 2.0], [2.0, 0.2]])

    optimizer = EnsembleOptimizer(
        EnsembleConfig(method="scipy", metric="balanced_accuracy", scipy_max_iter=2, scipy_pop_size=3, seed=3)
    )
    result = optimizer.fit([model_a, model_b], labels)

    assert result.method == "scipy_weighted_logits"
    assert len(result.weights) == 2
    assert abs(sum(result.weights) - 1.0) < 1e-6
    assert result.metrics["balanced_accuracy"] >= 0.5


def test_cartesian_program_optimizer_produces_a_genome() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    model_a = torch.tensor([[3.0, 0.1], [2.0, 0.2], [0.2, 2.5], [0.1, 2.7]])
    model_b = torch.tensor([[0.2, 2.0], [2.0, 0.2], [0.3, 2.0], [2.0, 0.2]])

    optimizer = EnsembleOptimizer(
        EnsembleConfig(
            method="cgp",
            metric="balanced_accuracy",
            cgp_nodes=4,
            cgp_population=6,
            cgp_generations=3,
            cgp_elite=2,
            seed=5,
        )
    )
    result = optimizer.fit([model_a, model_b], labels)

    assert result.method == "cartesian_program"
    assert "genome" in result.metadata
    assert result.score >= 0.5
