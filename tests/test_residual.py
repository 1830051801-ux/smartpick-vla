"""Tests for bounded residual composition and one SAC update."""

import math
from pathlib import Path

import torch
from torch import nn

from smartpick_vla.models import FrozenBasePolicy, compose_bounded_action
from smartpick_vla.training import (
    ResidualSAC,
    ResidualSACConfig,
    load_checkpoint,
    save_checkpoint,
)


class _ChunkBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state).reshape(-1, 2, 2)


def test_bounded_composition_matches_formula_and_clips() -> None:
    base = torch.tensor([[0.95, -0.90], [0.00, 0.10]])
    logits = torch.tensor([[10.0, -10.0], [0.0, 1.0]])
    final, residual = compose_bounded_action(
        base,
        logits,
        residual_scale=(0.2, 0.1),
        action_low=(-1.0, -0.95),
        action_high=(1.0, 0.95),
    )
    expected_residual = torch.tensor([0.2, 0.1]) * torch.tanh(logits)
    expected_final = torch.minimum(
        torch.maximum(base + expected_residual, torch.tensor([-1.0, -0.95])),
        torch.tensor([1.0, 0.95]),
    )

    torch.testing.assert_close(residual, expected_residual)
    torch.testing.assert_close(final, expected_final)
    assert final[0, 0] == 1.0
    assert final[0, 1] == -0.95


def test_frozen_base_policy_never_accumulates_gradients() -> None:
    base = _ChunkBase()
    frozen = FrozenBasePolicy(base, action_index=1)
    frozen.train()
    output = frozen(torch.randn(4, 3, requires_grad=True))

    assert output.shape == (4, 2)
    assert not output.requires_grad
    assert not frozen.training
    assert not base.training
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert all(parameter.grad is None for parameter in base.parameters())


def test_residual_sac_one_update_is_finite_and_bounded() -> None:
    torch.manual_seed(11)
    config = ResidualSACConfig(
        observation_dim=4,
        action_dim=2,
        hidden_dims=(32, 32),
        residual_scale=(0.15, 0.05),
        action_low=(-1.0, -0.5),
        action_high=(1.0, 0.5),
    )
    trainer = ResidualSAC(config)
    batch_size = 12
    base_action = torch.empty(batch_size, 2).uniform_(-0.3, 0.3)
    next_base_action = torch.empty(batch_size, 2).uniform_(-0.3, 0.3)
    actor_before = next(trainer.actor.parameters()).detach().clone()
    metrics = trainer.update(
        {
            "observation": torch.randn(batch_size, 4),
            "base_action": base_action,
            "action": base_action.clone(),
            "reward": torch.randn(batch_size),
            "next_observation": torch.randn(batch_size, 4),
            "next_base_action": next_base_action,
            "done": torch.tensor([0.0] * 10 + [1.0, 1.0]),
        }
    )

    assert int(trainer.update_count) == 1
    assert all(math.isfinite(value) for value in metrics.values())
    assert not torch.equal(actor_before, next(trainer.actor.parameters()))

    observation = torch.randn(batch_size, 4)
    action = trainer.select_action(observation, base_action, deterministic=True)
    assert action.shape == (batch_size, 2)
    assert (action >= torch.tensor([-1.0, -0.5])).all()
    assert (action <= torch.tensor([1.0, 0.5])).all()
    assert ((action - base_action).abs() <= torch.tensor([0.15, 0.05]) + 1e-6).all()


def test_sac_rejects_unbounded_replay_actions() -> None:
    trainer = ResidualSAC(ResidualSACConfig(observation_dim=3, action_dim=2, hidden_dims=(16,)))
    batch = {
        "observation": torch.zeros(2, 3),
        "base_action": torch.zeros(2, 2),
        "action": torch.tensor([[0.0, 0.0], [1.1, 0.0]]),
        "reward": torch.zeros(2),
        "next_observation": torch.zeros(2, 3),
        "next_base_action": torch.zeros(2, 2),
        "done": torch.zeros(2),
    }

    try:
        trainer.update(batch)
    except ValueError as error:
        assert "outside configured action bounds" in str(error)
    else:
        raise AssertionError("out-of-bounds replay action was accepted")


def test_sac_rejects_action_outside_residual_envelope() -> None:
    trainer = ResidualSAC(
        ResidualSACConfig(
            observation_dim=3,
            action_dim=2,
            hidden_dims=(16,),
            residual_scale=(0.1, 0.2),
        )
    )
    batch = {
        "observation": torch.zeros(2, 3),
        "base_action": torch.zeros(2, 2),
        "action": torch.tensor([[0.0, 0.0], [0.1001, 0.0]]),
        "reward": torch.zeros(2),
        "next_observation": torch.zeros(2, 3),
        "next_base_action": torch.zeros(2, 2),
        "done": torch.zeros(2),
    }

    try:
        trainer.update(batch)
    except ValueError as error:
        assert "exceeds configured residual bounds" in str(error)
    else:
        raise AssertionError("replay action outside the residual envelope was accepted")


def test_residual_sac_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(17)
    config = ResidualSACConfig(
        observation_dim=4,
        action_dim=5,
        hidden_dims=(16, 16),
        residual_scale=(0.1, 0.1, 0.05, 0.05, 0.2),
    )
    trainer = ResidualSAC(config)
    batch_size = 8
    base_action = torch.empty(batch_size, 5).uniform_(-0.4, 0.4)
    trainer.update(
        {
            "observation": torch.randn(batch_size, 4),
            "base_action": base_action,
            "action": base_action.clone(),
            "reward": torch.randn(batch_size),
            "next_observation": torch.randn(batch_size, 4),
            "next_base_action": torch.empty(batch_size, 5).uniform_(-0.4, 0.4),
            "done": torch.zeros(batch_size),
        }
    )
    checkpoint = save_checkpoint(
        tmp_path / "residual_sac.pt",
        trainer,
        optimizers=trainer.optimizers,
        step=int(trainer.update_count),
        config=config,
        extra={"base_policy": "frozen-compact-vla"},
    )

    restored = ResidualSAC(config)
    info = load_checkpoint(checkpoint, restored, optimizers=restored.optimizers)

    assert info.step == 1
    assert info.config["action_dim"] == 5
    assert info.extra == {"base_policy": "frozen-compact-vla"}
    assert int(restored.update_count) == 1
    for name, expected in trainer.state_dict().items():
        torch.testing.assert_close(expected, restored.state_dict()[name])
    assert all(optimizer.state for optimizer in restored.optimizers.values())
