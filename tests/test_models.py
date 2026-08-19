"""CPU tests for the compact supervised policies and checkpoints."""

from pathlib import Path

import numpy as np
import torch

from smartpick_vla.evaluation.controller import TemporalPolicyController
from smartpick_vla.models import (
    BehaviorCloningConfig,
    BehaviorCloningPolicy,
    ByteTextEncoder,
    CompactVLAConfig,
    CompactVLAPolicy,
    TemporalVLAConfig,
    TemporalVLAPolicy,
)
from smartpick_vla.training import load_checkpoint, save_checkpoint, supervised_train_step


def test_byte_encoder_handles_utf8_and_empty_instruction() -> None:
    encoder = ByteTextEncoder(32, max_length=12, nhead=4, dropout=0.0)
    encoded, padding = encoder(["pick scratch", "分拣合格品", ""])

    assert encoded.shape == (3, 12, 32)
    assert padding.shape == (3, 12)
    assert not padding[:, 0].any()  # Mandatory BOS keeps even empty text valid.
    assert torch.isfinite(encoded).all()


def test_compact_vla_action_horizon_and_parameter_count() -> None:
    assert CompactVLAConfig().robot_state_dim == 24
    assert CompactVLAConfig().action_dim == 5
    assert BehaviorCloningConfig().robot_state_dim == 24
    assert BehaviorCloningConfig().action_dim == 5
    config = CompactVLAConfig(
        robot_state_dim=24,
        action_dim=5,
        action_horizon=3,
        d_model=32,
        nhead=4,
        decoder_layers=1,
        language_layers=1,
        feedforward_dim=64,
        language_max_length=16,
        vision_grid_size=2,
        dropout=0.0,
    )
    model = CompactVLAPolicy(config)
    actions = model(
        rgb=torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8),
        instruction=["pick accepted part", "reject unknown object"],
        robot_state=torch.randn(2, 24),
    )

    assert actions.shape == (2, 3, 5)
    assert torch.isfinite(actions).all()
    assert actions.abs().max() <= 1.0
    assert 0 < model.parameter_count() < 2_000_000
    assert model.parameter_count(trainable_only=True) == model.parameter_count()


def test_configured_backbone_freezing_is_preserved_in_train_mode() -> None:
    model = CompactVLAPolicy(
        CompactVLAConfig(
            robot_state_dim=4,
            action_dim=2,
            action_horizon=2,
            d_model=32,
            nhead=4,
            decoder_layers=1,
            language_layers=1,
            feedforward_dim=64,
            language_max_length=8,
            vision_grid_size=2,
            freeze_vision=True,
            freeze_language=True,
        )
    )
    model.train()

    assert not model.vision_encoder.training
    assert not model.language_encoder.training
    assert all(not parameter.requires_grad for parameter in model.vision_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.language_encoder.parameters())
    assert model.parameter_count(trainable_only=True) < model.parameter_count()


def test_single_step_bc_optimizer_update_and_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(7)
    config = BehaviorCloningConfig(
        robot_state_dim=24,
        action_dim=5,
        d_model=32,
        hidden_dim=48,
        language_max_length=12,
        language_layers=1,
        nhead=4,
        dropout=0.0,
    )
    model = BehaviorCloningPolicy(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {
        "rgb": torch.rand(3, 3, 32, 32),
        "instruction": ["accept", "scratch", "unknown"],
        "robot_state": torch.randn(3, 24),
        "action": torch.empty(3, 5).uniform_(-0.8, 0.8),
    }
    before = model.action_head[-1].weight.detach().clone()
    result = supervised_train_step(model, optimizer, batch)

    assert result.loss >= 0.0
    assert result.gradient_norm > 0.0
    assert result.batch_size == 3
    assert not torch.equal(before, model.action_head[-1].weight)

    checkpoint_path = save_checkpoint(
        tmp_path / "bc.pt",
        model,
        optimizers=optimizer,
        step=1,
        config=config,
        extra={"kind": "unit-test"},
    )
    safely_loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert safely_loaded["checkpoint_version"] == 2
    restored_model = BehaviorCloningPolicy(config)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-3)
    info = load_checkpoint(
        checkpoint_path,
        restored_model,
        optimizers=restored_optimizer,
    )

    assert info.step == 1
    assert info.config["action_dim"] == 5
    assert info.config["robot_state_dim"] == 24
    assert info.extra == {"kind": "unit-test"}
    for expected, restored in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(expected, restored)


def test_checkpoint_restores_tensor_encoded_numpy_rng_state(tmp_path: Path) -> None:
    config = BehaviorCloningConfig(d_model=32, hidden_dim=32, nhead=4, language_layers=1)
    model = BehaviorCloningPolicy(config)
    checkpoint_path = save_checkpoint(tmp_path / "rng.pt", model, config=config)
    expected_torch = torch.rand(3)
    expected_numpy = np.random.random(3)

    restored_model = BehaviorCloningPolicy(config)
    load_checkpoint(checkpoint_path, restored_model, restore_rng=True)
    actual_torch = torch.rand(3)
    actual_numpy = np.random.random(3)

    torch.testing.assert_close(actual_torch, expected_torch)
    np.testing.assert_allclose(actual_numpy, expected_numpy)


def test_action_chunk_supervised_step_supports_padding_mask() -> None:
    config = CompactVLAConfig(
        robot_state_dim=24,
        action_dim=5,
        action_horizon=3,
        d_model=32,
        nhead=4,
        decoder_layers=1,
        language_layers=1,
        feedforward_dim=64,
        language_max_length=8,
        vision_grid_size=2,
        dropout=0.0,
    )
    model = CompactVLAPolicy(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = supervised_train_step(
        model,
        optimizer,
        {
            "rgb": torch.rand(2, 3, 32, 32),
            "instruction": ["one", "two"],
            "robot_state": torch.randn(2, 24),
            "action": torch.empty(2, 3, 5).uniform_(-1.0, 1.0),
            "action_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        },
    )

    assert result.loss >= 0.0
    assert result.gradient_norm > 0.0


def test_temporal_vla_fuses_left_padded_history_and_trains() -> None:
    config = TemporalVLAConfig(
        robot_state_dim=24,
        action_dim=5,
        action_horizon=3,
        observation_horizon=4,
        d_model=32,
        nhead=4,
        temporal_layers=1,
        decoder_layers=1,
        language_layers=1,
        feedforward_dim=64,
        language_max_length=12,
        vision_grid_size=2,
        dropout=0.0,
    )
    model = TemporalVLAPolicy(config)
    history_rgb = torch.randint(0, 256, (2, 4, 3, 32, 32), dtype=torch.uint8)
    history_state = torch.randn(2, 4, 24)
    history_mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
    actions = model(history_rgb, ["accepted", "scratch"], history_state, history_mask)

    assert actions.shape == (2, 3, 5)
    assert torch.isfinite(actions).all()
    assert actions.abs().max() <= 1.0
    assert model.parameter_count() > 0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = supervised_train_step(
        model,
        optimizer,
        {
            "rgb": history_rgb[:, -1],
            "instruction": ["accepted", "scratch"],
            "robot_state": history_state[:, -1],
            "rgb_history": history_rgb,
            "robot_state_history": history_state,
            "history_mask": history_mask,
            "action": torch.empty(2, 3, 5).uniform_(-1.0, 1.0),
            "action_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        },
    )
    assert result.loss >= 0.0
    assert result.gradient_norm > 0.0


def test_temporal_controller_preserves_observation_history() -> None:
    config = TemporalVLAConfig(
        d_model=32,
        nhead=4,
        temporal_layers=1,
        decoder_layers=1,
        language_layers=1,
        feedforward_dim=64,
        language_max_length=12,
        vision_grid_size=2,
        observation_horizon=3,
        action_horizon=2,
    )
    controller = TemporalPolicyController(TemporalVLAPolicy(config))
    observation = {
        "rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        "robot_state": np.zeros(24, dtype=np.float32),
        "instruction": "sort the accepted part",
    }
    first = controller.act(observation)
    observation["rgb"] = np.full((32, 32, 3), 17, dtype=np.uint8)
    second = controller.act(observation)

    assert first.action.shape == (5,)
    assert second.action.shape == (5,)
    assert first.replanned and second.replanned
