"""End-to-end bounded residual collection smoke test."""

from dataclasses import asdict
from pathlib import Path

from smartpick_vla.envs import SmartPickEnv
from smartpick_vla.envs.randomization import DomainRandomizationConfig
from smartpick_vla.evaluation.residual_controller import ResidualPolicyController
from smartpick_vla.models import CompactVLAConfig, CompactVLAPolicy
from smartpick_vla.training.checkpoint import save_checkpoint
from smartpick_vla.training.online_residual import ResidualOnlineConfig, train_residual_online


def test_tiny_online_residual_training_and_controller(tmp_path: Path) -> None:
    model_config = CompactVLAConfig(
        robot_state_dim=24,
        action_dim=5,
        action_horizon=2,
        d_model=32,
        nhead=4,
        decoder_layers=1,
        language_layers=1,
        feedforward_dim=64,
        language_max_length=16,
        vision_grid_size=2,
    )
    base = CompactVLAPolicy(model_config)
    base_checkpoint = save_checkpoint(
        tmp_path / "base.pt",
        base,
        config=model_config,
        extra={
            "policy_kind": "compact_vla",
            "model_config": asdict(model_config),
        },
    )
    output = tmp_path / "residual"
    manifest = train_residual_online(
        base_checkpoint,
        output,
        config=ResidualOnlineConfig(
            seed=9,
            environment_steps=6,
            warmup_steps=2,
            updates_per_step=1,
            batch_size=2,
            replay_capacity=16,
            observation_dim=29,
            hidden_dims=(16, 16),
            residual_scale=(0.1, 0.1, 0.08, 0.05, 0.04),
            image_size=32,
            max_episode_steps=3,
        ),
        domain_randomization=DomainRandomizationConfig(enabled=False),
    )

    assert manifest["base_policy_frozen"]
    assert manifest["gradient_updates"] == 4
    assert (output / "last.pt").is_file()
    controller = ResidualPolicyController(base_checkpoint, output / "last.pt")
    env = SmartPickEnv(image_size=32, max_episode_steps=2)
    observation, _ = env.reset(seed=4)
    decision = controller.act(observation)
    assert decision.action.shape == (5,)
    assert decision.residual_action is not None
    assert decision.base_action is not None
    env.close()
