"""Checks for EXP-008 hierarchical visual fusion.

Run ``--config-only`` before the server is available. The default mode also
executes CPU tensor forwards, fallback equivalence, diagnostics, and gradients;
that mode is intentionally deferred until the target environment is connected.
"""
import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "v1": ROOT / "configs" / "track2_av_convtasnet_hierarchical_v1.yml",
    "v2": ROOT / "configs" / "track2_av_convtasnet_hierarchical_v2.yml",
}


def check_configs():
    for version, path in CONFIGS.items():
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        model_config = config["audionet"]["audionet_config"]
        training = config["training"]
        data_config = config["datamodule"]["data_config"]

        assert model_config["hierarchical_fusion_version"] == version
        assert model_config["hierarchical_video_layers"] == [1, 3, 5]
        assert model_config["D"] == 5
        assert model_config["audio_index"] == 1
        assert model_config["R"] == 4
        assert model_config["skip_con"] is False
        assert "av_model.video." in training["freeze_param_prefixes"]
        assert "av_model.video" in training["force_eval_module_prefixes"]
        assert data_config["degrade_prob"] == 0.0


def make_tiny_model(version):
    from look2hear.models.av_convtasnet import AV_model

    return AV_model(
        N=8,
        L=8,
        B=8,
        Sc=8,
        H=16,
        P=3,
        X=2,
        E=8,
        V=8,
        K=3,
        D=5,
        F=8,
        R=4,
        skip_con=False,
        audio_index=1,
        norm="gln",
        causal=False,
        fusion_type="concat",
        hierarchical_fusion_version=version,
        hierarchical_video_layers=(1, 3, 5),
        hierarchical_residual_init=0.0,
        hierarchical_gate_hidden=4,
        hierarchical_stage_residual_init=0.0,
    )


def load_baseline_state(target, baseline_state):
    missing, unexpected = target.load_state_dict(baseline_state, strict=False)
    assert not unexpected
    assert missing
    assert all("hierarchical_" in key for key in missing)


def check_tensor_paths():
    import torch

    from look2hear.models.av_convtasnet import Video_Sequential

    torch.manual_seed(0)
    batch, audio_samples, video_steps = 2, 128, 9
    audio = torch.randn(batch, audio_samples)
    video = torch.randn(batch, 8, video_steps)

    video_stack = Video_Sequential(8, 8, 3, skip_con=False, repeat=5).eval()
    final = video_stack(video)
    selected_final, levels = video_stack.forward_with_layers(video, (1, 3, 5))
    assert len(levels) == 3
    assert all(level.shape == (batch, 8, video_steps) for level in levels)
    assert torch.allclose(final, selected_final, atol=0.0, rtol=0.0)
    assert torch.allclose(final, levels[-1], atol=0.0, rtol=0.0)

    baseline = make_tiny_model("none").eval()
    baseline_state = baseline.state_dict()
    baseline_output = baseline(audio, video)

    # With every new residual scale at zero, V1 and V2 must recover the
    # final-layer-only baseline path after loading the same legacy state.
    v1 = make_tiny_model("v1").eval()
    load_baseline_state(v1, baseline_state)
    v1_output = v1(audio, video)
    assert torch.allclose(v1_output, baseline_output, atol=1e-6, rtol=1e-6)

    v2 = make_tiny_model("v2").eval()
    load_baseline_state(v2, baseline_state)
    v2_output, diagnostics = v2.forward_with_hierarchical_diagnostics(
        audio, video
    )
    assert torch.allclose(v2_output, baseline_output, atol=1e-6, rtol=1e-6)
    assert diagnostics["video_layers"] == (1, 3, 5)
    assert diagnostics["aggregation_scales"].shape == (2,)
    assert diagnostics["stage_residual_scales"].shape == (3,)
    assert len(diagnostics["stage_gates"]) == 3
    assert all(torch.isfinite(gate).all() for gate in diagnostics["stage_gates"])

    # Non-zero hierarchy paths must affect the output and receive gradients.
    trainable = make_tiny_model("v2").eval()
    load_baseline_state(trainable, baseline_state)
    with torch.no_grad():
        trainable.hierarchical_visual_aggregator.residual_scales.fill_(0.1)
        for conditioner in trainable.hierarchical_stage_conditioners:
            conditioner.residual_scale.fill_(0.1)
    differentiable_audio = audio.clone().requires_grad_(True)
    differentiable_video = video.clone().requires_grad_(True)
    changed_output = trainable(differentiable_audio, differentiable_video)
    assert not torch.allclose(changed_output, baseline_output)
    assert torch.isfinite(changed_output).all()
    changed_output.square().mean().backward()
    assert differentiable_audio.grad is not None
    assert differentiable_video.grad is not None
    assert torch.isfinite(differentiable_audio.grad).all()
    assert torch.isfinite(differentiable_video.grad).all()
    assert trainable.hierarchical_visual_aggregator.residual_scales.grad is not None
    for conditioner in trainable.hierarchical_stage_conditioners:
        assert conditioner.residual_scale.grad is not None
        assert conditioner.video_proj.weight.grad is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Parse and validate V1/V2 YAML without importing torch or forwarding.",
    )
    args = parser.parse_args()

    check_configs()
    if args.config_only:
        print("EXP-008 hierarchical fusion config checks passed.")
        return
    check_tensor_paths()
    print("EXP-008 hierarchical fusion tensor checks passed.")


if __name__ == "__main__":
    main()
