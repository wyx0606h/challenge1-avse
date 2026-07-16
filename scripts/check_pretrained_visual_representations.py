"""Checks for EXP-009 pretrained visual representation experiments.

Use ``--config-only`` without AV-HuBERT assets. ``--with-avhubert`` is the
deferred target-server check that loads the external checkpoint and runs small
tensor/gradient checks.
"""
import argparse
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "v1": (
        ROOT / "configs" / "track2_av_convtasnet_pretrained_visual_v1.yml"
    ),
    "v2": (
        ROOT / "configs" / "track2_av_convtasnet_pretrained_visual_v2.yml"
    ),
}


def load_configs():
    configs = {}
    for version, path in CONFIGS.items():
        with path.open("r", encoding="utf-8") as handle:
            configs[version] = yaml.safe_load(handle)
    return configs


def check_configs(configs):
    for version, config in configs.items():
        model = config["audionet"]["audionet_config"]
        training = config["training"]
        data = config["datamodule"]["data_config"]

        assert model["visual_encoder_type"] == "avhubert"
        assert model["visual_adapter_out"] == model["E"] == 512
        assert model["hierarchical_fusion_version"] == "v1"
        assert model["hierarchical_video_layers"] == [1, 3, 5]
        assert model["D"] == 5
        assert model["audio_index"] == 1
        assert model["R"] == 4
        assert model["skip_con"] is False
        assert data["degrade_prob"] == 0.0
        assert "av_model.video." in training["freeze_param_prefixes"]
        assert "av_model.video" in training["force_eval_module_prefixes"]
        assert not any(
            prefix.startswith("video_model")
            for prefix in training["freeze_param_prefixes"]
        )

        if version == "v1":
            assert model["visual_feature_mode"] == "frontend"
            assert model["fusion_type"] == "concat"
        else:
            assert model["visual_feature_mode"] == "dual"
            assert model["visual_context_layer"] == 12
            assert model["visual_context_residual_init"] == 0.05
            assert model["fusion_type"] == "audio_anchored_gate"
            assert model["fusion_visual_residual_init"] == 0.05


def check_external_assets():
    for name in ("AVHUBERT_ROOT", "AVHUBERT_CKPT"):
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"{name} is not set")
        if not Path(value).expanduser().exists():
            raise FileNotFoundError(f"{name} does not exist: {value}")


def check_server_tensor_paths(configs):
    import torch

    from look2hear.models.av_convtasnet import AV_ConvTasNet

    torch.manual_seed(0)
    audio = torch.randn(1, 6400)
    mouth = torch.randn(1, 8, 88, 88)

    for version, config in configs.items():
        model_config = dict(config["audionet"]["audionet_config"])
        model = AV_ConvTasNet(sample_rate=16000, **model_config)
        external_names = (
            "visual_frontend",
            "context_encoder",
            "context_layer_norm",
            "context_post_extract_proj",
        )
        for name, parameter in model.video_model.named_parameters():
            if name.startswith(external_names):
                assert not parameter.requires_grad, name
        adapter_parameters = [
            parameter
            for name, parameter in model.video_model.named_parameters()
            if not name.startswith(external_names)
        ]
        assert adapter_parameters
        assert all(parameter.requires_grad for parameter in adapter_parameters)

        model.train()
        output, diagnostics = model.forward_with_hierarchical_diagnostics(
            audio, mouth
        )
        assert output.shape[0] == audio.shape[0]
        assert torch.isfinite(output).all()
        assert "pretrained_visual" in diagnostics
        if version == "v2":
            visual = diagnostics["pretrained_visual"]
            assert "context_gate" in visual
            assert torch.isfinite(visual["context_gate"]).all()
            assert "fusion" in diagnostics
            assert torch.isfinite(
                diagnostics["fusion"]["visual_gate"]
            ).all()

        output.square().mean().backward()
        assert model.video_model.local_adapter[1].projection.weight.grad is not None
        if version == "v2":
            assert model.video_model.context_residual_scale.grad is not None
            assert model.av_model.concat.visual_residual_scale.grad is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Validate YAML without importing torch or loading AV-HuBERT.",
    )
    parser.add_argument(
        "--with-avhubert",
        action="store_true",
        help="Load external AV-HuBERT assets and run deferred tensor checks.",
    )
    args = parser.parse_args()

    configs = load_configs()
    check_configs(configs)
    if args.config_only:
        print("EXP-009 pretrained visual config checks passed.")
        return
    if not args.with_avhubert:
        raise SystemExit(
            "Use --config-only locally or --with-avhubert on the target server."
        )
    check_external_assets()
    check_server_tensor_paths(configs)
    print("EXP-009 pretrained visual server tensor checks passed.")


if __name__ == "__main__":
    main()
