"""Frozen AV-HuBERT visual representations with small trainable adapters.

The optional AV-HuBERT/fairseq dependency is imported only when this class is
instantiated. This keeps the official baseline and local static checks usable
without cloning AV-HuBERT or downloading its checkpoint.
"""
import os
import sys
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_external_path(explicit, environment_name, description):
    value = explicit or os.environ.get(environment_name)
    if not value:
        raise RuntimeError(
            f"{description} is required for the AV-HuBERT visual encoder. "
            f"Pass it explicitly or set {environment_name}."
        )
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def _load_avhubert_backbone(repository_root, checkpoint_path):
    """Load the AV-HuBERT pretraining backbone from an official checkpoint."""
    repository_root = Path(repository_root)
    fairseq_source = repository_root / "fairseq"
    for path in (repository_root, fairseq_source):
        path_text = str(path)
        if path.exists() and path_text not in sys.path:
            sys.path.insert(0, path_text)

    try:
        from fairseq import checkpoint_utils, utils
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "Could not import the AV-HuBERT-compatible fairseq stack. Follow "
            "the official AV-HuBERT environment instructions, then make sure "
            "AVHUBERT_ROOT points to the repository root."
        ) from error

    user_dir = (
        repository_root / "avhubert"
        if (repository_root / "avhubert").is_dir()
        else repository_root
    )
    try:
        utils.import_user_module(Namespace(user_dir=str(user_dir)))
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [str(checkpoint_path)]
        )
    except Exception as error:
        raise RuntimeError(
            "Failed to load the AV-HuBERT checkpoint. Verify the official "
            "repository revision, fairseq/hydra versions, checkpoint license, "
            f"and checkpoint integrity: {checkpoint_path}"
        ) from error

    if not models:
        raise RuntimeError(f"No model was loaded from {checkpoint_path}")
    model = models[0]

    # The official fine-tuned VSR checkpoint wraps the pretraining model as
    # ``encoder.w2v_model``. Also accept common pretraining/wrapper layouts so
    # the error is explicit if a different official checkpoint is selected.
    if hasattr(model, "encoder") and hasattr(model.encoder, "w2v_model"):
        backbone = model.encoder.w2v_model
    elif hasattr(model, "w2v_model"):
        backbone = model.w2v_model
    elif hasattr(model, "feature_extractor_video"):
        backbone = model
    else:
        raise RuntimeError(
            "Unsupported AV-HuBERT checkpoint layout: expected "
            "encoder.w2v_model, w2v_model, or feature_extractor_video."
        )

    required = ("feature_extractor_video", "encoder_embed_dim")
    missing = [name for name in required if not hasattr(backbone, name)]
    if missing:
        raise RuntimeError(
            "AV-HuBERT backbone is missing required attributes: "
            + ", ".join(missing)
        )
    return backbone


class AVHubertVideoModel(nn.Module):
    """Adapt frozen AV-HuBERT visual speech features to the AVSE separator.

    ``feature_mode='frontend'`` implements EXP-009 V1. It projects the official
    AV-HuBERT visual frontend output to ``output_dim``.

    ``feature_mode='dual'`` implements EXP-009 V2. It keeps the frontend
    projection as the base path and adds a small gated residual from a selected
    AV-HuBERT transformer layer.
    """

    def __init__(
        self,
        feature_mode="frontend",
        output_dim=512,
        context_layer=12,
        gate_hidden=128,
        context_residual_init=0.05,
        repository_root=None,
        checkpoint_path=None,
    ):
        super().__init__()
        feature_mode = str(feature_mode).lower()
        if feature_mode not in ("frontend", "dual"):
            raise ValueError(
                "feature_mode must be 'frontend' or 'dual', got "
                f"{feature_mode!r}"
            )
        output_dim = int(output_dim)
        context_layer = int(context_layer)
        gate_hidden = int(gate_hidden)
        context_residual_init = float(context_residual_init)
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if context_layer <= 0:
            raise ValueError(
                f"context_layer must be one-based and positive, got "
                f"{context_layer}"
            )
        if gate_hidden <= 0:
            raise ValueError(f"gate_hidden must be positive, got {gate_hidden}")
        if context_residual_init < 0:
            raise ValueError(
                "context_residual_init must be non-negative, got "
                f"{context_residual_init}"
            )

        repository_root = _resolve_external_path(
            repository_root,
            "AVHUBERT_ROOT",
            "AV-HuBERT repository root",
        )
        checkpoint_path = _resolve_external_path(
            checkpoint_path,
            "AVHUBERT_CKPT",
            "AV-HuBERT checkpoint",
        )
        backbone = _load_avhubert_backbone(
            repository_root,
            checkpoint_path,
        )

        self.feature_mode = feature_mode
        self.backend_out = output_dim
        self.frontend_out = int(backbone.encoder_embed_dim)
        self.context_layer = context_layer
        self.modality_fuse = str(
            getattr(backbone, "modality_fuse", "concat")
        ).lower()

        # Keep only the modules needed for visual-only extraction. This avoids
        # retaining the decoder/final projection and audio feature extractor
        # from a fine-tuned VSR wrapper.
        self.visual_frontend = backbone.feature_extractor_video
        if feature_mode == "dual":
            context_required = ("encoder", "layer_norm")
            missing = [
                name for name in context_required
                if not hasattr(backbone, name)
            ]
            if missing:
                raise RuntimeError(
                    "The selected AV-HuBERT checkpoint cannot provide "
                    "contextual features; missing: " + ", ".join(missing)
                )
            self.context_encoder = backbone.encoder
            self.context_layer_norm = backbone.layer_norm
            self.context_post_extract_proj = (
                backbone.post_extract_proj
                if getattr(backbone, "post_extract_proj", None) is not None
                else nn.Identity()
            )
            layers = getattr(self.context_encoder, "layers", None)
            if layers is not None and context_layer > len(layers):
                raise ValueError(
                    f"context_layer ({context_layer}) exceeds the AV-HuBERT "
                    f"encoder depth ({len(layers)})."
                )
        else:
            self.context_encoder = None
            self.context_layer_norm = None
            self.context_post_extract_proj = None

        self.local_adapter = nn.Sequential(
            nn.LayerNorm(self.frontend_out),
            _TemporalLinearAdapter(self.frontend_out, output_dim),
        )
        if feature_mode == "dual":
            self.context_adapter = nn.Sequential(
                nn.LayerNorm(self.frontend_out),
                _TemporalLinearAdapter(self.frontend_out, output_dim),
            )
            self.context_gate = nn.Sequential(
                nn.Conv1d(3 * output_dim, gate_hidden, 1),
                nn.PReLU(gate_hidden),
                nn.Conv1d(gate_hidden, output_dim, 1),
            )
            self.context_residual_scale = nn.Parameter(
                torch.tensor(context_residual_init, dtype=torch.float32)
            )
        else:
            self.context_adapter = None
            self.context_gate = None
            self.register_parameter("context_residual_scale", None)

        self._freeze_external_modules()

    def _external_modules(self):
        modules = [self.visual_frontend]
        if self.context_encoder is not None:
            modules.extend(
                [
                    self.context_encoder,
                    self.context_layer_norm,
                    self.context_post_extract_proj,
                ]
            )
        return modules

    def _freeze_external_modules(self):
        for module in self._external_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        """Train adapters while keeping every external AV-HuBERT module frozen."""
        super().train(mode)
        self._freeze_external_modules()
        return self

    def _extract_context(self, local_features):
        if self.context_encoder is None:
            return None
        if self.modality_fuse == "concat":
            audio_zeros = local_features.new_zeros(
                local_features.size(0),
                self.frontend_out,
                local_features.size(-1),
            )
            fused = torch.cat([audio_zeros, local_features], dim=1)
        elif self.modality_fuse == "add":
            fused = local_features
        else:
            raise RuntimeError(
                "Unsupported AV-HuBERT modality_fuse value: "
                f"{self.modality_fuse!r}"
            )

        fused = self.context_layer_norm(fused.transpose(1, 2))
        fused = self.context_post_extract_proj(fused)
        contextual, _ = self.context_encoder(
            fused,
            padding_mask=None,
            layer=self.context_layer - 1,
        )
        return contextual.transpose(1, 2)

    def _extract_frozen(self, mouth):
        if mouth.ndim != 5:
            raise RuntimeError(
                "AVHubertVideoModel expects [B, C, T, H, W], got "
                f"shape {tuple(mouth.shape)}"
            )
        with torch.no_grad():
            local = self.visual_frontend(mouth)
            context = (
                self._extract_context(local)
                if self.feature_mode == "dual"
                else None
            )
        return local, context

    def forward_with_diagnostics(self, mouth):
        local, context = self._extract_frozen(mouth)
        local_projected = self.local_adapter(local.transpose(1, 2)).transpose(
            1, 2
        )
        diagnostics = {
            "feature_mode": self.feature_mode,
            "local_feature_norm": local_projected.detach().float().norm(
                dim=1
            ).mean(),
        }
        if context is None:
            return local_projected, diagnostics

        context_projected = self.context_adapter(
            context.transpose(1, 2)
        ).transpose(1, 2)
        if context_projected.size(-1) != local_projected.size(-1):
            context_projected = F.interpolate(
                context_projected,
                size=local_projected.size(-1),
                mode="linear",
                align_corners=False,
            )
        gate = torch.sigmoid(
            self.context_gate(
                torch.cat(
                    [
                        local_projected,
                        context_projected,
                        torch.abs(local_projected - context_projected),
                    ],
                    dim=1,
                )
            )
        )
        scale = torch.tanh(self.context_residual_scale)
        output = local_projected + scale * gate * context_projected
        diagnostics.update(
            {
                "context_layer": self.context_layer,
                "context_gate": gate.detach(),
                "context_residual_scale": scale.detach(),
                "context_feature_norm": context_projected.detach().float().norm(
                    dim=1
                ).mean(),
            }
        )
        return output, diagnostics

    def forward(self, mouth):
        output, _ = self.forward_with_diagnostics(mouth)
        return output


class _TemporalLinearAdapter(nn.Module):
    """Apply a learned channel projection to a ``[B, T, C]`` tensor."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, features):
        return self.activation(self.projection(features))
