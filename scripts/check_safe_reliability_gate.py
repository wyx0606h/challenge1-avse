"""Focused sanity checks for EXP-004 safe reliability fusion.

This script uses synthetic tensors only. It does not require challenge data or
pretrained weights and does not report an experiment metric.
"""
import math

import torch

from look2hear.models.av_convtasnet import Concat, SafeReliabilityGatedFusion


def copy_concat_projection_weights(concat, safe_gate):
    """Split a baseline concat kernel into equivalent audio/video projections."""
    audio_channels = safe_gate.audio_channels
    with torch.no_grad():
        safe_gate.audio_proj.weight.copy_(
            concat.conv1d.weight[:, :audio_channels, :]
        )
        safe_gate.audio_proj.bias.copy_(concat.conv1d.bias)
        safe_gate.video_proj.weight.copy_(
            concat.conv1d.weight[:, audio_channels:, :]
        )


def set_constant_gate(safe_gate, bias):
    """Force a constant gate value for boundary tests."""
    final_conv = safe_gate.gate_net[-2]
    with torch.no_grad():
        final_conv.weight.zero_()
        final_conv.bias.fill_(bias)


def main():
    torch.manual_seed(0)
    batch, audio_channels, video_channels, out_channels = 2, 4, 3, 5
    audio_steps, video_steps = 11, 4
    audio = torch.randn(batch, audio_channels, audio_steps, requires_grad=True)
    video = torch.randn(batch, video_channels, video_steps, requires_grad=True)

    safe_gate = SafeReliabilityGatedFusion(
        audio_channels,
        video_channels,
        out_channels,
        gate_hidden=7,
        gate_init_bias=-2.0,
        scalar_gate=True,
    )
    fused, gate = safe_gate.forward_with_gate(audio, video)

    assert fused.shape == (batch, out_channels, audio_steps)
    assert gate.shape == (batch, 1, audio_steps)
    expected_gate = torch.full_like(gate, torch.sigmoid(torch.tensor(-2.0)))
    assert torch.allclose(gate, expected_gate, atol=1e-7, rtol=0.0)

    loss = fused.square().mean()
    loss.backward()
    assert audio.grad is not None and torch.isfinite(audio.grad).all()
    assert video.grad is not None and torch.isfinite(video.grad).all()

    # With a fully open gate, split projections must reproduce baseline concat.
    concat = Concat(audio_channels, video_channels, out_channels)
    copy_concat_projection_weights(concat, safe_gate)
    set_constant_gate(safe_gate, bias=20.0)
    with torch.no_grad():
        baseline_output = concat(audio.detach(), video.detach())
        open_gate_output = safe_gate(audio.detach(), video.detach())
    assert torch.allclose(open_gate_output, baseline_output, atol=1e-6, rtol=1e-6)

    # With a closed gate, changing video must not change the fused output.
    set_constant_gate(safe_gate, bias=-20.0)
    with torch.no_grad():
        closed_a = safe_gate(audio.detach(), video.detach())
        closed_b = safe_gate(audio.detach(), video.detach() * 100.0)
    assert torch.allclose(closed_a, closed_b, atol=1e-6, rtol=1e-6)

    parameter_count = sum(parameter.numel() for parameter in safe_gate.parameters())
    assert parameter_count > 0 and math.isfinite(float(parameter_count))
    print("EXP-004 safe reliability gate checks passed.")


if __name__ == "__main__":
    main()
