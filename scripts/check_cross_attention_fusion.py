"""Synthetic checks for the reviewed EXP-003 cross-attention fusion."""
import torch

from look2hear.models.av_convtasnet import CrossAttentionFusion


def main():
    torch.manual_seed(0)
    batch, audio_channels, video_channels = 2, 6, 5
    out_channels, heads = 8, 2
    audio_steps, video_steps, window = 37, 9, 2
    audio = torch.randn(batch, audio_channels, audio_steps, requires_grad=True)
    video = torch.randn(batch, video_channels, video_steps, requires_grad=True)

    fusion = CrossAttentionFusion(
        audio_channels,
        video_channels,
        out_channels,
        num_heads=heads,
        attention_window=window,
        position_bias_strength=0.1,
        residual_init=0.1,
    )
    fused, attention = fusion.forward_with_attention(audio, video)

    assert fused.shape == (batch, out_channels, audio_steps)
    assert attention.shape == (batch, heads, audio_steps, video_steps)
    assert torch.isfinite(fused).all() and torch.isfinite(attention).all()
    assert torch.allclose(
        attention.sum(dim=-1),
        torch.ones_like(attention.sum(dim=-1)),
        atol=1e-6,
        rtol=1e-6,
    )

    # Attention must use native video length and assign zero probability outside
    # the configured temporal window.
    bias = fusion._make_temporal_bias(
        audio_steps, video_steps, attention.device, attention.dtype
    )
    outside_window = torch.isneginf(bias)
    expanded_mask = outside_window[None, None, :, :].expand_as(attention)
    assert torch.count_nonzero(attention.masked_select(expanded_mask)) == 0
    assert audio_steps * video_steps < audio_steps * audio_steps

    loss = fused.square().mean()
    loss.backward()
    assert audio.grad is not None and torch.isfinite(audio.grad).all()
    assert video.grad is not None and torch.isfinite(video.grad).all()

    # A zero residual scale must recover the projected audio path exactly.
    with torch.no_grad():
        fusion.residual_scale.zero_()
        audio_only = fusion.audio_proj(audio.detach())
        zero_residual = fusion(audio.detach(), video.detach())
    assert torch.allclose(zero_residual, audio_only, atol=1e-6, rtol=1e-6)

    print("EXP-003 cross-attention fusion checks passed.")


if __name__ == "__main__":
    main()
