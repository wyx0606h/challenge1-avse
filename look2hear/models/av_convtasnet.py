"""
Audio-Visual Conv-TasNet Model

The original AV_model (Kai Li, 2021) and all of its building blocks are inlined
here so the model is fully self-contained inside the ``look2hear`` package and
needs no external module imports.

The ``AV_ConvTasNet`` wrapper bundles a video encoder together with the
audio-visual separator and stores every architectural initialization
hyper-parameter. The default remains the official frozen ResNet encoder.
Experiments can explicitly select an external frozen AV-HuBERT backbone with
small trainable adapters.

    model = AV_ConvTasNet.from_pretrain("best_model.pth")
    enhanced = model(mixture, lip_frames)   # raw lip frames, no separate video net

The legacy ResNet checkpoint remains self-contained. AV-HuBERT experiments
require the same external repository/checkpoint environment used at training
when reconstructing the model.
"""
import torch
import torch.nn as nn

from ..models.base import BaseModel
from ..videomodels import AVHubertVideoModel, ResNetVideoModel


# ---------- Basic Part -------------
class Conv1D(nn.Conv1d):
    """Applies a 1D convolution over an input signal composed of several input planes."""

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class GlobalLayerNorm(nn.Module):
    """
    Calculate Global Layer Normalization

    dim: (int or list or torch.Size) - input shape from an expected input of size
    eps: a value added to the denominator for numerical stability.
    elementwise_affine: when True, this module has learnable per-element affine
        parameters initialized to ones (weights) and zeros (biases).
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """
    Calculate Cumulative Layer Normalization

    dim: the dim you want to norm
    elementwise_affine: learnable per-element affine parameters
    """

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    else:
        return nn.BatchNorm1d(dim)


# ---------- Audio Part -------------
class Encoder(nn.Module):
    """
    Audio Encoder

    in_channels: Audio in_channels is 1
    out_channels: Encoder part output's channels
    kernel_size: Conv1D's kernel size
    stride: Conv1D's stride size
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Encoder, self).__init__()
        self.conv = Conv1D(in_channels, out_channels, kernel_size, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: [B, T]
        out: [B, N, T]
        """
        x = self.conv(x)
        x = self.relu(x)
        return x


class Decoder(nn.ConvTranspose1d):
    """
    Decoder

    This module can be seen as the gradient of Conv1d with respect to its input.
    It is also known as a fractionally-strided convolution or a deconvolution
    (although it is not an actual deconvolution operation).
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Audio_1DConv(nn.Module):
    """
    Audio part 1-D Conv Block

    in_channels: Encoder's output channels
    out_channels: 1DConv output channels
    b_conv: the B_conv channels
    sc_conv: the skip-connection channels
    kernel_size: the depthwise conv kernel size
    dilation: the depthwise conv dilation
    norm: 1D Conv normalization's type
    causal: Two choice(causal or noncausal)
    skip_con: Whether to use skip connection
    """

    def __init__(self,
                 in_channels=256,
                 out_channels=512,
                 b_conv=256,
                 sc_conv=256,
                 kernel_size=3,
                 dilation=1,
                 norm='gln',
                 causal=False,
                 skip_con=False):
        super(Audio_1DConv, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm, out_channels)
        self.pad = (dilation * (kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        self.dconv = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=self.pad, dilation=dilation, groups=out_channels)
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm, out_channels)
        self.B_conv = nn.Conv1d(out_channels, b_conv, 1)
        self.Sc_conv = nn.Conv1d(out_channels, sc_conv, 1)
        self.causal = causal
        self.skip_con = skip_con

    def forward(self, x):
        """
        x: [B, N, T]
        out: [B, N, T]
        """
        out = self.conv1x1(x)
        out = self.prelu1(out)
        out = self.norm1(out)
        out = self.dconv(out)
        if self.causal:
            out = out[:, :, :-self.pad]
        out = self.prelu2(self.norm2(out))
        if self.skip_con:
            skip = self.Sc_conv(out)
            B = self.B_conv(out)
            # [B, N, T]
            return skip, B + x
        else:
            B = self.B_conv(out)
            # [B, N, T]
            return B + x


class Audio_Sequential(nn.Module):
    def __init__(self, repeats, blocks,
                 in_channels=256,
                 out_channels=512,
                 b_conv=256,
                 sc_conv=256,
                 kernel_size=3,
                 norm='gln',
                 causal=False,
                 skip_con=False):
        super(Audio_Sequential, self).__init__()
        self.lists = nn.ModuleList([])
        self.skip_con = skip_con
        self.repeats = int(repeats)
        self.blocks = int(blocks)
        for r in range(repeats):
            for b in range(blocks):
                self.lists.append(Audio_1DConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    b_conv=b_conv,
                    sc_conv=sc_conv,
                    kernel_size=kernel_size,
                    dilation=(2 ** b),
                    norm=norm,
                    causal=causal,
                    skip_con=skip_con))

    def forward(self, x):
        """
        x: [B, N, T]
        out: [B, N, T]
        """
        if self.skip_con:
            skip_connection = 0
            for i in range(len(self.lists)):
                skip, out = self.lists[i](x)
                x = out
                skip_connection += skip
            return skip_connection
        else:
            for i in range(len(self.lists)):
                out = self.lists[i](x)
                x = out
            return x

    def forward_repeat(self, x, repeat_index):
        """Run one complete TCN repeat and return its residual state.

        Hierarchical V2 injects visual evidence only at repeat boundaries. The
        baseline ``forward`` remains unchanged; this focused method exposes the
        same ordered blocks without rebuilding the separator or renaming its
        checkpoint keys.
        """
        if self.skip_con:
            raise RuntimeError(
                "forward_repeat requires skip_con=False because stage fusion "
                "needs the residual state, not an accumulated skip output."
            )
        repeat_index = int(repeat_index)
        if repeat_index < 0 or repeat_index >= self.repeats:
            raise IndexError(
                f"repeat_index must be in [0, {self.repeats}), got "
                f"{repeat_index}"
            )
        start = repeat_index * self.blocks
        end = start + self.blocks
        for block in self.lists[start:end]:
            x = block(x)
        return x


# ---------- Video Part -------------
class Video_1Dconv(nn.Module):
    """
    Video part 1-D Conv Block

    in_channels: video Encoder output channels
    conv_channels: dconv channels
    kernel_size: the depthwise conv kernel size
    dilation: the depthwise conv dilation
    residual: Whether to use residual connection
    skip_con: Whether to use skip connection
    first_block: first block, not residual
    """

    def __init__(self,
                 in_channels,
                 conv_channels,
                 kernel_size,
                 dilation=1,
                 residual=True,
                 skip_con=True,
                 first_block=True):
        super(Video_1Dconv, self).__init__()
        self.first_block = first_block
        # first block, not residual
        self.residual = residual and not first_block
        self.bn = nn.BatchNorm1d(in_channels) if not first_block else None
        self.relu = nn.ReLU() if not first_block else None
        self.dconv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            dilation=dilation,
            padding=(dilation * (kernel_size - 1)) // 2,
            bias=True)
        self.bconv = nn.Conv1d(in_channels, conv_channels, 1)
        self.sconv = nn.Conv1d(in_channels, conv_channels, 1)
        self.skip_con = skip_con

    def forward(self, x):
        """
        x: [B, N, T]
        out: [B, N, T]
        """
        if not self.first_block:
            y = self.bn(self.relu(x))
            y = self.dconv(y)
        else:
            y = self.dconv(x)
        # skip connection
        if self.skip_con:
            skip = self.sconv(y)
            if self.residual:
                y = y + x
                return skip, y
            else:
                return skip, y
        else:
            y = self.bconv(y)
            if self.residual:
                y = y + x
                return y
            else:
                return y


class Video_Sequential(nn.Module):
    """
    All the Video Part

    in_channels: front3D part in_channels
    out_channels: Video Conv1D part out_channels
    kernel_size: the kernel size of Video Conv1D
    skip_con: skip connection
    repeat: Conv1D repeats
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 skip_con=True,
                 repeat=5):
        super(Video_Sequential, self).__init__()
        self.conv1d_list = nn.ModuleList([])
        self.skip_con = skip_con
        self.repeat = int(repeat)
        for i in range(repeat):
            in_channels = out_channels if i else in_channels
            self.conv1d_list.append(
                Video_1Dconv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    skip_con=skip_con,
                    residual=True,
                    first_block=(i == 0)))

    def forward(self, x):
        """
        x: [B, N, T]
        out: [B, N, T]
        """
        final, _ = self.forward_with_layers(x, ())
        return final

    def forward_with_layers(self, x, layer_indices):
        """Return the normal output plus selected one-based block states.

        The selected tensors are residual block outputs, even when the legacy
        ``skip_con`` mode is enabled. This gives every requested depth the same
        channel semantics while preserving the original final-output behavior.
        """
        layer_indices = tuple(int(index) for index in layer_indices)
        if tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError(
                "layer_indices must be sorted unique one-based indices, got "
                f"{layer_indices}"
            )
        if layer_indices and (
            layer_indices[0] < 1 or layer_indices[-1] > len(self.conv1d_list)
        ):
            raise ValueError(
                "layer_indices must be within [1, {}], got {}".format(
                    len(self.conv1d_list), layer_indices
                )
            )

        requested = set(layer_indices)
        captured = {}
        skip_connection = 0
        for index, block in enumerate(self.conv1d_list, start=1):
            if self.skip_con:
                skip, out = block(x)
                skip_connection = skip_connection + skip
            else:
                out = block(x)
            x = out
            if index in requested:
                captured[index] = x

        final = skip_connection if self.skip_con else x
        return final, tuple(captured[index] for index in layer_indices)


class HierarchicalVisualAggregator(nn.Module):
    """V1: add shallow/mid V-TCN evidence to the deepest visual state.

    The deepest selected feature is an unchanged base path. Earlier features
    pass through identity-initialized 1x1 adapters and bounded residual scales.
    Zero scales therefore recover the previous final-layer-only representation
    exactly, which makes warm-start comparisons interpretable.
    """

    def __init__(self, channels, num_levels, residual_init=0.05):
        super(HierarchicalVisualAggregator, self).__init__()
        channels = int(channels)
        num_levels = int(num_levels)
        residual_init = float(residual_init)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_levels < 2:
            raise ValueError(f"num_levels must be at least 2, got {num_levels}")
        if residual_init < 0:
            raise ValueError(
                f"residual_init must be non-negative, got {residual_init}"
            )

        self.channels = channels
        self.num_levels = num_levels
        self.adapters = nn.ModuleList(
            nn.Conv1d(channels, channels, 1, bias=False)
            for _ in range(num_levels - 1)
        )
        self.residual_scales = nn.Parameter(
            torch.full((num_levels - 1,), residual_init)
        )
        self._reset_adapters_to_identity()

    def _reset_adapters_to_identity(self):
        with torch.no_grad():
            for adapter in self.adapters:
                adapter.weight.zero_()
                diagonal = torch.arange(self.channels)
                adapter.weight[diagonal, diagonal, 0] = 1.0

    def effective_scales(self):
        """Return bounded scalar weights used by the shallow residual paths."""
        return torch.tanh(self.residual_scales)

    def forward(self, features):
        features = tuple(features)
        if len(features) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} visual levels, got {len(features)}"
            )
        reference_shape = features[-1].shape
        for feature in features:
            if feature.dim() != 3 or feature.shape != reference_shape:
                raise RuntimeError(
                    "All hierarchical visual features must share [B, C, T]; "
                    f"got {[tuple(item.shape) for item in features]}"
                )
            if feature.size(1) != self.channels:
                raise RuntimeError(
                    f"Expected {self.channels} visual channels, got "
                    f"{feature.size(1)}"
                )

        output = features[-1]
        for scale, adapter, feature in zip(
            self.effective_scales(), self.adapters, features[:-1]
        ):
            output = output + scale * adapter(feature)
        return output


class HierarchicalStageConditioner(nn.Module):
    """V2: inject one visual depth through a reliability-gated residual."""

    def __init__(self, audio_channels, video_channels, gate_hidden=128,
                 residual_init=0.05):
        super(HierarchicalStageConditioner, self).__init__()
        audio_channels = int(audio_channels)
        video_channels = int(video_channels)
        gate_hidden = int(gate_hidden)
        residual_init = float(residual_init)
        if min(audio_channels, video_channels, gate_hidden) <= 0:
            raise ValueError(
                "audio_channels, video_channels, and gate_hidden must be positive"
            )
        if residual_init < 0:
            raise ValueError(
                f"residual_init must be non-negative, got {residual_init}"
            )

        self.audio_channels = audio_channels
        self.video_channels = video_channels
        self.video_proj = nn.Conv1d(
            video_channels, audio_channels, 1, bias=False
        )
        self.gate = nn.Sequential(
            nn.Conv1d(audio_channels * 3, gate_hidden, 1),
            nn.PReLU(),
            nn.Conv1d(gate_hidden, audio_channels, 1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(residual_init))
        self._reset_stable()

    def _reset_stable(self):
        with torch.no_grad():
            if self.audio_channels == self.video_channels:
                self.video_proj.weight.zero_()
                diagonal = torch.arange(self.audio_channels)
                self.video_proj.weight[diagonal, diagonal, 0] = 1.0
            else:
                nn.init.xavier_uniform_(self.video_proj.weight)
            # Start with a neutral 0.5 gate. The small residual scale limits its
            # effect while allowing the gate to specialize during fine-tuning.
            self.gate[2].weight.zero_()
            self.gate[2].bias.zero_()

    def forward_with_gate(self, audio, video):
        if audio.dim() != 3 or video.dim() != 3:
            raise RuntimeError("audio and video must both be [B, C, T] tensors")
        if audio.size(1) != self.audio_channels:
            raise RuntimeError(
                f"Expected {self.audio_channels} audio channels, got "
                f"{audio.size(1)}"
            )
        if video.size(1) != self.video_channels:
            raise RuntimeError(
                f"Expected {self.video_channels} video channels, got "
                f"{video.size(1)}"
            )

        aligned_video = torch.nn.functional.interpolate(
            video, size=audio.size(-1)
        )
        projected_video = self.video_proj(aligned_video)
        gate_input = torch.cat(
            [audio, projected_video, torch.abs(audio - projected_video)], dim=1
        )
        visual_gate = self.gate(gate_input)
        output = (
            audio
            + torch.tanh(self.residual_scale) * visual_gate * projected_video
        )
        return output, visual_gate

    def forward(self, audio, video):
        output, _ = self.forward_with_gate(audio, video)
        return output


class Concat(nn.Module):
    """
    Baseline audio-visual fusion by direct concatenation.

    audio_channels: Audio Part Channels
    video_channels: Video Part Channels
    out_channels: Concat Net channels
    """

    def __init__(self, audio_channels, video_channels, out_channels):
        super(Concat, self).__init__()
        self.audio_channels = audio_channels
        self.video_channels = video_channels
        # project
        self.conv1d = nn.Conv1d(audio_channels + video_channels, out_channels, 1)

    def forward(self, a, v):
        """
        a: audio features, N x A x Ta
        v: video features, N x V x Tv
        """
        if a.size(1) != self.audio_channels or v.size(1) != self.video_channels:
            raise RuntimeError("Dimention mismatch for audio/video features, "
                               "{:d}/{:d} vs {:d}/{:d}".format(
                                   a.size(1), v.size(1), self.audio_channels,
                                   self.video_channels))
        # up-sample video features
        v = torch.nn.functional.interpolate(v, size=a.size(-1))
        # concat: n x (A+V) x Ta
        y = torch.cat([a, v], dim=1)
        # conv1d
        return self.conv1d(y)


class ReliabilityGatedFusion(nn.Module):
    """
    Reliability-aware fusion for Track 2 visual degradation experiments.

    The baseline treats every visual frame as equally useful after temporal
    alignment. Track 2 breaks that assumption: lips may be occluded, frozen,
    low-resolution, dropped, or desynchronized. This module first projects
    audio/video streams to a common channel width, then predicts a per-time,
    per-channel gate from their joint evidence. The gate controls how much
    visual information is injected into the audio stream.

    A small and inspectable gate is intentional here. It keeps the first
    experiment close to AV-ConvTasNet while making the learned visual reliance
    easy to trace later.
    """

    def __init__(self, audio_channels, video_channels, out_channels,
                 gate_hidden=128):
        super(ReliabilityGatedFusion, self).__init__()
        self.audio_channels = audio_channels
        self.video_channels = video_channels
        self.out_channels = out_channels

        gate_hidden = int(gate_hidden)
        if gate_hidden <= 0:
            raise ValueError(f"gate_hidden must be positive, got {gate_hidden}")

        self.audio_proj = nn.Conv1d(audio_channels, out_channels, 1)
        self.video_proj = nn.Conv1d(video_channels, out_channels, 1)

        # The absolute difference is a cheap mismatch cue. Large differences can
        # indicate unreliable visual guidance, AV desync, or irrelevant motion.
        self.gate = nn.Sequential(
            nn.Conv1d(out_channels * 3, gate_hidden, 1),
            nn.PReLU(),
            nn.Conv1d(gate_hidden, out_channels, 1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, a, v):
        """
        a: audio features, N x A x Ta
        v: video features, N x V x Tv
        """
        if a.size(1) != self.audio_channels or v.size(1) != self.video_channels:
            raise RuntimeError("Dimention mismatch for audio/video features, "
                               "{:d}/{:d} vs {:d}/{:d}".format(
                                   a.size(1), v.size(1), self.audio_channels,
                                   self.video_channels))

        v = torch.nn.functional.interpolate(v, size=a.size(-1))
        a_proj = self.audio_proj(a)
        v_proj = self.video_proj(v)

        gate_in = torch.cat([a_proj, v_proj, torch.abs(a_proj - v_proj)], dim=1)
        visual_gate = self.gate(gate_in)

        # Residual audio path protects the model when visual evidence is poor.
        fused = a_proj + visual_gate * v_proj
        return self.out_proj(fused)


class AudioAnchoredVisualFusion(nn.Module):
    """Audio-first visual fusion with baseline-compatible projection weights.

    The original concat projection remains at ``conv1d`` with the same key and
    tensor shape, so an EXP-008 checkpoint initializes it exactly. A bounded
    gate scales only the visual input before that projection. Zero visual scale
    therefore keeps the checkpoint's learned audio slice and bias while
    removing unreliable visual evidence.
    """

    def __init__(
        self,
        audio_channels,
        video_channels,
        out_channels,
        gate_hidden=128,
        visual_residual_init=0.05,
    ):
        super(AudioAnchoredVisualFusion, self).__init__()
        self.audio_channels = int(audio_channels)
        self.video_channels = int(video_channels)
        self.out_channels = int(out_channels)
        gate_hidden = int(gate_hidden)
        visual_residual_init = float(visual_residual_init)
        if gate_hidden <= 0:
            raise ValueError(f"gate_hidden must be positive, got {gate_hidden}")
        if visual_residual_init < 0:
            raise ValueError(
                "visual_residual_init must be non-negative, got "
                f"{visual_residual_init}"
            )

        # Keep this name/shape identical to Concat for exact warm-start loading.
        self.conv1d = nn.Conv1d(
            self.audio_channels + self.video_channels,
            self.out_channels,
            1,
        )
        self.audio_gate_proj = nn.Conv1d(
            self.audio_channels, gate_hidden, 1
        )
        self.video_gate_proj = nn.Conv1d(
            self.video_channels, gate_hidden, 1
        )
        self.gate = nn.Sequential(
            nn.Conv1d(gate_hidden * 3, gate_hidden, 1),
            nn.PReLU(gate_hidden),
            nn.Conv1d(gate_hidden, self.video_channels, 1),
            nn.Sigmoid(),
        )
        self.visual_residual_scale = nn.Parameter(
            torch.tensor(visual_residual_init, dtype=torch.float32)
        )

    def forward_with_diagnostics(self, a, v):
        if a.size(1) != self.audio_channels or v.size(1) != self.video_channels:
            raise RuntimeError(
                "Dimention mismatch for audio/video features, "
                "{:d}/{:d} vs {:d}/{:d}".format(
                    a.size(1),
                    v.size(1),
                    self.audio_channels,
                    self.video_channels,
                )
            )
        v = torch.nn.functional.interpolate(v, size=a.size(-1))
        audio_gate = self.audio_gate_proj(a)
        video_gate = self.video_gate_proj(v)
        gate = self.gate(
            torch.cat(
                [
                    audio_gate,
                    video_gate,
                    torch.abs(audio_gate - video_gate),
                ],
                dim=1,
            )
        )
        scale = torch.tanh(self.visual_residual_scale)
        scaled_video = scale * gate * v
        output = self.conv1d(torch.cat([a, scaled_video], dim=1))
        diagnostics = {
            "visual_gate": gate,
            "visual_residual_scale": scale,
        }
        return output, diagnostics

    def forward(self, a, v):
        output, _ = self.forward_with_diagnostics(a, v)
        return output


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion: audio queries aligned visual evidence.

    Direct concatenation asks the following separator blocks to discover the
    audio-visual relationship by themselves. Cross attention makes that
    relationship explicit: each audio time step forms a query and attends over
    the time-aligned visual sequence. This is a compact AV-ConvTasNet-friendly
    version of recent AVSS/AVTSE attention fusion ideas.
    """

    def __init__(self, audio_channels, video_channels, out_channels,
                 num_heads=4, dropout=0.0):
        super(CrossAttentionFusion, self).__init__()
        self.audio_channels = audio_channels
        self.video_channels = video_channels
        self.out_channels = out_channels

        num_heads = int(num_heads)
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if out_channels % num_heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.audio_proj = nn.Conv1d(audio_channels, out_channels, 1)
        self.video_proj = nn.Conv1d(video_channels, out_channels, 1)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.out_proj = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, a, v):
        """
        a: audio features, N x A x Ta
        v: video features, N x V x Tv
        """
        if a.size(1) != self.audio_channels or v.size(1) != self.video_channels:
            raise RuntimeError("Dimention mismatch for audio/video features, "
                               "{:d}/{:d} vs {:d}/{:d}".format(
                                   a.size(1), v.size(1), self.audio_channels,
                                   self.video_channels))

        v = torch.nn.functional.interpolate(v, size=a.size(-1))
        a_proj = self.audio_proj(a)
        v_proj = self.video_proj(v)

        # MultiheadAttention expects [batch, time, channel] when batch_first=True.
        query = a_proj.transpose(1, 2)
        key_value = v_proj.transpose(1, 2)
        attended, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False,
        )
        fused = self.norm(query + attended)
        return self.out_proj(fused.transpose(1, 2))


def make_fusion_module(
    fusion_type,
    audio_channels,
    video_channels,
    out_channels,
    gate_hidden=128,
    num_heads=4,
    dropout=0.0,
    visual_residual_init=0.05,
):
    """Create the selected audio-visual fusion module."""
    fusion_type = (fusion_type or "concat").lower()
    if fusion_type == "concat":
        return Concat(audio_channels, video_channels, out_channels)
    if fusion_type == "reliability_gate":
        return ReliabilityGatedFusion(
            audio_channels,
            video_channels,
            out_channels,
            gate_hidden=gate_hidden,
        )
    if fusion_type == "audio_anchored_gate":
        return AudioAnchoredVisualFusion(
            audio_channels,
            video_channels,
            out_channels,
            gate_hidden=gate_hidden,
            visual_residual_init=visual_residual_init,
        )
    if fusion_type == "cross_attention":
        return CrossAttentionFusion(
            audio_channels,
            video_channels,
            out_channels,
            num_heads=num_heads,
            dropout=dropout,
        )
    raise ValueError(
        "Unsupported fusion_type {!r}. Expected one of: concat, "
        "reliability_gate, audio_anchored_gate, cross_attention.".format(
            fusion_type
        )
    )


class AV_model(nn.Module):
    """
    Audio and Visual Speech Separation

    Audio Part
        N   Number of filters in autoencoder
        L   Length of the filters (in samples)
        B   Number of channels in bottleneck and the residual paths' 1x1-conv blocks
        SC  Number of channels in skip-connection paths' 1x1-conv blocks
        H   Number of channels in convolutional blocks
        P   Kernel size in convolutional blocks
        X   Number of convolutional blocks in each repeat
    Video Part
        E   Number of filters in video autoencoder
        V   Number of channels in convolutional blocks
        K   Kernel size in convolutional blocks
        D   Number of repeats
    Concat Part
        F   Number of channels in convolutional blocks
    Other Setting
        R   Number of all repeats
        skip_con    Skip Connection
        audio_index Number repeats of audio part
        norm    Normalization type
        causal  Two choice(causal or noncausal)
        fusion_type Direct concat, reliability gate, or cross attention
        hierarchical_fusion_version Disabled, V1 aggregation, or V2 staged gates
    """

    def __init__(
            self,
            # audio conf
            N=256,
            L=40,
            B=256,
            Sc=256,
            H=512,
            P=3,
            X=8,
            # video conf
            E=256,
            V=256,
            K=3,
            D=5,
            # fusion index
            F=256,
            # other
            R=4,
            skip_con=False,
            audio_index=2,
            norm="gln",
            causal=False,
            fusion_type="concat",
            fusion_gate_hidden=128,
            fusion_heads=4,
            fusion_dropout=0.0,
            fusion_visual_residual_init=0.05,
            hierarchical_fusion_version="none",
            hierarchical_video_layers=(1, 3, 5),
            hierarchical_residual_init=0.05,
            hierarchical_gate_hidden=128,
            hierarchical_stage_residual_init=0.05):
        super(AV_model, self).__init__()
        hierarchical_fusion_version = str(
            hierarchical_fusion_version or "none"
        ).lower()
        if hierarchical_fusion_version not in ("none", "v1", "v2"):
            raise ValueError(
                "hierarchical_fusion_version must be one of none/v1/v2, got "
                f"{hierarchical_fusion_version!r}"
            )
        hierarchical_video_layers = tuple(
            int(index) for index in hierarchical_video_layers
        )
        if hierarchical_fusion_version != "none":
            if tuple(sorted(set(hierarchical_video_layers))) != (
                hierarchical_video_layers
            ):
                raise ValueError(
                    "hierarchical_video_layers must be sorted unique one-based "
                    f"indices, got {hierarchical_video_layers}"
                )
            if len(hierarchical_video_layers) < 2:
                raise ValueError(
                    "Hierarchical fusion requires at least two visual layers"
                )
            if (
                hierarchical_video_layers[0] < 1
                or hierarchical_video_layers[-1] > D
            ):
                raise ValueError(
                    f"hierarchical_video_layers must be within [1, {D}], got "
                    f"{hierarchical_video_layers}"
                )
            if hierarchical_video_layers[-1] != D:
                raise ValueError(
                    "The deepest hierarchical layer must equal D so the base "
                    "path recovers the existing final V-TCN output"
                )
        if hierarchical_fusion_version == "v2":
            if skip_con:
                raise ValueError(
                    "hierarchical_fusion_version='v2' requires skip_con=False"
                )
            required_post_repeats = len(hierarchical_video_layers) - 1
            if R - audio_index < required_post_repeats:
                raise ValueError(
                    "V2 needs at least one post-fusion repeat boundary for each "
                    "non-local stage: R - audio_index must be >= {}, got {}"
                    .format(required_post_repeats, R - audio_index)
                )

        self.hierarchical_fusion_version = hierarchical_fusion_version
        self.hierarchical_video_layers = hierarchical_video_layers
        self.video = Video_Sequential(E, V, K, skip_con=skip_con, repeat=D)
        if hierarchical_fusion_version != "none":
            self.hierarchical_visual_aggregator = HierarchicalVisualAggregator(
                V,
                len(hierarchical_video_layers),
                residual_init=hierarchical_residual_init,
            )
        else:
            self.hierarchical_visual_aggregator = None
        # n x S > n x N x T
        self.encoder = Encoder(1, N, L, stride=L // 2)
        # before repeat blocks, always cLN
        self.cln = CumulativeLayerNorm(N)
        # n x N x T > n x B x T
        self.conv1x1 = Conv1D(N, B, 1)
        # repeat blocks
        # n x B x T => n x B x T
        self.skip_con = skip_con
        self.audio_conv = Audio_Sequential(
            audio_index,
            X,
            in_channels=B,
            out_channels=H,
            b_conv=B,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        self.concat = make_fusion_module(
            fusion_type,
            B,
            V,
            F,
            gate_hidden=fusion_gate_hidden,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
            visual_residual_init=fusion_visual_residual_init,
        )
        self.feats_conv = Audio_Sequential(
            R - audio_index,
            X,
            in_channels=F,
            out_channels=H,
            b_conv=F,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        if hierarchical_fusion_version == "v2":
            stage_channels = [B] + [F] * (len(hierarchical_video_layers) - 1)
            self.hierarchical_stage_conditioners = nn.ModuleList(
                HierarchicalStageConditioner(
                    audio_channels=channels,
                    video_channels=V,
                    gate_hidden=hierarchical_gate_hidden,
                    residual_init=hierarchical_stage_residual_init,
                )
                for channels in stage_channels
            )
        else:
            self.hierarchical_stage_conditioners = nn.ModuleList([])
        # mask 1x1 conv
        # n x F x T => n x N x T
        self.mask = Conv1D(F, N, 1)
        # n x N x T => n x 1 x To
        self.decoder = Decoder(N, 1, kernel_size=L, stride=L // 2, bias=True)

    def check_forward_args(self, x, v):
        if x.dim() != 2:
            raise RuntimeError(
                "{} accept 1/2D tensor as audio input, but got {:d}".format(
                    self.__class__.__name__, x.dim()))
        if v.dim() != 3:
            raise RuntimeError(
                "{} accept 2/3D tensor as video input, but got {:d}".format(
                    self.__class__.__name__, v.dim()))
        if x.size(0) != v.size(0):
            raise RuntimeError(
                "auxiliary input do not have same batch size with input chunk, {:d} vs {:d}"
                .format(x.size(0), v.size(0)))

    def _forward_impl(self, x, v, return_diagnostics=False):
        """
        x: raw waveform chunks, N x C
        v: time variant lip embeddings, N x T x D
        """
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
            v = torch.unsqueeze(v, 0)
        # check args
        self.check_forward_args(x, v)

        # n x 1 x S => n x N x T
        w = self.encoder(x)
        # n x B x T
        a = self.conv1x1(self.cln(w))
        # audio feats: n x B x T
        a = self.audio_conv(a)

        diagnostics = {}
        if self.hierarchical_fusion_version == "none":
            # lip embeddings: N x T x D => N x V x T
            fused_video = self.video(v)
            video_levels = ()
        else:
            _, video_levels = self.video.forward_with_layers(
                v, self.hierarchical_video_layers
            )
            fused_video = self.hierarchical_visual_aggregator(video_levels)
            if return_diagnostics:
                diagnostics["video_layers"] = self.hierarchical_video_layers
                diagnostics["aggregation_scales"] = (
                    self.hierarchical_visual_aggregator.effective_scales()
                )

        if self.hierarchical_fusion_version == "v2":
            if return_diagnostics:
                a, gate = self.hierarchical_stage_conditioners[
                    0
                ].forward_with_gate(a, video_levels[0])
                stage_gates = [gate]
            else:
                a = self.hierarchical_stage_conditioners[0](a, video_levels[0])
                stage_gates = None

        # audio/video fusion
        if return_diagnostics and hasattr(
            self.concat, "forward_with_diagnostics"
        ):
            y, fusion_diagnostics = self.concat.forward_with_diagnostics(
                a, fused_video
            )
            diagnostics["fusion"] = fusion_diagnostics
        else:
            y = self.concat(a, fused_video)

        # fused audio-visual features: n x F x T
        if self.hierarchical_fusion_version == "v2":
            for repeat_index in range(self.feats_conv.repeats):
                y = self.feats_conv.forward_repeat(y, repeat_index)
                stage_index = repeat_index + 1
                if stage_index < len(video_levels):
                    conditioner = self.hierarchical_stage_conditioners[
                        stage_index
                    ]
                    if return_diagnostics:
                        y, gate = conditioner.forward_with_gate(
                            y, video_levels[stage_index]
                        )
                        stage_gates.append(gate)
                    else:
                        y = conditioner(y, video_levels[stage_index])
            if return_diagnostics:
                diagnostics["stage_gates"] = tuple(stage_gates)
                diagnostics["stage_residual_scales"] = torch.stack(
                    [
                        torch.tanh(conditioner.residual_scale)
                        for conditioner in self.hierarchical_stage_conditioners
                    ]
                )
        else:
            y = self.feats_conv(y)
        # n x N x T
        m = torch.nn.functional.relu(self.mask(y))
        # n x To
        output = self.decoder(w * m)
        return output, diagnostics

    def forward(self, x, v):
        output, _ = self._forward_impl(x, v, return_diagnostics=False)
        return output

    def forward_with_hierarchical_diagnostics(self, x, v):
        """Return enhanced audio and optional V1/V2 scale/gate diagnostics."""
        return self._forward_impl(x, v, return_diagnostics=True)


class AV_ConvTasNet(BaseModel):
    """
    Audio-Visual Conv-TasNet for Speech Enhancement.

    The default path bundles the frozen legacy ResNet lip encoder exactly as
    before. EXP-009 can opt into a frozen external AV-HuBERT frontend/context
    encoder with small trainable adapters. In both cases the forward pass takes
    raw lip frames and runs visual extraction internally.

    Args:
        N..causal: audio/video/fusion hyper-parameters of :class:`AV_model`.
        video_relu_type: activation of the ResNet video encoder ('relu'/'prelu').
        video_pretrain: path to the ResNet backbone weights. Only used to
            initialize a *fresh* model for training; it is intentionally **not**
            stored in the checkpoint (the trained weights already are), so it can
            be ``None`` when loading via ``from_pretrain``.
        visual_encoder_type: ``resnet`` (default) or the opt-in ``avhubert``.
        visual_repository_root/visual_checkpoint_path: optional external paths;
            environment variables are preferred and paths are not serialized.
    """

    def __init__(
        self,
        # Audio part parameters
        N=256,
        L=40,
        B=256,
        Sc=256,
        H=512,
        P=3,
        X=8,
        R=4,
        # Video part parameters
        E=256,
        V=256,
        K=3,
        D=5,
        # Fusion parameters
        F=256,
        fusion_type="concat",
        fusion_gate_hidden=128,
        fusion_heads=4,
        fusion_dropout=0.0,
        fusion_visual_residual_init=0.05,
        hierarchical_fusion_version="none",
        hierarchical_video_layers=(1, 3, 5),
        hierarchical_residual_init=0.05,
        hierarchical_gate_hidden=128,
        hierarchical_stage_residual_init=0.05,
        # Other parameters
        sample_rate=16000,
        skip_con=False,
        audio_index=2,
        norm="gln",
        causal=False,
        # Video encoder parameters
        video_relu_type="prelu",
        video_pretrain=None,
        visual_encoder_type="resnet",
        visual_feature_mode="frontend",
        visual_context_layer=12,
        visual_adapter_out=512,
        visual_gate_hidden=128,
        visual_context_residual_init=0.05,
        visual_repository_root=None,
        visual_checkpoint_path=None,
    ):
        super().__init__(sample_rate=sample_rate)
        visual_encoder_type = str(visual_encoder_type).lower()
        if visual_encoder_type not in ("resnet", "avhubert"):
            raise ValueError(
                "visual_encoder_type must be 'resnet' or 'avhubert', got "
                f"{visual_encoder_type!r}"
            )
        self.visual_encoder_type = visual_encoder_type

        # Record every architectural init argument so the checkpoint is
        # self-describing and from_pretrain can rebuild the exact model without a
        # config file. video_pretrain is deliberately excluded: the trained video
        # weights live in the state_dict, and the external backbone file may not
        # exist on the machine that loads the checkpoint.
        self._model_args = dict(
            N=N, L=L, B=B, Sc=Sc, H=H, P=P, X=X, R=R,
            E=E, V=V, K=K, D=D, F=F,
            fusion_type=fusion_type,
            fusion_gate_hidden=fusion_gate_hidden,
            fusion_heads=fusion_heads,
            fusion_dropout=fusion_dropout,
            fusion_visual_residual_init=fusion_visual_residual_init,
            hierarchical_fusion_version=hierarchical_fusion_version,
            hierarchical_video_layers=tuple(hierarchical_video_layers),
            hierarchical_residual_init=hierarchical_residual_init,
            hierarchical_gate_hidden=hierarchical_gate_hidden,
            hierarchical_stage_residual_init=hierarchical_stage_residual_init,
            sample_rate=sample_rate,
            skip_con=skip_con,
            audio_index=audio_index,
            norm=norm,
            causal=causal,
            video_relu_type=video_relu_type,
            visual_encoder_type=visual_encoder_type,
            visual_feature_mode=visual_feature_mode,
            visual_context_layer=visual_context_layer,
            visual_adapter_out=visual_adapter_out,
            visual_gate_hidden=visual_gate_hidden,
            visual_context_residual_init=visual_context_residual_init,
        )

        # Video feature extractor (lip frames -> [B, E, Tv]). The official
        # ResNet path is unchanged. AV-HuBERT is opt-in and loads its external
        # assets lazily only when selected.
        if visual_encoder_type == "resnet":
            self.video_model = ResNetVideoModel(
                relu_type=video_relu_type,
                pretrain=video_pretrain,
            )
            for parameter in self.video_model.parameters():
                parameter.requires_grad = False
            self.video_model.eval()
        else:
            self.video_model = AVHubertVideoModel(
                feature_mode=visual_feature_mode,
                output_dim=visual_adapter_out,
                context_layer=visual_context_layer,
                gate_hidden=visual_gate_hidden,
                context_residual_init=visual_context_residual_init,
                repository_root=visual_repository_root,
                checkpoint_path=visual_checkpoint_path,
            )

        # The separator video branch is built with in_channels=E. It must match
        # the selected encoder output width.
        if E != self.video_model.backend_out:
            raise ValueError(
                f"E ({E}) must equal the video encoder output width "
                f"({self.video_model.backend_out})."
            )

        self.av_model = AV_model(
            N=N, L=L, B=B, Sc=Sc, H=H, P=P, X=X, R=R,
            E=E, V=V, K=K, D=D, F=F,
            fusion_type=fusion_type,
            fusion_gate_hidden=fusion_gate_hidden,
            fusion_heads=fusion_heads,
            fusion_dropout=fusion_dropout,
            fusion_visual_residual_init=fusion_visual_residual_init,
            hierarchical_fusion_version=hierarchical_fusion_version,
            hierarchical_video_layers=hierarchical_video_layers,
            hierarchical_residual_init=hierarchical_residual_init,
            hierarchical_gate_hidden=hierarchical_gate_hidden,
            hierarchical_stage_residual_init=hierarchical_stage_residual_init,
            skip_con=skip_con,
            audio_index=audio_index,
            norm=norm,
            causal=causal,
        )

        # Keep machine-specific initialization paths OUT of HuggingFace config.
        # The legacy ResNet weights live in the saved state_dict; AV-HuBERT
        # experiments resolve their licensed external assets from the target
        # environment when reconstructing the architecture.
        cfg = getattr(self, "_hub_mixin_config", None)
        if isinstance(cfg, dict):
            cfg.pop("video_pretrain", None)
            cfg.pop("visual_repository_root", None)
            cfg.pop("visual_checkpoint_path", None)

    def forward(self, x, mouth):
        """
        Args:
            x: [B, T] audio mixture.
            mouth: lip frames, ``[B, Tv, H, W]`` or ``[B, 1, Tv, H, W]``.

        Returns:
            out: [B, T] enhanced speech.
        """
        if mouth.ndim == 4:
            # add channel dim for grayscale: [B, Tv, H, W] -> [B, 1, Tv, H, W]
            mouth = mouth.unsqueeze(1)

        if self.visual_encoder_type == "resnet":
            # The official encoder is fully frozen.
            with torch.no_grad():
                v = self.video_model(mouth.type_as(x))
        else:
            # AV-HuBERT freezes its external backbone internally but keeps the
            # experiment adapters trainable.
            v = self.video_model(mouth.type_as(x))

        return self.av_model(x, v)

    def forward_with_hierarchical_diagnostics(self, x, mouth):
        """Run inference while exposing optional V1/V2 fusion diagnostics."""
        if mouth.ndim == 4:
            mouth = mouth.unsqueeze(1)
        if self.visual_encoder_type == "resnet":
            with torch.no_grad():
                v = self.video_model(mouth.type_as(x))
            visual_diagnostics = None
        else:
            v, visual_diagnostics = self.video_model.forward_with_diagnostics(
                mouth.type_as(x)
            )
        output, diagnostics = (
            self.av_model.forward_with_hierarchical_diagnostics(x, v)
        )
        if visual_diagnostics is not None:
            diagnostics["pretrained_visual"] = visual_diagnostics
        return output, diagnostics

    def train(self, mode=True):
        """Set training mode, while keeping configured frozen modules in eval.

        PyTorch's ``train()`` recurses into every submodule, which would put the
        video encoder's BatchNorm layers back in training mode and let their
        running stats drift. Lightning re-asserts train mode on every ``fit``, so
        this override is the load-bearing guard that keeps the pretrained encoder
        truly frozen (stats fixed, no dropout) throughout training. Extra module
        prefixes can be pinned by train.py for experiments that freeze internal
        separator branches with BatchNorm buffers.
        """
        super().train(mode)
        if self.visual_encoder_type == "resnet":
            self.video_model.eval()
        else:
            # AVHubertVideoModel.train keeps only the small adapters in the
            # requested mode and re-pins the external backbone to eval.
            self.video_model.train(mode)
        for prefix in getattr(self, "_force_eval_module_prefixes", ()):
            module = self
            for part in str(prefix).split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None:
                module.eval()
        return self

    def get_model_args(self):
        """Return the full set of init arguments for serialization."""
        return dict(self._model_args)
