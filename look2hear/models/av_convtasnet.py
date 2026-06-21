"""
Audio-Visual Conv-TasNet Model

The original AV_model (Kai Li, 2021) and all of its building blocks are inlined
here so the model is fully self-contained inside the ``look2hear`` package and
needs no external module imports.

The ``AV_ConvTasNet`` wrapper bundles the frozen ResNet video encoder together
with the audio-visual separator and stores every initialization hyper-parameter,
so ``BaseModel.serialize`` writes a single self-describing checkpoint that holds
*both* the video-encoder and separator weights. Loading then becomes::

    model = AV_ConvTasNet.from_pretrain("best_model.pth")
    enhanced = model(mixture, lip_frames)   # raw lip frames, no separate video net

with no hyper-parameters and no external pretrained file required at the call
site.
"""
import torch
import torch.nn as nn

from ..models.base import BaseModel
from ..videomodels import ResNetVideoModel


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
        if self.skip_con:
            skip_connection = 0
            for i in range(len(self.conv1d_list)):
                skip, out = self.conv1d_list[i](x)
                x = out
                skip_connection += skip
            return skip_connection
        else:
            for i in range(len(self.conv1d_list)):
                out = self.conv1d_list[i](x)
                x = out
            return x


class Concat(nn.Module):
    """
    Audio and Visual Concatenated Part

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
            causal=False):
        super(AV_model, self).__init__()
        self.video = Video_Sequential(E, V, K, skip_con=skip_con, repeat=D)
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
        self.concat = Concat(B, V, F)
        self.feats_conv = Audio_Sequential(
            R - audio_index,
            X,
            in_channels=B,
            out_channels=H,
            b_conv=B,
            sc_conv=Sc,
            kernel_size=P,
            norm=norm,
            causal=causal,
            skip_con=skip_con)
        # mask 1x1 conv
        # n x B x T => n x N x T
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

    def forward(self, x, v):
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
        # lip embeddings: N x T x D => N x V x T
        v = self.video(v)

        # audio/video fusion
        y = self.concat(a, v)

        # n x (B+V) x T
        y = self.feats_conv(y)
        # n x N x T
        m = torch.nn.functional.relu(self.mask(y))
        # n x To
        return self.decoder(w * m)


class AV_ConvTasNet(BaseModel):
    """
    Audio-Visual Conv-TasNet for Speech Enhancement.

    Bundles the frozen ResNet lip-reading video encoder together with the
    :class:`AV_model` separator so a single checkpoint carries *both* sets of
    weights. The forward pass takes raw lip frames and runs the video encoder
    internally, so no separate video model is needed at train or test time.

    All initialization hyper-parameters are recorded and written into the
    checkpoint by ``serialize``, so the model rebuilds with
    ``AV_ConvTasNet.from_pretrain(path)`` alone -- including the video encoder,
    whose weights come from the checkpoint rather than the external
    ``video_pretrain`` file.

    Args:
        N..causal: audio/video/fusion hyper-parameters of :class:`AV_model`.
        video_relu_type: activation of the ResNet video encoder ('relu'/'prelu').
        video_pretrain: path to the ResNet backbone weights. Only used to
            initialize a *fresh* model for training; it is intentionally **not**
            stored in the checkpoint (the trained weights already are), so it can
            be ``None`` when loading via ``from_pretrain``.
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
        # Other parameters
        sample_rate=16000,
        skip_con=False,
        audio_index=2,
        norm="gln",
        causal=False,
        # Video encoder parameters
        video_relu_type="prelu",
        video_pretrain=None,
    ):
        super().__init__(sample_rate=sample_rate)

        # Record every architectural init argument so the checkpoint is
        # self-describing and from_pretrain can rebuild the exact model without a
        # config file. video_pretrain is deliberately excluded: the trained video
        # weights live in the state_dict, and the external backbone file may not
        # exist on the machine that loads the checkpoint.
        self._model_args = dict(
            N=N, L=L, B=B, Sc=Sc, H=H, P=P, X=X, R=R,
            E=E, V=V, K=K, D=D, F=F,
            sample_rate=sample_rate,
            skip_con=skip_con,
            audio_index=audio_index,
            norm=norm,
            causal=causal,
            video_relu_type=video_relu_type,
        )

        # Frozen video feature extractor (lip frames -> [B, E, Tv]).
        self.video_model = ResNetVideoModel(
            relu_type=video_relu_type,
            pretrain=video_pretrain,
        )
        # The ResNet encoder output width is fixed (backend_out, 512); the video
        # branch of the separator is built with in_channels=E. They must match or
        # the first video Conv1d fails with an opaque channel-mismatch error.
        if E != self.video_model.backend_out:
            raise ValueError(
                f"E ({E}) must equal the video encoder output width "
                f"({self.video_model.backend_out})."
            )
        # Always freeze the video encoder: when loading via from_pretrain
        # (video_pretrain=None) ResNetVideoModel.init_from never runs, so freeze
        # here unconditionally to keep it out of the optimizer in every path.
        for p in self.video_model.parameters():
            p.requires_grad = False
        # Keep the encoder in eval mode so its pretrained BatchNorm running stats
        # are used as-is and never updated. requires_grad/no_grad do NOT stop BN
        # buffer updates -- only eval mode does. See train() below, which re-pins
        # this every time the parent module is switched back to train mode.
        self.video_model.eval()

        self.av_model = AV_model(
            N=N, L=L, B=B, Sc=Sc, H=H, P=P, X=X, R=R,
            E=E, V=V, K=K, D=D, F=F,
            skip_con=skip_con,
            audio_index=audio_index,
            norm=norm,
            causal=causal,
        )

        # Keep ``video_pretrain`` OUT of the HuggingFace config. PyTorchModelHubMixin
        # captures every __init__ argument into config.json; video_pretrain is just
        # a path to the backbone used to *initialise* a fresh model, and the trained
        # video weights already live in the state_dict. Persisting a local training
        # path would make from_pretrained() try to load a file that does not exist
        # on the downloader's machine. Drop it so the rebuilt model uses the default
        # (video_pretrain=None) and takes its video weights from the checkpoint.
        cfg = getattr(self, "_hub_mixin_config", None)
        if isinstance(cfg, dict):
            cfg.pop("video_pretrain", None)

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

        # Video encoder is frozen; never accumulate grads through it.
        with torch.no_grad():
            v = self.video_model(mouth.type_as(x))

        return self.av_model(x, v)

    def train(self, mode=True):
        """Set training mode, but keep the frozen video encoder in eval.

        PyTorch's ``train()`` recurses into every submodule, which would put the
        video encoder's BatchNorm layers back in training mode and let their
        running stats drift. Lightning re-asserts train mode on every ``fit``, so
        this override is the load-bearing guard that keeps the pretrained encoder
        truly frozen (stats fixed, no dropout) throughout training.
        """
        super().train(mode)
        self.video_model.eval()
        return self

    def get_model_args(self):
        """Return the full set of init arguments for serialization."""
        return dict(self._model_args)
