# EXP-003: Native-rate temporal cross-attention fusion

## Summary

- **Experiment ID:** EXP-003
- **Name:** Native-rate temporal cross-attention for Track 2 AV-ConvTasNet
- **Date:** 2026-07-07; design review revised 2026-07-10
- **Owner:** TODO
- **Status:** Planned
- **Purpose:** Test explicit audio-to-visual interaction without duplicating low-rate visual tokens or discarding the AV alignment prior.
- **Source baseline:** `baseline/track2-source-v1` (`a3cce6c5c362ae0ce79d59494c52b36995a44c49`)
- **Initial prototype commit:** `b5f8e79aa4818310229afc8d1f054a19b8d8b92c`
- **Branch:** `exp/cross-attention-v2`
- **Formal-run commit SHA:** TODO; record the exact training commit

## Background and review finding

The baseline upsamples the video features to the audio feature length,
concatenates both streams, and applies a 1x1 convolution. This is inexpensive
but leaves all cross-modal interaction to the following separator blocks.

The initial EXP-003 prototype also upsampled video to the audio feature length
before applying attention. For a two-second segment with `L=40` and stride 20,
the audio encoder produces roughly 1600 tokens while 25 fps video provides
roughly 50 tokens. Upsampling first changes the attention matrix from about
`1600 x 50` to `1600 x 1600`. It repeats visual information and increases the
attention matrix by roughly 32 times without creating new visual evidence.

The prototype also had no explicit temporal-position prior. Attention could
therefore match an audio token to any repeated visual token based only on
content. Finally, LayerNorm and a new output projection transformed the whole
audio path immediately, making warm-start behavior harder to interpret.

## Reviewed module architecture

The revised module keeps visual features at their native temporal rate:

```text
audio [B, A, Ta] --1x1 conv--> audio_base [B, Ta, F]
video [B, V, Tv] --1x1 conv--> video_token [B, Tv, F]

query = LayerNorm(audio_base)
key = value = LayerNorm(video_token)

context = MultiHeadAttention(
    query,
    key,
    value,
    relative_time_bias,
    local_window=12,
)

fused = audio_base + learnable_residual_scale * context
```

Audio is the query because the separator output must retain the dense audio
timeline. Video is key/value because lip motion supplies auxiliary evidence.
The output length is therefore always `Ta`, while attention complexity is
`O(Ta * Tv)` rather than `O(Ta^2)`.

### Temporal prior

Each audio position is mapped linearly onto the native video timeline. The
attention logits receive the following shared bias:

```text
bias(i, j) = -0.1 * distance(mapped_audio_i, video_j)
```

Positions farther than 12 video frames are masked. Track 2 samples AV desync in
the range +/-10 video frames (about +/-400 ms at 25 fps), so a 12-frame window
contains the configured maximum shift plus a small margin. The soft distance
bias still prefers nearby positions without forcing exact synchronization.

The mask is an inductive prior, not a sparse attention kernel: PyTorch still
computes a dense `Ta x Tv` matrix. The efficiency gain comes from preserving
the short native video sequence.

### Stable initialization

The Q, K, V, and attention output projections start as trainable identity
matrices. The visual context residual starts at `0.1`:

```text
initial fused = audio_base + 0.1 * context
```

This prevents randomly initialized attention from dominating the audio path.
The residual scale is global and learned; it is not a sample-dependent visual
reliability gate and should not be interpreted as one.

### Baseline warm-start

The baseline fusion kernel can be split into audio and visual input slices:

```text
W_concat = [W_audio, W_video]
```

During `--warm_start`, `train.py` copies these slices into the EXP-003 audio and
video 1x1 projections. The baseline bias is assigned only to the audio
projection. Attention, normalization, and residual-scale parameters are new and
retain the reviewed initialization. This does not make the initial full output
identical to baseline, but it avoids throwing away both trained modality
projections.

## Falsifiable hypothesis

Using the same data snapshot, baseline checkpoint, seed, optimization settings,
and evaluation pipeline:

- EXP-003 should improve or preserve the primary Track 2 metric relative to
  direct concatenation.
- It should use substantially less attention memory than the original
  upsample-then-attend prototype.
- Attention should remain finite, respect the temporal window, and show
  non-uniform AV correspondences on at least some validation samples.

The baseline metrics have already been measured by the team. Their exact
artifact path and values must be linked during the server run before computing
the EXP-003 delta. No result is claimed here.

## Controlled differences

Changed:

- direct concatenation is replaced by audio-query-to-video cross-attention;
- video remains at native temporal resolution;
- relative-time bias and a +/-12-frame video window are applied;
- attention projections use identity initialization;
- visual context enters through a learnable residual initialized to 0.1;
- baseline concat weights are split during warm-start;
- `forward_with_attention()` exposes per-head weights for diagnostics.

Unchanged:

- audio encoder, temporal separator, mask estimator, and decoder;
- frozen ResNet video encoder and video temporal stack;
- loss, optimizer, scheduler, data pipeline, and degradation probability;
- baseline and EXP-002 configuration files.

## Modified files

| File | Change | Reason |
|---|---|---|
| `look2hear/models/av_convtasnet.py` | Review `CrossAttentionFusion` and its arguments | Native-rate, time-aware, stable attention |
| `train.py` | Add baseline concat-to-attention projection migration | Controlled warm-start |
| `scripts/check_cross_attention_fusion.py` | Add synthetic architecture checks | Verify shape, window, gradients, and residual fallback |
| `configs/track2_av_convtasnet_cross_attention.yml` | Add reviewed EXP-003 settings and V2 output name | Reproducible isolated run |
| `EXPERIMENTS.md` | Point EXP-003 to its dedicated branch | Correct provenance |
| `docs/experiments/EXP-003-cross-attention-fusion.md` | Replace prototype narrative with reviewed protocol | Traceable claims and risks |

## Configuration

- **Configuration file:** `configs/track2_av_convtasnet_cross_attention.yml`
- **Configuration Git blob SHA-1:** `79f34b2f812f5ab54a66f35b2d25283b51248725`
- **Random seed:** TODO; the current config does not declare one
- **Batch size:** 8
- **Learning rate/scheduler:** Adam lr 0.001; ReduceLROnPlateau, factor 0.5, patience 10
- **Epochs/stopping:** maximum 500; early stopping on `val_loss`, patience 20
- **Attention:** 4 heads, dropout 0.0, window 12, time-bias strength 0.1
- **Residual initialization:** 0.1
- **Experiment name:** `Track2-AVConvTasNet-CrossAttention-V2`

The V2 experiment name prevents automatic resume from an incompatible
prototype checkpoint.

## Data and environment

- **Dataset/source and version:** TODO / pending server confirmation
- **License/status:** TODO
- **Manifest/checksum:** TODO
- **Train/validation/test splits:** configured as `tr/cv/tt`, pending verification
- **Speaker-disjoint policy:** TODO
- **Degradation:** Track 2 online visual degradation with `degrade_prob: 1.0`
- **Host, GPU, CUDA, Python, PyTorch, Lightning:** TODO

The dataset does not pass degradation type or shift size to the model. The
attention module learns only from enhancement loss and the fixed temporal prior.

## Commands

### Preflight and synthetic check

```bash
git status --short --branch
python tools/check_track2_setup.py --check-env
python -m compileall look2hear/models/av_convtasnet.py train.py
python scripts/check_cross_attention_fusion.py
```

### Formal warm-start run

```bash
python train.py \
  --conf_dir configs/track2_av_convtasnet_cross_attention.yml \
  --warm_start <verified-baseline-checkpoint>
```

Use a server-specific copied configuration for absolute data paths and actual
GPU IDs. Do not commit server paths into the tracked configuration.

## Required smoke tests and diagnostics

Before formal training:

- load one real batch and verify audio/video shapes and finite values;
- verify baseline projection migration and inspect missing/unexpected keys;
- run one forward, training loss, backward, and optimizer step;
- measure peak memory with the intended batch size;
- confirm attention shape is `[B, heads, Ta, Tv]`, not `[B, heads, Ta, Ta]`.

During evaluation, record attention entropy, temporal offset from the nominal
alignment, residual-scale value, and examples for normal, masked, frozen, and
desynchronized video. These diagnostics support interpretation but do not
replace challenge metrics.

## Risks and limitations

- A fixed +/-12-frame window is based on the current degradation generator. If
  the challenge data contains larger unknown offsets, useful evidence may be
  masked.
- Identity attention initialization is a training prior, not a performance
  guarantee.
- A global residual scale cannot suppress visual context for individual bad
  samples; that belongs to a separate reliability-gating experiment.
- Attention may still learn nearly uniform weights, especially when all
  training samples are degraded.
- Cross-attention adds parameters and compute over concat even after removing
  the unnecessary quadratic audio-length attention.
- Warm-start reuses modality projections but cannot initialize the new
  attention behavior from baseline.

## Runtime, artifacts, and results

- **Formal-run commit:** TODO
- **Server configuration/hash:** TODO
- **Baseline checkpoint/hash:** TODO
- **Start/end time, GPU-hours, peak memory:** TODO
- **Logs/checkpoints/predictions/metrics:** TODO

| Scope | Metric | Existing baseline | EXP-003 | Delta | Verified |
|---|---|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | No |

No formal EXP-003 result is available. Keep status `Planned` until the formal
run begins; use `Completed` only after artifacts and metrics are verified.

## Conclusion and next step

- **Conclusion:** TODO
- **Worth retaining:** Inconclusive
- **Candidate for merge:** No while unverified
- **Next step:** complete server smoke checks, then run one controlled warm-start experiment against the existing baseline metrics
