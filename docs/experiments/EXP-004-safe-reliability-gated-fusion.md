# EXP-004: Safe reliability-gated visual fusion

## Summary

- **Experiment ID:** EXP-004
- **Name:** Safe reliability-gated visual fusion for Track 2 AV-ConvTasNet
- **Date:** 2026-07-09; design review revised 2026-07-10
- **Owner:** TODO
- **Status:** Planned
- **Purpose:** Test whether a conservatively initialized gate can reduce harmful use of degraded visual evidence without changing the audio representation selected by the gate.
- **Parent experiment:** `EXP-002` implementation commit `b5f8e79aa4818310229afc8d1f054a19b8d8b92c`
- **Source baseline:** `baseline/track2-source-v1` (`a3cce6c5c362ae0ce79d59494c52b36995a44c49`)
- **Branch:** `exp/safe-reliability-gate`
- **Initial implementation commit:** `16773f98ee7fa20a7884b348970428604d7c1b96`
- **Formal-run commit SHA:** TODO; record the exact commit used for training

## Background and motivation

The Track 2 training pipeline applies one of five visual corruptions: full mask,
occlusion, low resolution/noise/blur, frame freeze, or AV desynchronization.
The baseline temporally resamples the video features, concatenates audio and
video, and applies one 1x1 convolution. It has no explicit mechanism for
reducing visual influence when the visual stream is unreliable.

EXP-002 introduced a channel-wise gate, but a regression has been reported by
the user. The exact checkpoint, seed, dataset snapshot, and metric delta have
not yet been entered in the EXP-002 record, so this is motivating evidence, not
a verified experimental result.

The initial EXP-004 implementation interpolated between two candidates:

```text
(1 - gate) * audio_safe + gate * av_candidate
```

That form has an interpretation problem. `av_candidate` contains a second,
independent audio projection, so changing the gate changes both the audio
mapping and the visual contribution. The gate therefore cannot be interpreted
strictly as visual reliability. The reviewed implementation removes that
confounder.

## Reviewed method

After temporal alignment, the revised fusion is:

```text
a_f = Conv1d_audio(audio)              # includes the single fusion bias
v_f = Conv1d_video(video)              # no bias
g   = sigmoid(GateNet[a_f, v_f, |a_f - v_f|])
y   = a_f + g * v_f
```

By default, `g` has shape `[batch, 1, time]`, so all channels share one visual
reliability value at each time step. This scalar temporal gate is deliberately
more constrained and easier to inspect than EXP-002's `[batch, channels, time]`
gate.

The final GateNet convolution starts with zero weights and bias `-2.0`:

```text
initial g = sigmoid(-2) ~= 0.1192
```

Thus the first forward pass preserves the full audio projection and attenuates
the visual residual to about 11.9%. This is a conservative initialization, not
a guarantee that the model will outperform the baseline.

### Baseline compatibility

The baseline fusion can be written as:

```text
Conv1d(concat[audio, video]) = W_a * audio + W_v * video + bias
```

EXP-004 places the bias only in `Conv1d_audio` and uses no bias in
`Conv1d_video`. Therefore, when `g=1`, its projection is exactly equivalent to
the baseline fusion. When `--warm_start` loads a baseline checkpoint,
`train.py` splits the baseline kernel into `W_a` and `W_v`; only the new gate
network remains newly initialized. This makes a warm-start comparison easier
to interpret.

## Falsifiable hypothesis

Under the same data snapshot, seed, optimization settings, and evaluation
pipeline as the reproduced baseline:

- EXP-004 should improve or preserve the primary Track 2 validation metric
  relative to EXP-002.
- It should avoid a material regression relative to direct concatenation on
  severely corrupted visual samples.
- Its learned gate should not remain constant at the initialization value for
  all samples and times. A constant gate would indicate that the reliability
  estimator has not learned a useful conditional policy.

The exact acceptance threshold is TODO until the team confirms the official
primary metric and records a reproducible baseline result. No metric claim is
made by this document.

## Controlled differences

Changed from baseline/EXP-002:

- fusion only: audio projection plus gated visual residual;
- scalar temporal gate by default;
- zero-weight, negative-bias gate initialization;
- optional baseline-concat weight migration for `--warm_start`;
- diagnostic `forward_with_gate()` for inspecting gate values.
- a new `Track2-AVConvTasNet-SafeReliabilityGate-V2` output directory, avoiding
  accidental auto-resume from an incompatible initial EXP-004 checkpoint.

Intentionally unchanged:

- audio encoder, temporal separator, mask estimator, and decoder;
- frozen ResNet video encoder and video temporal stack;
- loss, optimizer, scheduler, and training duration;
- Track 2 manifests and visual degradation pipeline;
- baseline and previous experiment configuration files.

## Modified files

| File | Change | Reason |
|---|---|---|
| `look2hear/models/av_convtasnet.py` | Revise `SafeReliabilityGatedFusion` | Make the gate control only visual residuals |
| `train.py` | Split baseline concat weights during safe-gate warm-start | Preserve trained baseline fusion projections |
| `scripts/check_safe_reliability_gate.py` | Add synthetic boundary and gradient checks | Verify shape, initialization, fallback, and equivalence |
| `configs/track2_av_convtasnet_safe_reliability_gate.yml` | Dedicated EXP-004 configuration | Keep baseline and EXP-002 configs unchanged |
| `EXPERIMENTS.md` | Track planned experiment | Maintain provenance |
| `docs/experiments/EXP-004-safe-reliability-gated-fusion.md` | Reviewed protocol | Separate verified facts, hypotheses, and TODOs |

## Configuration

- **Configuration file:** `configs/track2_av_convtasnet_safe_reliability_gate.yml`
- **Configuration Git blob SHA-1:** `31dac5dd54ca76e09612afb15b7ac1c84a60160f`
- **Random seed:** TODO; the current config does not declare one
- **Batch size:** 8
- **Learning rate/scheduler:** Adam lr 0.001; ReduceLROnPlateau, factor 0.5, patience 10
- **Epochs/stopping:** maximum 500; early stopping on `val_loss`, patience 20
- **Gate settings:** hidden width 128; initial bias -2.0; scalar temporal gate
- **Visual degradation probability:** 1.0 for train, validation, and test loaders

`degrade_prob: 1.0` means every sample receives a corruption. This matches the
current Track 2 baseline config and keeps the architecture comparison
controlled, but it gives the gate no clean-versus-corrupt clip-level contrast.
Some corruption types affect only a frame interval, so temporal contrast may
still exist. A later mixed-clean experiment must use a new experiment ID rather
than silently changing EXP-004.

## Data

- **Dataset/source:** VoxCeleb2-derived training corpus; exact release TODO
- **Version/date:** TODO
- **License/status:** TODO
- **Data path:** TODO; do not record a personal machine path in Git
- **Manifest/checksum:** TODO
- **Train split:** `DataPreProcess/vox2/tr`, pending verification
- **Validation split:** `DataPreProcess/vox2/cv`, pending verification
- **Test split:** `DataPreProcess/vox2/tt`, pending verification
- **Speaker-disjoint policy:** TODO
- **Degradation policy:** one online Track 2 visual corruption per degraded sample; exact sampled distribution should be captured at run time

The dataset currently returns mixture, target, degraded mouth frames, and a
filename. It does not return the degradation type or mask to the model. The gate
is therefore trained only through the enhancement loss, without direct
reliability supervision.

## Environment

- **Host/job ID:** TODO
- **Operating system:** TODO
- **GPU model/count and VRAM:** TODO
- **NVIDIA driver/CUDA:** TODO
- **Python/PyTorch/Lightning:** TODO
- **CPU/RAM/storage:** TODO
- **Environment export path:** TODO

## Commands

### Preflight

```bash
git status --short --branch
git rev-parse HEAD
python tools/check_track2_setup.py --check-env
```

### Synthetic code check

```bash
python scripts/check_safe_reliability_gate.py
python -m compileall look2hear/models/av_convtasnet.py train.py
```

### Formal run from scratch

```bash
python train.py --conf_dir configs/track2_av_convtasnet_safe_reliability_gate.yml
```

### Optional controlled warm-start

```bash
python train.py \
  --conf_dir configs/track2_av_convtasnet_safe_reliability_gate.yml \
  --warm_start <verified-baseline-checkpoint>
```

The formal protocol must choose either from-scratch or warm-start before the
run and apply the same initialization policy to the comparison experiment. Do
not compare a warm-started EXP-004 against a from-scratch baseline as if fusion
were the only changed variable.

Do not resume an initial EXP-004 `last.ckpt`: the reviewed projection layout is
intentionally different. The V2 experiment name isolates new outputs. An old
checkpoint may be retained as an artifact, but it is not a valid resume point
for this implementation.

### Evaluation

```bash
# TODO: record the exact validation/test command after baseline reproduction
# and evaluation assets are confirmed.
```

## Required diagnostics

In addition to the challenge metrics, record the following gate statistics on
a fixed validation subset:

- overall mean, standard deviation, minimum, and maximum;
- per-utterance mean distribution;
- temporal gate plots for at least one example of each degradation type;
- performance with normal video, zeroed video, and temporally shifted video;
- NaN/Inf checks for model output, loss, gradients, and gate values.

`forward_with_gate(audio_features, video_features)` exposes the gate for a
focused diagnostic script. The normal model forward path remains unchanged.

## Risks and limitations

- A low initial gate is only an optimization prior; it does not prove visual
  reliability is learned.
- Zero initialization of the final gate weights makes the first gate constant.
  The final layer learns immediately, while the preceding GateNet layer starts
  receiving a gate-dependent gradient after the final weights move away from
  zero.
- Scalar gating may be too restrictive if reliability differs by feature
  channel, but it is a cleaner first test than channel-wise gating.
- The absolute feature difference is only a learned mismatch cue. Large values
  do not automatically mean that video is unreliable.
- With `degrade_prob: 1.0` and no degradation labels, the enhancement loss may
  learn a nearly constant visual prior instead of sample-dependent reliability.
- Warm-start migration preserves baseline fusion projections, but the initial
  gate still attenuates their visual contribution; the initial full-model output
  is therefore intentionally not identical to the baseline.

## Runtime, artifacts, and results

- **Start/end time and wall-clock:** TODO
- **GPU-hours and peak memory:** TODO
- **Log path:** TODO
- **Checkpoint path/checksum:** TODO
- **Saved configuration path:** TODO
- **Prediction and metric paths:** TODO
- **Interrupted/restarted:** TODO

| Scope | Metric | Baseline | EXP-002 | EXP-004 | Delta vs baseline | Verified |
|---|---|---:|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO | No |

No formal EXP-004 result is available. Keep status `Planned` until the command,
artifacts, and metrics have all been verified.

## Failures, conclusion, and next step

- **Failures/anomalies:** TODO
- **Conclusion:** TODO; distinguish measured evidence from interpretation
- **Worth retaining:** TODO / Inconclusive
- **Candidate for merge:** No while unverified
- **Next step:** run the synthetic check, confirm data/environment/seed and a reproducible baseline, then execute one controlled EXP-004 training protocol
