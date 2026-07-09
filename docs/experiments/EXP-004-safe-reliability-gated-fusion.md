# EXP-004: Safe reliability-gated visual fusion

## Summary

- **Experiment ID:** EXP-004
- **Name:** Safe reliability-gated visual fusion for Track 2 AV-ConvTasNet
- **Date:** 2026-07-09
- **Owner:** TODO
- **Status:** Planned
- **Purpose:** Improve the first reliability gate after observed regression by starting from a conservative audio-safe path and opening the visual path only when useful.
- **Parent baseline/experiment:** `EXP-002` / `exp/experiment-cross` (`b5f8e79aa4818310229afc8d1f054a19b8d8b92c`), ultimately based on `baseline/track2-source-v1`
- **Branch:** `exp/safe-reliability-gate`
- **Commit SHA:** TODO

## Background

The first reliability gate directly used `audio_projection + gate * visual_projection`. That design made the visual path active from the beginning of training and constrained fusion to an additive visual residual. If Track 2 visual degradation is severe, this can inject unreliable visual evidence before the gate has learned a meaningful reliability estimate.

This experiment keeps the visual-reliability hypothesis but changes the fusion form to be more conservative and closer to a safe fallback.

## Hypothesis

A safe interpolation gate can reduce the regression observed in the first gate by preserving an audio-safe route at initialization:

```text
gate = 0 -> audio_safe
gate = 1 -> av_candidate
fused = audio_safe + gate * (av_candidate - audio_safe)
```

With a negative gate bias, training begins near audio-only behavior and learns to use visual information only where it improves the enhancement objective.

## Difference from baseline / EXP-002

Compared with EXP-002, this experiment adds `SafeReliabilityGatedFusion` and uses `fusion_type: safe_reliability_gate`.

Key changes:

- separate `audio_safe` candidate from an `audio+visual` candidate;
- scalar temporal gate by default (`[B, 1, T]`) instead of full channel-wise gate;
- gate final layer initialized with zero weights and bias `-2.0`, so initial visual usage is about `sigmoid(-2) ~= 0.12`;
- no formal run or metric result is claimed in this protocol.

## Modified files

| File | Change | Reason |
|---|---|---|
| `look2hear/models/av_convtasnet.py` | Add `SafeReliabilityGatedFusion` and factory option | Conservative visual reliability fusion |
| `configs/track2_av_convtasnet_safe_reliability_gate.yml` | Dedicated Track 2 config | Keep previous configs unchanged |
| `EXPERIMENTS.md` | Add planned experiment row | Track provenance |
| `docs/experiments/EXP-004-safe-reliability-gated-fusion.md` | Planned protocol | Record hypothesis and TODOs |

## Configuration

- **Configuration file:** `configs/track2_av_convtasnet_safe_reliability_gate.yml`
- **Configuration snapshot/hash:** TODO
- **Random seed:** TODO
- **Batch size:** 8
- **Learning rate/scheduler:** Adam lr 0.001, ReduceLROnPlateau
- **Epochs or stopping rule:** 500 epochs with early stopping patience 20
- **Other changed parameters:** `fusion_type: safe_reliability_gate`, `fusion_gate_hidden: 128`, `fusion_gate_init_bias: -2.0`, `fusion_gate_scalar: true`

## Data

- **Dataset/source:** TODO / pending confirmation
- **Version/date:** TODO
- **License/status:** TODO
- **Data path:** TODO
- **Manifest/checksum:** TODO
- **Train split:** `DataPreProcess/vox2/tr` TODO verify
- **Validation split:** `DataPreProcess/vox2/cv` TODO verify
- **Test split:** `DataPreProcess/vox2/tt` TODO verify
- **Speaker-disjoint policy:** TODO
- **Augmentation/degradation policy:** Track 2 online visual degradation, TODO verify exact run settings

## Commands

### Sanity check

```bash
python -m compileall look2hear/models/av_convtasnet.py
# TODO: instantiate model after environment and training assets are confirmed
```

### Formal run

```bash
python train.py --conf_dir configs/track2_av_convtasnet_safe_reliability_gate.yml
```

## Results

No result yet. Do not mark completed until the formal run, artifacts, and metrics are verified.
