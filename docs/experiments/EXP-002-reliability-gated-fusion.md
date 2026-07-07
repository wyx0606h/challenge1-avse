# EXP-002: Reliability-gated visual fusion

## Summary

- **Experiment ID:** EXP-002
- **Name:** Reliability-gated visual fusion for Track 2 AV-ConvTasNet
- **Date:** 2026-07-07
- **Owner:** TODO
- **Status:** Planned
- **Purpose:** Test whether a lightweight learned gate can reduce over-reliance on degraded visual features while preserving useful lip guidance.
- **Parent baseline/experiment:** `baseline/track2-source-v1` (`a3cce6c5c362ae0ce79d59494c52b36995a44c49`)
- **Branch:** `exp/experiment-cross`
- **Commit SHA:** TODO

## Background

Track 2 deliberately corrupts the target visual stream with occlusion, low resolution, frame freeze, dropped frames, and AV desynchronization. The baseline fusion module temporally aligns video features and directly concatenates them with audio features. This assumes that visual features are uniformly useful after alignment, which is risky when the visual stream is unreliable.

## Hypothesis

A per-time, per-channel visual reliability gate can learn to inject visual evidence only when it is useful. The expected behavior is smaller degradation under severe visual corruption and no major regression under milder corruption.

## Difference from baseline

Replace direct audio-video concatenation with `ReliabilityGatedFusion` in `look2hear/models/av_convtasnet.py`, selected by `fusion_type: "reliability_gate"`. The audio encoder, frozen video encoder, separator blocks, loss, optimizer, and data pipeline remain unchanged.

## Modified files

| File | Change | Reason |
|---|---|---|
| `look2hear/models/av_convtasnet.py` | Add reliability-gated fusion module and config switch | Learn visual trust under Track 2 degradation |
| `configs/track2_av_convtasnet_reliability_gate.yml` | Dedicated Track 2 config | Keep baseline config untouched |
| `EXPERIMENTS.md` | Add planned experiment row | Track provenance |
| `docs/experiments/EXP-002-reliability-gated-fusion.md` | Planned protocol | Record hypothesis and TODOs |

## Configuration

- **Configuration file:** `configs/track2_av_convtasnet_reliability_gate.yml`
- **Configuration snapshot/hash:** TODO
- **Random seed:** TODO
- **Batch size:** 8
- **Learning rate/scheduler:** Adam lr 0.001, ReduceLROnPlateau
- **Epochs or stopping rule:** 500 epochs with early stopping patience 20
- **Other changed parameters:** `fusion_type: reliability_gate`, `fusion_gate_hidden: 128`

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
# TODO: instantiate model after environment and video pretrain path are confirmed
```

### Formal run

```bash
python train.py --conf_dir configs/track2_av_convtasnet_reliability_gate.yml
```

## Results

No result yet. Do not mark completed until the formal run, artifacts, and metrics are verified.
