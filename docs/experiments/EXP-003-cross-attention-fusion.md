# EXP-003: Cross-attention visual fusion

## Summary

- **Experiment ID:** EXP-003
- **Name:** Cross-attention visual fusion for Track 2 AV-ConvTasNet
- **Date:** 2026-07-07
- **Owner:** TODO
- **Status:** Planned
- **Purpose:** Test whether explicit audio-query-to-visual-token interaction improves fusion over direct concatenation.
- **Parent baseline/experiment:** `baseline/track2-source-v1` (`a3cce6c5c362ae0ce79d59494c52b36995a44c49`)
- **Branch:** `exp/experiment-cross`
- **Commit SHA:** TODO

## Background

Prior AVSS/AVTSE work suggests that simple addition or concatenation does not fully model the relationship between audio and lip motion. Track 2 adds a second challenge: visual information can be temporally corrupted or misaligned, so the separator benefits from a module that can compare audio tokens with visual evidence rather than only stacking channels.

## Hypothesis

A compact cross-attention fusion block can learn better audio-visual correspondences than direct concatenation. Expected gains should appear most clearly when visual timing remains informative but direct channel stacking is insufficient.

## Difference from baseline

Replace direct audio-video concatenation with `CrossAttentionFusion` in `look2hear/models/av_convtasnet.py`, selected by `fusion_type: "cross_attention"`. The audio encoder, frozen video encoder, separator blocks, loss, optimizer, and data pipeline remain unchanged.

## Modified files

| File | Change | Reason |
|---|---|---|
| `look2hear/models/av_convtasnet.py` | Add cross-attention fusion module and config switch | Model audio-visual token interaction explicitly |
| `configs/track2_av_convtasnet_cross_attention.yml` | Dedicated Track 2 config | Keep baseline config untouched |
| `EXPERIMENTS.md` | Add planned experiment row | Track provenance |
| `docs/experiments/EXP-003-cross-attention-fusion.md` | Planned protocol | Record hypothesis and TODOs |

## Configuration

- **Configuration file:** `configs/track2_av_convtasnet_cross_attention.yml`
- **Configuration snapshot/hash:** TODO
- **Random seed:** TODO
- **Batch size:** 8
- **Learning rate/scheduler:** Adam lr 0.001, ReduceLROnPlateau
- **Epochs or stopping rule:** 500 epochs with early stopping patience 20
- **Other changed parameters:** `fusion_type: cross_attention`, `fusion_heads: 4`, `fusion_dropout: 0.0`

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
python train.py --conf_dir configs/track2_av_convtasnet_cross_attention.yml
```

## Results

No result yet. Do not mark completed until the formal run, artifacts, and metrics are verified.
