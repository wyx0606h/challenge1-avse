# Baseline Smoke Overfit

## Summary

- **Date:** 2026-06-27
- **Branch:** `exp/baseline-reproduction`
- **Purpose:** Verify that the official Track 2 model class, official loss, and
  smoke Track 2 Dataset/DataLoader can overfit a fixed tiny batch with visual
  degradation disabled.
- **Conventions:** See `docs/experiment-conventions.md`.
- **Status:** Completed

## Configuration

- **Repository:** `/home/avse/workspace-goodparts`
- **Data:** `/home/avse/data_pipeline/outputs/chinese_lips_baseline_smoke`
- **Environment:** `/home/avse/data_pipeline/.venv`
- **Output directory:** `/home/avse/experiments/baseline_smoke_overfit`
- **Config:** `configs/track2_smoke_overfit.yml`
- **Script:** `scripts/overfit_track2_smoke.py`
- **Dataset class:** `Track2StaticDataset`
- **DataLoader:** `batch_size=1`, `num_workers=0`, `shuffle=False`
- **Visual degradation:** `degrade_prob=0.0`
- **Seed:** `20260627`
- **Steps:** 120
- **Device:** CPU (`torch==2.4.1+cpu`)
- **Model:** official `AV_ConvTasNet` architecture, randomly initialized
- **Loss:** `PITLossWrapper(pairwise_neg_snr)`

## Result

- **Training loop:** completed
- **Backward / optimizer step:** completed every step
- **Finite checks:** loss, outputs, and gradients remained finite
- **Initial loss:** `3.0709826946258545`
- **Final loss:** `-8.647812843322754`
- **Initial MSE to target:** `0.07693256437778473`
- **Final MSE to target:** `0.005178818479180336`
- **Runtime:** `127.48` seconds
- **First fixed item:** `tr_mix_000001_s1_id00002_070_22_M_ZX_100.wav`

Artifacts are stored under
`/home/avse/experiments/baseline_smoke_overfit`; no generated WAV, checkpoint,
or raw log was written inside Git.

## Before Merge To Main

- Review whether the fixed-batch runner should remain as a tracked utility.
- Keep `main` stable and confirm all artifacts remain external to Git.
