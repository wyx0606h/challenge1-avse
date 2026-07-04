# Chinese-LiPS Baseline Smoke Eval

## Summary

- **Date:** 2026-07-05
- **Branch:** `exp/baseline-reproduction`
- **Purpose:** Run the official Track 2 baseline checkpoint on the already
  generated Chinese-LiPS smoke dataset to get an initial metric sense.
- **Status:** Complete.
- **Experiment output:**
  `/home/avse/experiments/chinese_lips_baseline_smoke_eval`

This is not official dev evaluation. The input dataset is the baseline training
Dataset format `tr/cv/tt/{mix.json,s1.json,s2.json}`, so the run reads each
split with `Track2StaticDataset` and compares `mix` and `enhanced` against the
clean target wavs from `s1.json` / `s2.json`.

## Inputs

- **Dataset:** `/home/avse/data_pipeline/outputs/chinese_lips_baseline_smoke`
- **Official Track 2 weights:**
  `/home/avse/avse-assets/models/Real-World-AVSE-Baseline-Track2`
- **Model repository:** `/home/avse/workspace-goodparts`
- **Data pipeline:** `/home/avse/data_pipeline`
- **Evaluation script:**
  `/home/avse/experiments/chinese_lips_baseline_smoke_eval/scripts/eval_chinese_lips_smoke_static.py`

The smoke dataset contains clean references, so reference metrics are available:
SI-SDR, PESQ, STOI, and MSE. CER and speaker similarity were not run because
this small training-format dataset does not include the full official dev
evaluation setup, ASR interpretation context, or enrollment protocol.

## Environment

- **Virtualenv:** `/home/avse/avse_gpu_venv`
- **Python:** `3.8.10`
- **PyTorch:** `2.4.1+cu121`
- **CUDA device:** `CUDA_VISIBLE_DEVICES=7`
- **Runtime device:** `cuda`
- **Metric packages:** `pesq`, `pystoi`, `soundfile`
- **degrade_prob:** `0.0`

`degrade_prob=0` was used intentionally. This run is a data/model sanity
evaluation over clean Chinese-LiPS videos; Track 2 training should later use
`degrade_prob=1.0` to match the official robust-training setting.

## Command

```bash
CUDA_VISIBLE_DEVICES=7 /home/avse/avse_gpu_venv/bin/python \
  /home/avse/experiments/chinese_lips_baseline_smoke_eval/scripts/eval_chinese_lips_smoke_static.py \
  --workspace-root /home/avse/workspace-goodparts \
  --dataset-root /home/avse/data_pipeline/outputs/chinese_lips_baseline_smoke \
  --ckpt /home/avse/avse-assets/models/Real-World-AVSE-Baseline-Track2 \
  --output-root /home/avse/experiments/chinese_lips_baseline_smoke_eval \
  --splits tr cv tt \
  --batch-size 1 \
  --num-workers 0 \
  --degrade-prob 0.0 \
  --device cuda \
  --seed 20260705 \
  --save-audio
```

## Outputs

- `metrics/per_item_metrics.csv`: 64 target items
- `metrics/summary.json`
- `config.json`
- `audio/{tr,cv,tt}/*_{mix,target,enhanced}.wav`: 192 wav files

## Metrics

Mean values:

| scope | n | mix SI-SDR | enhanced SI-SDR | delta SI-SDR | mix PESQ | enhanced PESQ | delta PESQ | mix STOI | enhanced STOI | delta STOI | mix MSE | enhanced MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 64 | -0.019 | -1.771 | -1.752 | 1.197 | 1.403 | +0.206 | 0.709 | 0.632 | -0.077 | 0.01019 | 0.00710 |
| tr | 32 | -0.078 | -3.438 | -3.360 | 1.256 | 1.298 | +0.043 | 0.698 | 0.602 | -0.096 | 0.01355 | 0.01068 |
| cv | 16 | 0.017 | 2.048 | +2.032 | 1.147 | 1.545 | +0.398 | 0.713 | 0.691 | -0.021 | 0.00741 | 0.00240 |
| tt | 16 | 0.064 | -2.256 | -2.319 | 1.131 | 1.471 | +0.339 | 0.728 | 0.634 | -0.094 | 0.00627 | 0.00463 |

Median deltas:

| scope | delta SI-SDR | delta PESQ | delta STOI |
|---|---:|---:|---:|
| overall | +2.706 | +0.098 | +0.031 |
| tr | -1.385 | +0.026 | +0.004 |
| cv | +11.077 | +0.418 | +0.113 |
| tt | +9.950 | +0.415 | +0.122 |

Improved item counts:

| metric | improved / total |
|---|---:|
| SI-SDR | 35 / 64 |
| PESQ | 40 / 64 |
| STOI | 37 / 64 |

## Interpretation

The dataset has clean targets, so these metrics are meaningful as a first
sanity check. The official Track 2 checkpoint was not trained on this
Chinese-LiPS smoke distribution, and the dataset is very small, so the numbers
should not be treated as a stable benchmark.

The run shows mixed behavior:

- PESQ improves on average and for most items.
- MSE improves overall, indicating the output often moves closer to the clean
  target in waveform distance.
- Median SI-SDR improves, especially on `cv` and `tt`, but mean SI-SDR is worse
  because some items degrade strongly.
- STOI mean decreases even though the median delta is positive, again pointing
  to outlier sensitivity and domain mismatch.

This is enough to confirm the eval path and reference metric calculation over
the smoke dataset. For a training benchmark, use a larger Chinese-LiPS split,
train or fine-tune on that distribution, then evaluate with the same script and
also compare against official dev via `eval_real.py`.
