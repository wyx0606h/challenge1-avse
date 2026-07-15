# EXP-005: Baseline fine-tune on dev-matched Chinese-LiPS v1

## Summary

- **Experiment ID:** EXP-005
- **Name:** Baseline AV-ConvTasNet on Chinese-LiPS Track2-dev-matched v1
- **Date:** 2026-07-13
- **Status:** Evaluation running
- **Purpose:** Test whether correcting the Chinese-LiPS audio-domain mismatch
  reduces the negative transfer previously observed when fine-tuning on
  `/home/avse/processed_Chineselips`.
- **Primary controlled variable:** dataset root only.
- **Architecture:** official baseline `AV_ConvTasNet` with original concat
  audio-video fusion; no ReliabilityGate, no SafeReliabilityGate, no
  CrossAttention.

## Motivation

Previous Chinese-LiPS fine-tuning degraded official Track 2 dev performance.
The new dataset
`/home/avse/processed_Chineselips_track2dev_match_v1` keeps the same Track 2
training layout but remixes Chinese-LiPS with official-dev-like RMS and SNR.

Dataset build report:

```text
/home/avse/data_pipeline/docs/dataset_reports/chinese_lips_track2dev_match_v1_build.md
```

Key dataset signal:

| dataset | mix RMS p50 | mix peak p50 |
|---|---:|---:|
| original Chinese-LiPS sample | 0.12048 | 0.67345 |
| dev-matched Chinese-LiPS v1 | 0.02768 | 0.16016 |
| official Track 2 dev profile | 0.02765 | 0.18549 |

## Configuration

Configuration file:

```text
configs/track2_av_convtasnet_chineselips_devmatched_v1_degrade0_2gpu.yml
```

Important settings:

- warm-start checkpoint:
  `/home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth`
- training data:
  `/home/avse/processed_Chineselips_track2dev_match_v1/tr`
- validation data:
  `/home/avse/processed_Chineselips_track2dev_match_v1/cv`
- test data:
  `/home/avse/processed_Chineselips_track2dev_match_v1/tt`
- `degrade_prob: 0.0`
- optimizer: Adam, lr `1e-5`
- epochs: `80`
- early stopping: `val_loss`, patience `8`
- per-GPU batch size: `4`
- DDP visible GPUs: physical GPU `5,6`

This first run intentionally uses `degrade_prob=0.0` to compare against the
earlier degrade0 Chinese-LiPS fine-tune with the fewest additional variables.
A later run can repeat the same config with `degrade_prob=1.0`.

## Launch

tmux session:

```text
baseline_devmatched_v1_d0
```

Command:

```bash
cd /home/avse/workspace-goodparts
mkdir -p Experiments/Track2-AVConvTasNet-Baseline-ChineseLipsDevMatchedV1-Degrade0-2GPU/logs

CUDA_VISIBLE_DEVICES=5,6 \
/home/avse/avse_gpu_venv/bin/python train.py \
  --conf_dir configs/track2_av_convtasnet_chineselips_devmatched_v1_degrade0_2gpu.yml \
  --warm_start /home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth \
  2>&1 | tee Experiments/Track2-AVConvTasNet-Baseline-ChineseLipsDevMatchedV1-Degrade0-2GPU/logs/train.tmux.log
```

## Runtime Notes

- 2026-07-13: GPU inventory before launch showed GPUs 5 and 6 essentially idle.
- 2026-07-13: Config added and launch prepared from `/home/avse/workspace-goodparts`.
- 2026-07-13: Formal training launched in tmux session
  `baseline_devmatched_v1_d0`.
- 2026-07-13: Startup checks passed in the training log: dataset counts matched
  the new v1 build, warm-start loaded from `official_serialized.pth`, DDP
  initialized with two ranks, and Lightning sanity checking completed.
- 2026-07-13: Training entered epoch 0 and reached at least batch `128/1889`.
  GPUs 5 and 6 were both active at about 5.7 GiB memory usage.
- 2026-07-13 20:44 CST: Training was still running in tmux at epoch `53`.
  The best visible validation score at that point was `val_loss=-6.12` from
  epoch `52`; top checkpoints retained `epoch=46.ckpt`, `epoch=47.ckpt`,
  `epoch=50.ckpt`, `epoch=51.ckpt`, and `epoch=52.ckpt`.
- 2026-07-13 20:46 CST: Training advanced to epoch `54`. The epoch 53
  validation shown on screen was `val_loss=-5.92`, so it did not replace the
  current top checkpoint. `last.ckpt` was updated and GPUs 5/6 remained active.
- 2026-07-13 21:03 CST: Training was still running at epoch `57`. Epoch `54`
  improved validation to `val_loss=-6.33485`, and epoch `56` improved again to
  the current best `val_loss=-6.51544`. The retained top checkpoints were
  `epoch=46.ckpt`, `epoch=50.ckpt`, `epoch=52.ckpt`, `epoch=54.ckpt`, and
  `epoch=56.ckpt`; `last.ckpt` was updated at 21:01 CST.
- 2026-07-13 21:44 CST: Training finished by early stopping after epoch `64`.
  The best score remained epoch `56` with `val_loss=-6.51544`. Later retained
  top checkpoints were `epoch=54.ckpt`, `epoch=56.ckpt`, `epoch=60.ckpt`,
  `epoch=61.ckpt`, and `epoch=63.ckpt`; `epoch=64` had `val_loss=-6.31` and
  did not enter top 5.
- 2026-07-13 21:44 CST: `best_model.pth` was saved successfully and verified
  readable with PyTorch. It contains a serialized model dict with keys
  `model_args`, `model_name`, and `state_dict`; the state dict has 647 tensors.

Final artifacts:

```text
/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-Baseline-ChineseLipsDevMatchedV1-Degrade0-2GPU/best_model.pth
/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-Baseline-ChineseLipsDevMatchedV1-Degrade0-2GPU/epoch=56.ckpt
/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-Baseline-ChineseLipsDevMatchedV1-Degrade0-2GPU/last.ckpt
```

Current log:

```text
/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-Baseline-ChineseLipsDevMatchedV1-Degrade0-2GPU/logs/train.tmux.log
```

## Planned Evaluation

After training completes or a clearly best checkpoint is available:

1. evaluate on the same held-out devsplit 10% protocol used for the
   dev-domain diagnostic;
2. compare against official baseline on the same 10%;
3. compare against the previous original-ChineseLips fine-tuned checkpoint if
   outputs are available;
4. inspect `remix` objective metrics separately from `mix` no-reference/CER
   metrics.

Interpretation rule: this run should answer whether the dataset-domain fix is
useful before adding SafeReliabilityGate or other architecture changes.

## Full Dev Evaluation

Started after training completion on 2026-07-13.

Evaluation root:

```text
/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev
```

Official dev source:

```text
/home/avse/data/track2_dev/dev
```

Tool-compatible wrapper:

```text
/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev/data_root/track2/dev -> /home/avse/data/track2_dev/dev
```

Preflight item counts:

| scene | target-speaker eval items |
|---|---:|
| `mix` | 3054 |
| `remix` | 2196 |
| **total** | 5250 |

Checkpoint:

```text
/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-Baseline-ChineseLipsDevMatchedV1-Degrade0-2GPU/best_model.pth
```

Launch command:

```bash
cd /home/avse/workspace-goodparts

EXP=/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev \
REPO=/home/avse/workspace-goodparts \
PY38=/home/avse/avse_gpu_venv/bin/python \
PY311=/home/avse/avse_eval_py311/bin/python \
DATA_ROOT=/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev/data_root \
CKPT=/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-Baseline-ChineseLipsDevMatchedV1-Degrade0-2GPU/best_model.pth \
SAVE_DIR=/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev/full/enhanced \
METRICS_DIR=/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev/metrics \
LOG_DIR=/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev/logs \
CONFIG_DIR=/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev/config \
GPU_ENHANCE=5 \
GPU_METRICS=6 \
SPK_SHARDS=8 \
bash scripts/run_track2_finetuned_full_dev_metrics.sh
```

Expected outputs:

```text
/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev/full/enhanced
/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev/metrics
/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev/logs/run_all_metrics.log
```

Runtime notes:

- 2026-07-13 22:42 CST: first full-dev attempt was stopped during enhancement.
  The GPU inference environment used numpy `1.24.4`, while official dev
  landmark pickle files reference the numpy-2 namespace
  `numpy._core.multiarray`, causing skipped items with
  `No module named 'numpy._core'`.
- 2026-07-13 22:47 CST: added a compatibility loader in
  `look2hear/datas/real_test_dataset.py` so numpy-1 environments can read
  these landmark pickle files without disabling `align_face`.
- 2026-07-13 22:47 CST: removed the stale unsupported
  `--funasr_vad_model` argument from
  `scripts/run_track2_finetuned_full_dev_metrics.sh`; FunASR VAD remains
  configured inside `look2hear/metrics/real_metrics.py`.
- 2026-07-13 22:47 CST: the incomplete first-attempt outputs were archived
  under
  `/home/avse/experiments/track2_chineselips_devmatched_v1_degrade0_full_dev/failed_attempts/`.
- 2026-07-13 22:47 CST: smoke enhancement with `--limit 1` passed on GPU 5
  with `align_face=True` and produced 4 wavs.
- 2026-07-13 22:50 CST: a second restart was stopped after detecting that GPU
  binding should be set before Python process startup. The wrapper now prefixes
  metric/enhancement commands with `CUDA_VISIBLE_DEVICES=$GPU_ENHANCE` or
  `CUDA_VISIBLE_DEVICES=$GPU_METRICS`. Partial outputs from that restart were
  also archived under `failed_attempts/`.
- 2026-07-13 22:50 CST: full-dev evaluation restarted in tmux session
  `exp005_full_dev_eval_v3`. Enhancement uses physical GPU 5 and metric stages
  will use physical GPU 6.
- 2026-07-13 23:00 CST: full-dev enhancement completed:
  5250/5250 wavs produced, `missing=0`, `nonfinite=0`, `all_zero=0`.
  Validation observed 128 files with a small length delta down to `-16`
  samples; the wrapper treats this as non-fatal and continued to metrics.
- 2026-07-13 23:00 CST: objective metric evaluation started in 4 shards.
- 2026-07-14 00:07 CST: full-dev metric run was manually stopped during
  DNSMOS after objective finished, because output amplitude inspection showed
  severe waveform gain explosions in the EXP-005 enhanced wavs.

Partial objective result before stopping:

| scope | n | SI-SDR | PESQ | STOI |
|---|---:|---:|---:|---:|
| `track2/remix` | 2196 | -25.0543 | 1.2163 | 0.2956 |

This objective result should **not** be used as a clean model-quality
comparison. The enhanced wav validation found `peak_max=2032.97`, while the
official baseline and previous original-ChineseLips fine-tune full-dev outputs
both had `peak_max≈0.65`.

Amplitude diagnosis:

| model/output set | peak p50 | peak p90 | peak max | peak > 1 | peak > 100 |
|---|---:|---:|---:|---:|---:|
| official baseline full-dev enhanced | 0.101 | 0.210 | 0.650 | 0 | 0 |
| previous original-ChineseLips fine-tune | 0.094 | 0.214 | 0.656 | 0 | 0 |
| EXP-005 dev-matched ChineseLips | 0.154 | 601.528 | 2032.970 | 2122 | 1578 |

Targeted checks:

- `best_model.pth` and `epoch=56.ckpt` gave identical outputs on a known
  exploding sample, so the serialized checkpoint is not the source of the
  issue.
- The same EXP-005 checkpoint was stable on its own
  `/home/avse/processed_Chineselips_track2dev_match_v1` validation/test domain:
  first 300 `cv` items had max peak `0.381`, and first 300 full-length `tt`
  items had max peak `0.389`.
- For official dev exploding samples, replacing the real mouth tensor with
  zeros or a constant mean tensor removed the explosion. Example:
  `track2/dev/mix/000242/s2` changed from peak `603.72` with real mouth to
  about `0.05` with zero/mean mouth.
- `align_face=False` also removed the explosion on sampled official-dev cases,
  while `align_face=True` reproduced it. Baseline and the previous fine-tune
  were stable with `align_face=True`, so this is not merely an eval-protocol
  mismatch.
- Restoring only the official warm-start BN buffers under
  `av_model.video.conv1d_list.*.bn.{running_mean,running_var,num_batches_tracked}`
  removed the explosion on sampled official-dev cases. Example:
  `track2/dev/mix/000242/s2` peak dropped from `603.72` to `0.127`.

Current interpretation:

EXP-005 did not globally corrupt the model, but fine-tuning on the dev-matched
ChineseLips visual domain updated the separator's internal visual branch
(`av_model.video`) BatchNorm statistics/parameters enough that official dev
landmark-aligned visual inputs became out-of-distribution for that branch. The
downstream mask is `relu(self.mask(y))`, so the mask is unbounded; when the
visual branch BN is mismatched, the fused features can produce extremely large
masks and waveform gain explosions.

Recommended next experiments:

1. rerun a small official-dev evaluation after restoring/freezing
   `av_model.video.conv1d_list` BN running statistics from the official
   warm-start checkpoint;
2. repeat training with visual branch BN frozen, or freeze all
   `av_model.video.*` parameters/buffers and fine-tune only safer audio/fusion
   layers;
3. add an evaluation safety check that aborts metrics when enhanced wav peak
   distribution exceeds a fixed threshold, so invalid metrics are not recorded
   as model-quality results;
4. consider adding bounded mask/output normalization only as a controlled
   ablation, because it may hide but not solve the BN/domain mismatch.
