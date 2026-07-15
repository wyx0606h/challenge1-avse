# EXP-006: Baseline fine-tune with frozen internal visual branch BN

## Summary

- **Experiment ID:** EXP-006
- **Name:** Baseline AV-ConvTasNet on Chinese-LiPS Track2-dev-matched v1 with frozen `av_model.video`
- **Date:** 2026-07-14
- **Status:** Completed
- **Purpose:** Fix the EXP-005 waveform gain explosions caused by adapting the separator's internal visual branch BatchNorm statistics to the ChineseLips visual domain.
- **Parent:** EXP-005 and the official Track 2 baseline checkpoint.
- **Architecture:** official baseline `AV_ConvTasNet` concat fusion; no ReliabilityGate/CrossAttention.

## Motivation

EXP-005 showed that the dev-matched ChineseLips audio-domain correction alone was not sufficient. The trained checkpoint was stable on its own ChineseLips-domain validation/test data, but official Track 2 dev inference with `align_face=True` produced severe gain explosions:

| output set | peak p50 | peak p90 | peak max | peak > 1 | peak > 100 |
|---|---:|---:|---:|---:|---:|
| official baseline full-dev enhanced | 0.101 | 0.210 | 0.650 | 0 | 0 |
| EXP-005 | 0.154 | 601.528 | 2032.970 | 2122 | 1578 |

Targeted checks isolated the issue to the separator's internal visual branch:

- replacing official-dev mouth input with zero/mean mouth removed explosions;
- `align_face=False` removed sampled explosions, but baseline and earlier fine-tunes were stable with `align_face=True`;
- restoring only official warm-start BN buffers under `av_model.video.conv1d_list.*.bn.*` removed sampled explosions.

Hypothesis: keep `av_model.video` parameters and BatchNorm buffers fixed from the official warm-start checkpoint, and fine-tune the remaining audio/fusion/separator layers on the dev-matched ChineseLips data.

## Code Changes

Added config-driven training controls:

- `training.freeze_param_prefixes`: sets matching parameters `requires_grad=False`;
- `training.force_eval_module_prefixes`: keeps matching modules in `eval()` when Lightning switches the parent model to train mode, preventing BatchNorm running-stat updates.

For EXP-006:

```yaml
training:
  freeze_param_prefixes:
    - av_model.video.
  force_eval_module_prefixes:
    - av_model.video
```

## Configuration

Configuration file:

```text
configs/track2_av_convtasnet_chineselips_devmatched_v1_degrade0_freeze_avvideo_2gpu.yml
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
- DDP visible GPUs: physical GPU `5,6`

## Smoke Checks

Before formal training:

1. config parse and Python compile passed;
2. model instantiation and warm-start from `official_serialized.pth` passed;
3. `av_model.video.*` freeze check passed: 38 parameter tensors frozen,
   `0` trainable under that prefix;
4. `av_model.video` remained in eval mode after `model.train()`;
5. sampled official-dev exploding cases with warm-start + frozen branch returned
   normal output scale:

| sample | peak | RMS |
|---|---:|---:|
| `track2/dev/mix/000242/s2` | 0.0941 | 0.0134 |
| `track2/dev/mix/000317/s2` | 0.2296 | 0.0284 |
| `track2/dev/remix/000008_000480/s1` | 0.1080 | 0.0190 |

## Launch

tmux session:

```text
baseline_devmatched_v1_d0_freeze_avvideo
```

Command:

```bash
cd /home/avse/workspace-goodparts
mkdir -p Experiments/Track2-AVConvTasNet-Baseline-ChineseLipsDevMatchedV1-Degrade0-FreezeAVVideoBN-2GPU/logs

CUDA_VISIBLE_DEVICES=5,6 \
/home/avse/avse_gpu_venv/bin/python train.py \
  --conf_dir configs/track2_av_convtasnet_chineselips_devmatched_v1_degrade0_freeze_avvideo_2gpu.yml \
  --warm_start /home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth \
  2>&1 | tee Experiments/Track2-AVConvTasNet-Baseline-ChineseLipsDevMatchedV1-Degrade0-FreezeAVVideoBN-2GPU/logs/train.tmux.log
```

## Runtime Notes

- 2026-07-14: prepared code/config after EXP-005 diagnosis.
- 2026-07-14 00:13:23 CST: formal training launched in tmux session
  `baseline_devmatched_v1_d0_freeze_avvideo`.
- 2026-07-14 00:13 CST: DDP startup succeeded on physical GPUs 5 and 6.
  Lightning reported 13.0M trainable parameters and 12.0M non-trainable
  parameters. Sanity check passed and training entered epoch 0.
- 2026-07-14 00:16 CST: training was still active at epoch 0. No checkpoint had
  been written yet, as validation had not run.
- 2026-07-14 00:18 CST: epoch 0 finished. Validation reached
  `val_loss=1.64966`, `train_loss_epoch=-3.65`, and Lightning saved
  `epoch=0.ckpt` plus `last.ckpt`. Training continued into epoch 1.
- 2026-07-14 00:54 CST: training was active in epoch 7. Validation improved
  across epochs 3-6, with current best `val_loss=1.11258` at `epoch=6.ckpt`.
  Top-k checkpoints retained at this point were `epoch=1.ckpt`,
  `epoch=3.ckpt`, `epoch=4.ckpt`, `epoch=5.ckpt`, and `epoch=6.ckpt`.
- 2026-07-14 06:19 CST: training finished normally after reaching
  `max_epochs=80`. Final and best checkpoint was `epoch=79.ckpt` with
  `val_loss=-4.26468`; `best_model.pth` was exported in the experiment
  directory. Next required step is an official-dev amplitude smoke check before
  any full evaluation.
- 2026-07-14 10:00 CST: official-dev amplitude smoke check passed on the three
  EXP-005 diagnostic samples with `align_face=True`:
  `track2/dev/mix/000242/s2` peak `0.0875`, RMS `0.0131`;
  `track2/dev/mix/000317/s2` peak `0.2299`, RMS `0.0290`;
  `track2/dev/remix/000008_000480/s1` peak `0.1021`, RMS `0.0186`.
  This clears the immediate EXP-005 waveform explosion failure mode.
- 2026-07-14 10:06 CST: full official-dev enhancement completed with 5250/5250
  wavs under
  `/home/avse/experiments/track2_exp006_freeze_avvideo_full_dev/enhanced`.
  Full amplitude validation passed: peak p50 `0.1043`, p90 `0.2130`,
  p99 `0.3024`, max `0.5158`, `peak > 1` count `0`.
- 2026-07-14 12:36 CST: full official-dev metrics pipeline completed:
  objective, UTMOS, DNSMOS, speaker similarity, and ASR. Pipeline log:
  `/home/avse/experiments/track2_exp006_freeze_avvideo_full_dev/logs/full_metrics_pipeline.log`.

## Full Dev Evaluation

Evaluation root:

```text
/home/avse/experiments/track2_exp006_freeze_avvideo_full_dev
```

Enhanced wavs:

```text
/home/avse/experiments/track2_exp006_freeze_avvideo_full_dev/enhanced
```

Metrics:

```text
/home/avse/experiments/track2_exp006_freeze_avvideo_full_dev/metrics
```

Baseline reference:

```text
/home/avse/experiments/track2_dev_official_baseline/metrics
```

### Objective Metrics

Only `remix` has clean reference and supports SI-SDR/PESQ/STOI.

| model | scope | n | SI-SDR | PESQ | STOI |
|---|---|---:|---:|---:|---:|
| official baseline | track2/remix | 2196 | -2.8480 | 1.2561 | 0.4698 |
| EXP-006 | track2/remix | 2196 | -0.5850 | 1.4002 | 0.5785 |

### No-Reference and Task Metrics

| metric | scope | baseline | EXP-006 | delta |
|---|---|---:|---:|---:|
| UTMOS | overall | 1.1758 | 1.2443 | +0.0685 |
| UTMOS | track2/mix | 1.1842 | 1.2152 | +0.0310 |
| UTMOS | track2/remix | 1.1640 | 1.2848 | +0.1208 |
| DNSMOS p808 | overall | 2.3703 | 2.4873 | +0.1170 |
| DNSMOS ovr | overall | 1.3955 | 1.4150 | +0.0195 |
| DNSMOS ovr | track2/mix | 1.4486 | 1.4111 | -0.0375 |
| DNSMOS ovr | track2/remix | 1.3218 | 1.4204 | +0.0986 |
| spk_sim | overall | 0.3702 | 0.4673 | +0.0971 |
| spk_sim | track2/mix | 0.3842 | 0.4762 | +0.0920 |
| spk_sim | track2/remix | 0.3506 | 0.4550 | +0.1044 |
| CER | overall | 0.8812 | 0.8725 | -0.0087 |
| CER | track2/mix | 0.8922 | 0.8495 | -0.0427 |
| CER | track2/remix | 0.8658 | 0.9045 | +0.0387 |

Lower CER is better; higher values are better for the other metrics listed in
this table.

## Interpretation

EXP-006 fixes the EXP-005 official-dev waveform explosion failure mode by
freezing the internal visual branch and its BatchNorm state. Under that
constraint, the Track2-dev-matched ChineseLips v1 dataset gives a consistent
positive signal on the official dev set:

- remix objective metrics improve clearly over the official baseline;
- UTMOS improves on overall, mix, and remix;
- speaker similarity improves on overall, mix, and remix;
- DNSMOS improves overall and on remix, while mix DNSMOS ovr drops slightly;
- ASR CER improves overall and on mix, but remix CER is worse than baseline.

The result supports the current working hypothesis that the original
ChineseLips conversion had a meaningful audio-domain mismatch with Track2 dev,
and that matching the ChineseLips mixture RMS/SNR distribution to the official
dev profile partly reduces that mismatch. The improvement is not complete:
the remaining mix DNSMOS drop, remix CER regression, 2 s versus 6 s duration
mismatch, and visual-domain mismatch still need follow-up experiments.
