# Track 2 Chinese-LiPS Fine-Tune LR Sweep, degrade_prob=0

## Summary

- **Date:** 2026-07-06
- **Branch:** `exp/baseline-reproduction`
- **Purpose:** debug the failed 500-step fine-tune by adding a local checkpoint
  roundtrip check and a small dev objective gate before any full dev run.
- **Status:** complete. The checkpoint serialization path is valid. Low-LR
  20-step fine-tunes pass the small dev gate; 50-step and 100-step runs fail.

This is still a stability sanity, not a final training result.

## Inputs And Environment

- **Repository:** `/home/avse/workspace-goodparts`
- **Training data:** `/home/avse/processed_Chineselips`
- **Official Track 2 weights:**
  `/home/avse/avse-assets/models/Real-World-AVSE-Baseline-Track2`
- **Dev eval data root:**
  `/home/avse/experiments/track2_dev_official_baseline/data_root`
- **Experiment root:**
  `/home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep`
- **Virtualenv:** `/home/avse/avse_gpu_venv`
- **GPU:** GPU 5, RTX 4090

New helper scripts:

- `scripts/serialize_track2_pretrained.py`
- `scripts/track2_dev_objective_gate.py`
- `scripts/run_track2_lr_sweep.py`

## Gate Definition

The gate uses Track 2 dev `remix` only, with `--limit 50`. In this repository
that means 50 remix directories, expanded to 100 target-speaker outputs.

Gate checks:

- every expected wav is produced;
- output is finite and non-zero;
- `peak_max <= 5.0`;
- objective metrics are available;
- SI-SDR is at least `-4.848`, i.e. no more than 2 dB below the official
  full-dev remix baseline `-2.848`;
- STOI is at least `0.430`.

## 0-Step Roundtrip

The official Hub-format checkpoint was serialized to local `.pth` and then
loaded through the same `eval_real.py from_pretrain` path used by fine-tuned
checkpoints.

Commands:

```bash
/home/avse/avse_gpu_venv/bin/python scripts/serialize_track2_pretrained.py \
  --pretrained-dir /home/avse/avse-assets/models/Real-World-AVSE-Baseline-Track2 \
  --output /home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth \
  --overwrite

/home/avse/avse_gpu_venv/bin/python scripts/track2_dev_objective_gate.py \
  --ckpt /home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth \
  --data-root /home/avse/experiments/track2_dev_official_baseline/data_root \
  --output-dir /home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/gate_limit50 \
  --gpus 5 --limit 50 --overwrite
```

Roundtrip gate result:

| scope | n | SI-SDR | PESQ | STOI | peak_max | result |
|---|---:|---:|---:|---:|---:|---|
| dev remix gate | 100 | -2.0497 | 1.2767 | 0.5062 | 0.243879 | pass |

Conclusion: the `.pth` serialization/load path is not the cause of the
500-step failure.

## LR Sweep

Command:

```bash
/home/avse/avse_gpu_venv/bin/python scripts/run_track2_lr_sweep.py \
  --dataset-root /home/avse/processed_Chineselips \
  --pretrained-dir /home/avse/avse-assets/models/Real-World-AVSE-Baseline-Track2 \
  --data-root /home/avse/experiments/track2_dev_official_baseline/data_root \
  --output-dir /home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/sweep_9runs \
  --gpus 5 --overwrite
```

Training settings common to all runs:

- official `AV_ConvTasNet` initialized from the official Track 2 baseline;
- `degrade_prob=0.0`;
- `batch_size=4`;
- `num_workers=2`;
- `segment=2.0`;
- `normalize_audio=false`;
- `val_batches=10`;
- seed `20260706`.

Results:

| run | pass | SI-SDR | PESQ | STOI | peak_max | last train loss | best val loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lr_1em06_steps_20` | yes | -1.5509 | 1.3000 | 0.5107 | 0.225945 | -0.611423 | 3.662077 |
| `lr_1em06_steps_50` | no | -6.7550 | 1.2470 | 0.3961 | 0.300938 | 0.887752 | 3.827975 |
| `lr_1em06_steps_100` | no | -38.5294 | 1.2250 | 0.2018 | 17728.138672 | -2.040175 | 3.609643 |
| `lr_3em07_steps_20` | yes | -1.5696 | 1.3013 | 0.5098 | 0.226091 | -0.608139 | 3.677935 |
| `lr_3em07_steps_50` | no | -6.7944 | 1.2442 | 0.3950 | 0.301420 | 0.908216 | 3.850521 |
| `lr_3em07_steps_100` | no | -38.5531 | 1.1981 | 0.2007 | 17667.279297 | -2.028597 | 3.663707 |
| `lr_1em07_steps_20` | yes | -1.5770 | 1.3052 | 0.5095 | 0.226178 | -0.606076 | 3.683342 |
| `lr_1em07_steps_50` | no | -6.8089 | 1.2549 | 0.3945 | 0.301810 | 0.911524 | 3.856429 |
| `lr_1em07_steps_100` | no | -38.5656 | 1.1895 | 0.2003 | 17685.535156 | -2.024531 | 3.680361 |

Best gate-passing run by SI-SDR:

- `lr_1em06_steps_20`
- checkpoint:
  `/home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/sweep_9runs/runs/lr_1em06_steps_20/train/best_model.pth`

Artifacts:

- `roundtrip/official_serialized.pth`
- `roundtrip/gate_limit50/gate_summary.json`
- `sweep_9runs/sweep_summary.json`
- per-run `train/`, `gate_limit50/`, logs, metrics, and generated gate wavs

## Interpretation

The failed 500-step run was not caused by checkpoint serialization. It was
caused by fine-tuning instability.

The low-LR sweep shows a consistent pattern:

- 20 steps is safe for all tested learning rates and does not damage the small
  dev gate;
- 50 steps already fails objective thresholds even though peak remains normal;
- 100 steps triggers the same severe output-scale failure seen in the original
  500-step sanity.

Training loss alone is not a reliable acceptance signal here. The 100-step runs
can show lower training loss while dev audio is unusable. Future experiments
must keep the dev objective/audio gate in the loop.

## Next Steps

- Do not run full dev for the failed 50/100-step checkpoints.
- If a full-dev sanity is needed, use `lr_1em06_steps_20` as the first
  candidate and run full dev objective only, before DNSMOS/UTMOS/CER.
- Investigate why continuing past roughly 20 steps destabilizes dev output:
  compare train/cv/dev waveform scales, try freezing model parts, and consider
  a smaller optimizer scope.
- Keep `degrade_prob=0.0` until the stability issue is understood; only then
  move to Track 2-style `degrade_prob=1.0`.
