# Track 2 Dev Official Baseline

## Summary

- **Date:** 2026-07-02 to 2026-07-04
- **Branch:** `exp/baseline-reproduction`
- **Purpose:** Reproduce official Track 2 baseline inference and dev
  evaluation from local `config.json` + `model.safetensors`.
- **Conventions:** See `docs/experiment-conventions.md`.
- **Status:** Inference and all currently supported dev metrics are complete:
  objective, DNSMOS, speaker similarity, UTMOS, and CER/ASR.

## Inputs

- **Repository:** `/home/avse/workspace-goodparts`
- **Dev data:** `/home/avse/data/track2_dev`
- **Evaluation assets:** `/home/avse/avse-assets/evaluation`
- **Official Track 2 weights:**
  `/home/avse/avse-assets/models/Real-World-AVSE-Baseline-Track2`
- **Experiment output:** `/home/avse/experiments/track2_dev_official_baseline`
- **Compatibility data root:**
  `/home/avse/experiments/track2_dev_official_baseline/data_root`

The released dev directory is flat (`mix/`, `remix`, manifests). A symlink
tree was used so `RealTestDataset` could read its expected
`track2/dev/{mix,remix}` layout without copying or modifying original data.

## Environment

Enhancement and the first metric pass used the reusable GPU environment:

- **Virtualenv:** `/home/avse/avse_gpu_venv`
- **Python:** `3.8.10`
- **PyTorch:** `2.4.1+cu121`
- **Torchaudio:** `2.4.1+cu121`
- **NumPy:** `1.24.4`
- **Transformers:** `4.46.3`
- **FunASR:** `1.3.1`
- **ModelScope:** `1.20.1`

UTMOS and CER required a Python 3.11 follow-up environment because UTMOSv2
requires Python `>=3.9` and the bundled Qwen3 ASR config requires newer
Transformers support:

- **Virtualenv:** `/home/avse/avse_eval_py311`
- **Python:** `3.11.15`
- **PyTorch:** `2.7.1+cu126`
- **Torchaudio:** `2.7.1+cu126`
- **Torchvision:** `0.22.1+cu126`
- **Transformers:** `5.13.0`
- **FunASR:** `1.3.14`
- **ModelScope:** `1.38.0`
- **UTMOSv2:** `1.3.1.dev0`
- **GPU used for follow-up metrics:** GPU 6, RTX 4090

Notes:

- Non-sandbox `nvidia-smi` reported 8 x RTX 4090, driver `550.144.03`,
  CUDA `12.4`.
- `eval_real.py` uses its own `--gpus` argument. Passing only an outer
  `CUDA_VISIBLE_DEVICES=6` is insufficient because the script defaults
  `--gpus 0` and overwrites the environment internally.
- UTMOSv2 source install from GitHub timed out once, so the package was
  installed from the existing local audit clone `/tmp/utmosv2-audit`.
  The model weight was cached at
  `/home/avse/.cache/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth`;
  the same weight also exists under
  `/home/avse/avse-assets/evaluation/utmosv2/models/fusion_stage3/`.

## Weight Check

Official model files:

| file | SHA256 |
|---|---|
| `config.json` | `de1819318f7c3fb79914314475ab4b223973af9104d42600e533cdb557bcf5f8` |
| `model.safetensors` | `600d1632346e9b85e64f2842000f41d2cdb9f6b1afca8011c5decdc5ce9e78c1` |

The local `AV_ConvTasNet.from_pretrained()` load was checked strictly:
`model_keys=647`, `checkpoint_keys=647`, `missing=0`, `unexpected=0`,
`shape_mismatch=0`, parameters `25,014,849`.

## Commands

Full enhancement:

```bash
CUDA_VISIBLE_DEVICES=3 /home/avse/avse_gpu_venv/bin/python eval_real.py \
  --conf_dir configs/track2_av_convtasnet.yml \
  --ckpt /home/avse/avse-assets/models/Real-World-AVSE-Baseline-Track2 \
  --data_root /home/avse/experiments/track2_dev_official_baseline/data_root \
  --track track2 --scene both --split dev --metrics none --mode enhance \
  --save_dir /home/avse/experiments/track2_dev_official_baseline/full/enhanced \
  --gpus 0
```

Eval-only metrics over the saved WAVs:

```bash
# Reference metrics on remix, 4 shards, then --merge_shards
/home/avse/avse_gpu_venv/bin/python eval_real.py ... \
  --metrics objective --mode eval --num_shards 4

# DNSMOS, 16 CPU shards, then --merge_shards
DNSMOS_DIR=/home/avse/avse-assets/evaluation/dnsmos \
  /home/avse/avse_gpu_venv/bin/python eval_real.py ... \
  --metrics dnsmos --mode eval --num_shards 16

# Speaker similarity, enrollment cache plus 8 CPU shards, then --merge_shards
/home/avse/avse_gpu_venv/bin/python eval_real.py ... \
  --metrics spk --mode eval --num_shards 8 \
  --wespeaker_ckpt /home/avse/avse-assets/evaluation/wespeaker/cnceleb-resnet34-LM/model_5.pt \
  --enroll_ckpt /home/avse/experiments/track2_dev_official_baseline/metrics/enroll_dev.pt

# UTMOS + CER/ASR in Python 3.11, sequential, tmux-safe.
/home/avse/experiments/track2_dev_official_baseline/scripts/run_missing_metrics_py311.sh
```

The follow-up script first checks whether `utmos_summary.csv` and
`asr_summary.csv` already contain `overall,5250`; complete metrics are skipped
to avoid repeated high-cost work.

## Inference Result

- **Expected items:** `5250`
- **Produced WAVs:** `5250`
- **mix / remix:** `3054 / 2196`
- **Sample rate:** `16000`
- **Non-finite outputs:** `0`
- **All-zero outputs:** `0`
- **Length match:** `5122`
- **Length mismatch:** `128`, each output shorter by `1` to `16` samples
- **Max peak:** `0.6504323483`
- **Minimum RMS:** `0.0007900272`
- **Log:** `/home/avse/experiments/track2_dev_official_baseline/logs/full_inference.log`

The dataset reported missing landmark pickles for some videos and used the
whole-face fallback path. No original dev file was changed.

## Metrics

| scope | n | SI-SDR | PESQ | STOI | UTMOS | DNSMOS p808 | DNSMOS sig | DNSMOS bak | DNSMOS ovr | CER | spk_sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **overall** | 5250 | -2.848 | 1.256 | 0.470 | 1.176 | 2.370 | 1.745 | 1.704 | 1.396 | 0.881 | 0.370 |
| track2 / mix | 3054 | -- | -- | -- | 1.184 | 2.402 | 1.819 | 1.776 | 1.449 | 0.892 | 0.384 |
| track2 / remix | 2196 | -2.848 | 1.256 | 0.470 | 1.164 | 2.326 | 1.642 | 1.603 | 1.322 | 0.866 | 0.351 |

Artifacts:

- `metrics/objective.csv`, `metrics/objective_summary.csv`
- `metrics/dnsmos.csv`, `metrics/dnsmos_summary.csv`
- `metrics/spk.csv`, `metrics/spk_summary.csv`
- `metrics/utmos.csv`, `metrics/utmos_summary.csv`
- `metrics/asr.csv`, `metrics/asr_summary.csv`
- `metrics/enroll_dev.pt`

## Baseline Reproduction Summary

The official Track 2 baseline was reproduced as an eval-only workflow:

- local official `AV_ConvTasNet` weights loaded strictly from
  `/home/avse/avse-assets/models/Real-World-AVSE-Baseline-Track2`;
- Track 2 dev was read through the repository `RealTestDataset` path using the
  compatibility symlink tree under the experiment directory;
- enhancement generated 5,250 target-speaker WAVs without NaN/Inf or all-zero
  output;
- all supported metrics were then computed from the fixed enhanced WAVs, so
  later metric reruns did not repeat baseline inference.

For future Track 2 dev evaluation, `/home/avse/avse_eval_py311` is the default
environment. The Python 3.8 environment remains documented because it produced
the original enhancement and Python 3.8-compatible metrics, but UTMOS and CER
should be run in Python 3.11.

Strict bit-for-bit reproduction of the README reference table is not a current
goal. The README values are kept as a sanity-check reference for scale and
direction; the local run is sufficient for baseline reproduction and evaluation
workflow validation.

## README Comparison

The README Track 2 dev reference row reports the official baseline reference
table refreshed in upstream commit `00e76b7`. This run is close on objective
metrics, DNSMOS, and speaker similarity. CER is higher than the refreshed
README overall because the local mix CER is higher; local remix CER is slightly
lower than the refreshed README remix value.

| scope | metric | this run | README | delta |
|---|---|---:|---:|---:|
| overall/remix | SI-SDR | -2.848 | -2.8507 | +0.0027 |
| overall/remix | PESQ | 1.256 | 1.2556 | +0.0004 |
| overall/remix | STOI | 0.470 | 0.4697 | +0.0003 |
| overall | UTMOS | 1.176 | 1.1777 | -0.0017 |
| overall | DNSMOS p808 | 2.370 | 2.3706 | -0.0006 |
| overall | DNSMOS sig | 1.745 | 1.7458 | -0.0008 |
| overall | DNSMOS bak | 1.704 | 1.7055 | -0.0015 |
| overall | DNSMOS ovr | 1.396 | 1.3960 | +0.0000 |
| overall | CER | 0.881 | 0.8707 | +0.0103 |
| overall | spk_sim | 0.370 | 0.3704 | -0.0004 |
| track2 / mix | UTMOS | 1.184 | 1.1875 | -0.0035 |
| track2 / mix | DNSMOS ovr | 1.449 | 1.4492 | -0.0002 |
| track2 / mix | CER | 0.892 | 0.8657 | +0.0263 |
| track2 / mix | spk_sim | 0.384 | 0.3844 | -0.0004 |
| track2 / remix | UTMOS | 1.164 | 1.1641 | -0.0001 |
| track2 / remix | DNSMOS ovr | 1.322 | 1.3220 | +0.0000 |
| track2 / remix | CER | 0.866 | 0.8776 | -0.0116 |
| track2 / remix | spk_sim | 0.351 | 0.3509 | +0.0001 |

Residual differences are small enough for the current baseline-reproduction
stage. The refreshed README table brings DNSMOS and objective metrics into
near-exact agreement with the local run; remaining CER differences are
scene-dependent and should be treated as ASR/runtime sensitivity rather than
inference failure.

## Smoke And Failure Notes

- Initial UTMOS smoke in Python 3.8 was blocked by Python version support.
- Initial CER smoke in Python 3.8 failed because the local Qwen3 config was not
  supported by Transformers `4.46.3`.
- The first Python 3.11 UTMOS smoke accidentally used `--gpus 0`, because
  `eval_real.py` overwrites `CUDA_VISIBLE_DEVICES` from its own argument. It
  OOMed on busy GPU 0 and produced no valid UTMOS values. Rerunning with
  `--gpus 6` succeeded.
- The final full UTMOS and ASR runs used `--gpus 6` and completed:
  UTMOS from 2026-07-04 11:34 to 12:52, ASR from 12:52 to 13:39.

## Code Notes

- `look2hear/datas/real_test_dataset.py` was adjusted to read NumPy-2-created
  landmark pickles from the Python 3.8 / NumPy 1.24 environment.
- `Fun-ASR/model.py` was adjusted so Python 3.8 can import its annotations.

## Follow-up Checks

- Decide whether to keep the NumPy pickle compatibility patch in the baseline
  branch before merging code changes, or regenerate landmark pickles with the
  target runtime.
- Keep `/home/avse/avse_eval_py311` as the default evaluation environment for
  Track 2 dev metrics, especially UTMOS and CER.
- README scores are treated as reference-scale values, not a strict
  bit-for-bit target for this phase.
