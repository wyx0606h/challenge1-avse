# Track 2 Dev Official Baseline

## Summary

- **Date:** 2026-07-02
- **Branch:** `exp/baseline-reproduction`
- **Purpose:** Reproduce official Track 2 baseline inference and available dev
  evaluation metrics from local `config.json` + `model.safetensors`.
- **Conventions:** See `docs/experiment-conventions.md`.
- **Status:** Inference complete; objective, DNSMOS, and speaker similarity
  complete. UTMOS and CER are blocked by the current Python 3.8 environment.

## Inputs

- **Repository:** `/home/avse/workspace-goodparts`
- **Dev data:** `/home/avse/data/track2_dev`
- **Evaluation assets:** `/home/avse/avse-assets/evaluation`
- **Official Track 2 weights:**
  `/home/avse/avse-assets/models/Real-World-AVSE-Baseline-Track2`
- **Experiment output:** `/home/avse/experiments/track2_dev_official_baseline`
- **Compatibility data root:**
  `/home/avse/experiments/track2_dev_official_baseline/data_root`

The released dev directory is flat (`mix/`, `remix/`, manifests). A symlink
tree was used so `RealTestDataset` could read its expected
`track2/dev/{mix,remix}` layout without copying or modifying the original data.

## Environment

- **Virtualenv:** `/home/avse/avse_gpu_venv`
- **Python:** `3.8.10`
- **PyTorch:** `2.4.1+cu121`
- **Torchaudio:** `2.4.1+cu121`
- **NumPy:** `1.24.4`
- **Transformers:** `4.46.3`
- **FunASR:** `1.3.1`
- **ModelScope:** `1.20.1`
- **GPU host check:** non-isolated `nvidia-smi` reported 8 x RTX 4090,
  driver `550.144.03`, CUDA `12.4`.
- **Sandbox note:** this Codex terminal does not expose `/dev/nvidia*`, so
  `torch.cuda.is_available()` reports `False` inside sandboxed commands.

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

Metrics were then run in eval-only mode over the saved WAVs:

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
```

CPU/GPU metric execution should have the same evaluation meaning here because
the enhancement WAVs were already fixed on disk. Device choice only affects
runtime, aside from tiny floating-point differences.

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

| scope | n | SI-SDR | PESQ | STOI | DNSMOS p808 | DNSMOS sig | DNSMOS bak | DNSMOS ovr | spk_sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **overall** | 5250 | -2.848 | 1.256 | 0.470 | 2.370 | 1.745 | 1.704 | 1.396 | 0.370 |
| track2 / mix | 3054 | -- | -- | -- | 2.402 | 1.819 | 1.776 | 1.449 | 0.384 |
| track2 / remix | 2196 | -2.848 | 1.256 | 0.470 | 2.326 | 1.642 | 1.603 | 1.322 | 0.351 |

Artifacts:

- `metrics/objective.csv`, `metrics/objective_summary.csv`
- `metrics/dnsmos.csv`, `metrics/dnsmos_summary.csv`
- `metrics/spk.csv`, `metrics/spk_summary.csv`
- `metrics/enroll_dev.pt`

## README Comparison

The README Track 2 dev reference row reports all metrics. This run reproduced
the metrics available in the current environment:

| scope | metric | this run | README | delta |
|---|---|---:|---:|---:|
| overall/remix | SI-SDR | -2.848 | -2.700 | -0.148 |
| overall/remix | PESQ | 1.256 | 1.243 | +0.013 |
| overall/remix | STOI | 0.470 | 0.469 | +0.001 |
| overall | DNSMOS p808 | 2.370 | 2.324 | +0.046 |
| overall | DNSMOS sig | 1.745 | 1.655 | +0.090 |
| overall | DNSMOS bak | 1.704 | 1.658 | +0.046 |
| overall | DNSMOS ovr | 1.396 | 1.356 | +0.040 |
| overall | spk_sim | 0.370 | 0.370 | +0.000 |
| track2 / mix | DNSMOS ovr | 1.449 | 1.384 | +0.065 |
| track2 / mix | spk_sim | 0.384 | 0.383 | +0.001 |
| track2 / remix | DNSMOS ovr | 1.322 | 1.319 | +0.003 |
| track2 / remix | spk_sim | 0.351 | 0.352 | -0.001 |

The reproduced values are close to the README table for objective and speaker
metrics. DNSMOS is consistently higher in this run, likely due metric/runtime
version differences; the README itself notes that exact checkpoint revision,
metric-cache revisions, command, and alignment setting still need to be
attached before treating that table as a strict reproducible reference.

## Missing Metrics

- **UTMOS:** not computed. The available UTMOSv2 package now requires Python
  `>=3.9`, while the current reusable GPU venv is Python `3.8.10`.
- **CER:** Fun-ASR smoke loaded the local remote code but failed on every item
  because the bundled `Qwen3-0.6B/config.json` requires Transformers `4.51.0`.
  Python 3.8 only exposed installable Transformers versions up to `4.46.3` in
  this environment, so `qwen3` is not recognized.

## Follow-up Environment For UTMOS/CER

The machine currently exposes only `/usr/bin/python3.8` on PATH and has no
`conda`, `mamba`, `micromamba`, `uv`, or `pyenv`. The recommended follow-up is
to create an isolated Python 3.11 evaluation environment outside the repository:

```bash
# Tool install location, outside Git.
cd /home/avse
curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
  -o /home/avse/micromamba-linux-64.tar.bz2
mkdir -p /home/avse/micromamba-bin
tar -xjf /home/avse/micromamba-linux-64.tar.bz2 -C /home/avse/micromamba-bin bin/micromamba

# New evaluation environment.
/home/avse/micromamba-bin/bin/micromamba create -y \
  -p /home/avse/avse_eval_py311 \
  -c pytorch -c nvidia -c conda-forge \
  python=3.11 pytorch torchaudio pytorch-cuda=12.1

/home/avse/avse_eval_py311/bin/python -m pip install \
  -r /home/avse/workspace-goodparts/requirements.txt \
  transformers==4.51.0 funasr modelscope utmosv2
```

After the environment is ready, rerun eval-only metrics over the already fixed
enhancement WAVs:

```bash
cd /home/avse/workspace-goodparts

# UTMOS, can be sharded if slow.
/home/avse/avse_eval_py311/bin/python eval_real.py \
  --conf_dir configs/track2_av_convtasnet.yml \
  --data_root /home/avse/experiments/track2_dev_official_baseline/data_root \
  --track track2 --scene both --split dev \
  --metrics utmos --mode eval \
  --save_dir /home/avse/experiments/track2_dev_official_baseline/full/enhanced \
  --out_csv /home/avse/experiments/track2_dev_official_baseline/metrics/utmos.csv

# CER with local Fun-ASR resources.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 \
  /home/avse/avse_eval_py311/bin/python eval_real.py \
  --conf_dir configs/track2_av_convtasnet.yml \
  --data_root /home/avse/experiments/track2_dev_official_baseline/data_root \
  --track track2 --scene both --split dev \
  --metrics asr --mode eval \
  --save_dir /home/avse/experiments/track2_dev_official_baseline/full/enhanced \
  --out_csv /home/avse/experiments/track2_dev_official_baseline/metrics/asr.csv \
  --funasr_model /home/avse/avse-assets/evaluation/funasr/Fun-ASR-Nano-2512 \
  --funasr_vad_model /home/avse/avse-assets/evaluation/funasr/fsmn-vad \
  --funasr_remote_code Fun-ASR/model.py
```

On 2026-07-02, Codex attempted to download micromamba for this follow-up
environment, but the network action was blocked by the current Codex usage
limit. No partial environment was created.

## Code Notes

- `look2hear/datas/real_test_dataset.py` was adjusted to read NumPy-2-created
  landmark pickles from the Python 3.8 / NumPy 1.24 environment.
- `Fun-ASR/model.py` was adjusted so Python 3.8 can import its annotations.

## Before Merge To Main

- Decide whether to keep the NumPy pickle compatibility patch in the baseline
  branch, or regenerate landmark pickles with the target runtime.
- Create a Python `>=3.9` evaluation environment before rerunning UTMOS and CER.
- If README values are used as a release-quality reference, pin metric package
  versions and record whether landmark alignment fallback occurred.
