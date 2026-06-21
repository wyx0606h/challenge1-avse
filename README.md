# Real-World Audio-Visual Speech Enhancement (AVSE) Challenge — Baseline

**ISCSLP 2026 Special Session**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%20%7C%20cu126-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.2+-792EE5.svg?logo=lightning&logoColor=white)](https://lightning.ai/)
[![ISCSLP 2026](https://img.shields.io/badge/ISCSLP-2026-1f6feb.svg)](#)
[![Tracks](https://img.shields.io/badge/Tracks-Real--World%20%7C%20Visual%20Degradation-success.svg)](#)

This repository hosts the official baseline system for the **Real-World AVSE Challenge** at ISCSLP 2026. It provides an end-to-end audio-visual speech enhancement (AVSE) pipeline — data preparation, an AV-ConvTasNet baseline model, training scripts, and a comprehensive offline evaluation suite — for two complementary tracks that push AVSE from the "clean video + synthetic speech" setting toward real-world deployment.

---

## 1. Background

Audio-only speech enhancement still struggles under low SNR, strong reverberation, and overlapping speakers. A speaker's lip movements and facial cues are tied to speech content and are immune to acoustic noise, which makes **audio-visual** speech enhancement one of the most promising directions in the field. Yet a large gap remains between academic benchmarks and real deployment:

- Mainstream datasets follow a *clean-video + synthetic-mixture* paradigm that misses the natural speech overlap, device variability, and acoustic complexity of real rooms.
- The visual stream in practice is often imperfect — occlusion, side views, motion blur, lighting changes, low resolution, dropped frames, even total loss — and model robustness under these conditions is rarely studied systematically.

This challenge tackles both gaps through two tracks:

| Track | Theme | What it tests |
|-------|-------|---------------|
| **Track 1** | Real-World Mixed Scenarios | Multi-speaker audio-visual data captured naturally, with speech mixed organically rather than synthetically — a realistic test of robustness and practicality. |
| **Track 2** | Visual Degradation | Visually degraded samples (occlusion, low quality, missing frames, blur) built on public data, to systematically probe how models hold up when the visual modality becomes unreliable. |

---

## 2. Challenge Schedule

| Date (2026) | Milestone |
|-------------|-----------|
| Jun 16 | Challenge registration opens |
| **Jun 22** | Release of baseline system, training-data references, and **development set** (Track 1 & 2) |
| Jul 12 | **Test set** released; leaderboard opens (≤ 3 submissions/team/day) |
| Jul 17 | Registration closes |
| Jul 24 | Leaderboard frozen |
| Jul 27 | Final rankings announced |
| Aug 3 | Paper submission deadline |
| Aug 31 | Paper acceptance notification |
| Sep 21 | Camera-ready deadline |

Participants develop and self-evaluate on the **dev** set (which ships clean references for the synthetic portion). The organizers score the held-out **test** set.

---

## 3. The Baseline System

### 3.1 Model — AV-ConvTasNet

A two-stream audio-visual separator that recovers one target speaker from a mixture, conditioned on that speaker's lip video:

- **Audio branch** — a Conv-TasNet separator (`N=256` encoder filters, `L=40` filter length, `B=256` bottleneck, `H=512` conv channels, `X=8` blocks × `R=4` repeats).
- **Video branch** — a *frozen* ResNet-34 lip-reading encoder (3D stem + ResNet-34, 512-d output) bundled inside the model. Its weights come from `video_pretrain/frcnn_128_512.backbone.pth.tar` at init and are then carried inside every checkpoint.
- **Fusion** — video features are temporally aligned to the audio bottleneck and concatenated (`F=256`).
- **Forward** — `model(mix_waveform[B,T], lip_frames[B,Tv,88,88]) -> enhanced[B,T]`.

Defining files: [look2hear/models/av_convtasnet.py](look2hear/models/av_convtasnet.py), [look2hear/videomodels/](look2hear/videomodels/). The model follows the [AV-ConvTasNet](https://github.com/JusperLee/AV-ConvTasNet) reference implementation.

### 3.2 Training

- Framework: PyTorch-Lightning + DDP (multi-GPU), SwanLab logging.
- Data: **dynamic on-the-fly two-speaker remix** over VoxCeleb2 single-speaker sources (Track 1), with online **visual degradation** of the target face for Track 2 (full-frame mask, lip occlusion, low resolution, frame freeze, AV desync).
- Loss: permutation-invariant SI-SDR/SNR (`PITLossWrapper`).
- Optimizer: Adam (`lr=1e-3`), `ReduceLROnPlateau`.

Entry points: [train.py](train.py), [run_train.sh](run_train.sh), configs in [configs/](configs/). The training pipeline is adapted from [Dolphin](https://github.com/JusperLee/Dolphin).

```bash
# Track 1 (clean video)
python train.py --conf_dir configs/track1_av_convtasnet.yml
# Track 2 (visual degradation)
python train.py --conf_dir configs/track2_av_convtasnet.yml
```

Checkpoints are PyTorch-Lightning files (`epoch=*.ckpt`, `last.ckpt`) whose weights live under an `audio_model.` prefix; `best_model.pth` is a self-describing `serialize()` payload (`model_name` + `state_dict` + `model_args`). The evaluation scripts rebuild the model from `conf.yml` and load either format automatically.

---

## 4. Dataset Layout

Each track has a `dev` and a `test` split, and each split has two **scenes**:

- **`mix`** — REAL recordings of two people talking at once. The `mix.wav` is a real noisy mixture; there is **no clean ground truth** → no-reference metrics only.
- **`remix`** — SYNTHETIC mixtures (two clean single-speaker clips summed), so they can **ship clean `s1.wav`/`s2.wav`** → reference-based metrics (SI-SDR/PESQ/STOI) *plus* the no-reference ones.

Every clip yields **two evaluation items** (one per target speaker, `s1` and `s2`); the model is run once per target with that speaker's lip video. Audio is 16 kHz mono; faces are 256×256 mp4 at 25 fps, aligned with the audio.

### What participants receive (`Real-World-AVSE/`)

The released package follows the standard challenge handout — **`dev` ships ground truth for self-evaluation; `test` is withheld** (only the model inputs):

```
Real-World-AVSE/track{1,2}/
├── dev/                              # full self-eval assets
│   ├── mix/<id>/        mix.wav  s1.mp4 s2.mp4  s1.txt s2.txt
│   ├── remix/<id>/      mix.wav  s1.mp4 s2.mp4  s1.txt s2.txt  s1.wav s2.wav   # clean GT
│   ├── mix_manifest.json            # mix_id, s1_speaker, s2_speaker
│   └── remix_manifest.json          # + s1_id, s2_id
└── test/                            # inputs only — GT withheld
    ├── mix/<id>/        mix.wav  s1.mp4 s2.mp4
    └── remix/<id>/      mix.wav  s1.mp4 s2.mp4          # NO clean wav, NO txt, NO manifest
```

| Asset | dev | test |
|-------|:---:|:---:|
| `mix.wav`, `s1/s2.mp4` (model inputs) | ✓ | ✓ |
| `s1/s2.txt` transcripts (CER) | ✓ | ✗ |
| `remix` clean `s1/s2.wav` (SI-SDR/PESQ/STOI) | ✓ | ✗ |
| `mix/remix_manifest.json` (speaker labels) | ✓ | ✗ |

So **participants develop and self-score on `dev`** with the full metric suite; the organizers score the held-out `test`. On the released `test` (no transcripts/clean refs/manifest), CER and reference-based metrics cannot be computed locally, and `eval_real.py` has no manifest to enumerate clips or read speaker labels — test scoring is organizer-side, using organizer-held voiceprints (see §6.2).

> **Note:** `dev` and `test` use **disjoint speaker sets**, so a voiceprint built on one split does not transfer to the other — each split is self-contained for enrollment.

The visual front end is shared with training: face mp4 → grayscale → resize to 96 → center-crop 88 → normalize (mean 0.421, std 0.165). ([look2hear/datas/real_test_dataset.py](look2hear/datas/real_test_dataset.py), [look2hear/datas/transform.py](look2hear/datas/transform.py))

---

## 5. Setup

The compute cluster is assumed to be **offline**. Set up the environment, then fetch all metric weights once on an internet-connected machine.

```bash
# 1. Create the conda env + install all deps (PyTorch CUDA 12.6, funasr, etc.)
bash install.sh                       # env name: real-avse
conda activate real-avse

# 2. Download pretrained metric weights (RUN ON AN INTERNET-CONNECTED NODE).
#    Caches UTMOSv2, DNSMOS onnx, Fun-ASR-Nano, and WeSpeaker locally.
bash download_models.sh
#    HF blocked but a mirror reachable? ->  HF_ENDPOINT=https://hf-mirror.com bash download_models.sh
```

What `download_models.sh` fetches and where it caches:

| Model | Purpose | Cache location |
|-------|---------|----------------|
| UTMOSv2 + wav2vec2-base | No-reference MOS | `~/.cache/utmosv2`, HF cache |
| DNSMOS (3 onnx) | No-reference P.835 | `~/.torchmetrics/DNSMOS` |
| Fun-ASR-Nano-2512 | Chinese ASR (CER) | `~/.cache/modelscope/hub` |
| WeSpeaker cnceleb-resnet34-LM | Speaker embedding | `pretrained/wespeaker_cnceleb_resnet34/` |

Key dependency versions are pinned in [requirements.txt](requirements.txt) (torch 2.7.1/cu126, torchmetrics, funasr, onnxruntime, transformers, modelscope). DNSMOS uses the **CPU** onnxruntime build by default (the onnx models are tiny); pass `GPU_ONNX=1 bash install.sh` only if you have a matching CUDA.

---

## 6. Evaluation

The evaluation entry point is [eval_real.py](eval_real.py), wrapped by [run_eval_real.sh](run_eval_real.sh) for offline multi-GPU runs.

### 6.1 Metric suite

| Metric | Type | Scenes | Backend |
|--------|------|--------|---------|
| **SI-SDR / PESQ / STOI** | reference-based | `remix` only (needs clean GT) | torchmetrics |
| **UTMOS** | no-reference MOS | mix + remix | UTMOSv2 |
| **DNSMOS** (p808/sig/bak/ovr) | no-reference P.835 | mix + remix | torchmetrics (onnx) |
| **CER** | text (Chinese) | mix + remix | Fun-ASR-Nano-2512 |
| **Speaker similarity** | identity preservation | mix + remix | WeSpeaker ResNet34 |

Metric details:

- **CER** — the enhanced speech is transcribed by Fun-ASR-Nano and compared to the reference `*.txt` after punctuation/whitespace stripping; character-level edit distance / reference length. Chinese has no word boundaries, so CER (not WER) is the text metric.
- **Speaker similarity** — cosine similarity between the WeSpeaker embedding of the enhanced speech and a per-speaker **enrollment voiceprint** (256-d, L2-normalized; ResNet34 + TSTP, 80-d kaldi-fbank front end). The WeSpeaker ResNet34 architecture is reproduced standalone in [look2hear/models/wespeaker_resnet34.py](look2hear/models/wespeaker_resnet34.py) — no `wespeaker` install needed.
- **Reference-based metrics** are computed only where a clean reference exists (`remix`).

All metric models are loaded **lazily** — a run only touches the weights for the metrics you request, so a subset (or `--metrics none` pipeline smoke test) needs nothing downloaded.

### 6.2 Speaker enrollment

Speaker similarity needs a clean voiceprint per speaker. Voiceprints are built from the **clean single-speaker sources** in `remix` (`s1.wav`/`s2.wav`), grouped by speaker label and mean-pooled. Because extracting embeddings over thousands of clips is slow and identical across runs, precompute them once:

```bash
# dev voiceprints (speakers seen in dev) — can be released with the baseline
python build_enrollment.py --split dev  --out pretrained/enroll_dev.pt
# test voiceprints — organizer-side only, NOT released
python build_enrollment.py --split test --out pretrained/enroll_test.pt
```

Then pass `--enroll_ckpt` (or `ENROLL_CKPT=...` to the wrapper) to skip the rebuild. If omitted, `eval_real.py` rebuilds voiceprints on the fly from the split's clean sources.

> **Methodology note:** the enrollment voiceprint of a speaker is the mean over all its clean `remix` sources, **including** the source of the clip currently being scored (no leave-one-out). Because each speaker has many dozens to a few hundred sources, the self-term is a small fraction (≈ 1/N) of the centroid and the inflation of `remix` speaker-similarity is minor; it is documented here for transparency.

### 6.3 Run modes

`--mode` decouples inference from scoring:

| Mode | What it does | Loads separation model? | Loads metric models? |
|------|--------------|:---:|:---:|
| `enhance` | run the model, save enhanced wavs only | ✓ | ✗ |
| `eval` | score pre-saved wavs in `--save_dir` | ✗ | ✓ |
| `both` (default) | enhance + score in one pass | ✓ | ✓ |

Enhanced audio mirrors the corpus layout — `<save_dir>/<track>/<split>/<scene>/<clip_id>/s{1,2}.wav` — and is written as 32-bit float PCM so `enhance` → `eval` reproduces the in-memory `both` numbers exactly.

### 6.4 Multi-GPU sharding

A comma-separated GPU list runs one shard per GPU (`items[shard_id::num_shards]`); each shard writes its own CSV and the wrapper merges them when all finish.

### 6.5 Quick start

```bash
# Participants: score the dev set on 4 GPUs with the released dev voiceprints
ENROLL_CKPT=pretrained/enroll_dev.pt \
  bash run_eval_real.sh "" both both all 0,1,2,3      # SPLIT defaults to dev

# Organizers: score the held-out test set
SPLIT=test bash run_eval_real.sh "" both both all 0,1,2,3

# Pipeline smoke test (no metric weights needed): 5 clips/scene, model only
bash run_eval_real.sh "" both both none 0 5
```

`run_eval_real.sh` positional args: `[CKPT] [TRACK] [SCENE] [METRICS] [GPUS] [LIMIT] [MODE]`.
Env overrides: `SPLIT` (default `dev`), `SAVE_DIR`, `ENROLL_CKPT`, `DATA_ROOT`.

Calling `eval_real.py` directly exposes the full flag set:

```bash
python eval_real.py \
  --track both --scene both --split dev --metrics all \
  --mode both --save_dir enhanced_out \
  --enroll_ckpt pretrained/enroll_dev.pt --gpus 0
```

| Flag | Meaning |
|------|---------|
| `--track` | `track1` / `track2` / `both` |
| `--scene` | `mix` (real, no GT) / `remix` (synthetic, has GT) / `both` |
| `--split` | `dev` / `test` |
| `--metrics` | comma list of `{utmos,dnsmos,asr,spk,objective}`, `all`, or `none` |
| `--mode` | `enhance` / `eval` / `both` |
| `--save_dir` | enhanced-audio dir (required for `enhance`/`eval`) |
| `--enroll_ckpt` | precomputed voiceprints `.pt` |
| `--num_shards` / `--shard_id` / `--merge_shards` | multi-GPU sharding |
| `--ckpt` / `--conf_dir` | checkpoint + model config (defaults to the latest ckpt next to `conf.yml`) |
| `--limit` | cap clips per (track, scene) for smoke tests |

### 6.6 Outputs

Three CSVs are written next to `--out_csv` (default `<exp_dir>/results/real_metrics.csv`):

- `real_metrics.csv` — per-item rows (key, track, scene, speaker_id, all metrics, ref/hyp text).
- `real_metrics_<track>_<scene>.csv` — per-group per-item rows.
- `real_metrics_summary.csv` — mean per metric, overall and per (track, scene) group.

---

## 7. Repository Map

```
configs/                 training configs (track1/track2/baseline)
look2hear/
├── datas/               datasets: VoxCeleb2 remix (train) + REAL-AVSE reader (eval)
├── models/              AV-ConvTasNet + standalone WeSpeaker ResNet34
├── videomodels/         frozen ResNet lip-reading encoder
├── system/              PyTorch-Lightning module
├── losses/              SI-SDR/SNR PIT loss
└── metrics/             sim_metrics (reference-based) + real_metrics (no-ref/CER/spk)
DataPreProcess/          VoxCeleb2 manifest builder + train/val/test splits
preprocess/              REAL-AVSE corpus preprocessing (landmarks, perturbation, remix)
Fun-ASR/  wespeaker/      vendored source for the ASR / speaker-embedding models
video_pretrain/          frozen lip-reading backbone weights
pretrained/              WeSpeaker checkpoint + enrollment voiceprints
train.py  test.py  eval_real.py  build_enrollment.py
run_train.sh  run_eval_real.sh  download_models.sh  install.sh
```

---

## 8. Organizers

Kai Li (Tsinghua University) · Wenze Ren (National Taiwan University) · Junjie Li (The Hong Kong Polytechnic University) · Cheng Yu (Ohio State University) · Peijun Yang (Wuhan University) · Haibin Wu (Meta) · Szu-Wei Fu (NVIDIA) · Wen-Chin Huang (Nagoya University) · Hsin-Min Wang (Academia Sinica) · Xiaolin Hu (Tsinghua University) · Ming Li (CUHK-Shenzhen) · DeLiang Wang (CUHK-Shenzhen) · Yu Tsao (Academia Sinica)

**Contact:** `realworldavse_iscslp2026@googlegroups.com`

---

## 9. Acknowledgements

The baseline model is based on [AV-ConvTasNet](https://github.com/JusperLee/AV-ConvTasNet), and the training pipeline is adapted from [Dolphin](https://github.com/JusperLee/Dolphin). The evaluation suite integrates [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2), DNSMOS (via torchmetrics), [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR), and [WeSpeaker](https://github.com/wenet-e2e/wespeaker). We thank the authors of these projects.

---

## 10. License

Released under the [Apache License 2.0](LICENSE). Vendored components (`Fun-ASR/`, `wespeaker/`) and the integrated metric models retain their own licenses.
