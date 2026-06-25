# Real-World Audio-Visual Speech Enhancement (AVSE) Challenge — Baseline

**ISCSLP 2026 Special Session**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ISCSLP 2026](https://img.shields.io/badge/ISCSLP-2026-1f6feb.svg)](#)
[![Tracks](https://img.shields.io/badge/Tracks-Real--World%20%7C%20Visual%20Degradation-success.svg)](#)
[![HF Track 1](https://img.shields.io/badge/🤗%20Model-Track%201-yellow.svg)](https://huggingface.co/JusperLee/Real-World-AVSE-Baseline-Track1)
[![HF Track 2](https://img.shields.io/badge/🤗%20Model-Track%202-yellow.svg)](https://huggingface.co/JusperLee/Real-World-AVSE-Baseline-Track2)

## Team workspace status

This repository preserves the official Real-World AVSE baseline and adds a
team-facing reproducibility and experiment workflow. The official baseline
implementation remains the reference layer; research changes must be developed
on separate branches and compared against a recorded baseline run.

Current team status:

| Item | Status |
|---|---|
| Official baseline source imported | Completed |
| Track 2 setup/data validation tools | Completed |
| Stable source baseline | `main` at tag `baseline/track2-source-v1` after the repository-organization merge |
| Server environment | TODO — not yet audited on the target server |
| Official challenge `dev` access | TODO — team registration is pending |
| Training data source and license | TODO — not yet confirmed |
| Hugging Face checkpoint access | Completed — gated files from revision `3777a458d13e8dc98877f69e542d6a8a7441835b` were downloaded and checksummed outside Git |
| Baseline smoke test on real challenge data | TODO — blocked by data and environment |
| Full baseline reproduction | TODO — no team reproduction result is claimed yet |
| Available compute | Expected: 3× RTX 4090 + 1× RTX 5090; driver, VRAM, CPU, RAM, disk, and mixed-GPU compatibility are TODO |

Do not interpret the reference scores later in this README as team reproduction
results. A result is considered reproduced only after it has a recorded Git
commit, configuration, checkpoint identity, data version, random seed, command,
environment snapshot, logs, and complete metrics.

### Branch and baseline policy

The repository uses two different kinds of stability:

- `main` is the reviewed stable branch. Direct development on `main` is
  forbidden.
- `baseline/track2-source-v1` is an immutable tag that preserves the exact
  official-source-plus-workflow starting point. Moving or recreating this tag
  is forbidden.
- `codex/bootstrap-track2` is an archival bootstrap branch. It documents the
  one-time official baseline import and is not a continuing experiment branch.
- `exp/*` branches contain baseline reproduction or one focused research
  change. New experiments start from the immutable baseline tag or the
  corresponding reviewed `main` commit.

A successful experiment does not modify the baseline tag. It must first record
its code, configuration, data, seed, environment, checkpoint, logs, and metrics.
Only an explicitly reviewed and validated change may later be considered for
`main`; failed or inconclusive experiments remain recorded on their own
branches and are not merged.

The source baseline tag does **not** mean that the team has reproduced the
official score. Baseline metric reproduction is tracked separately as
`EXP-001`.

### Team workflow

```text
Clone repository
→ check out baseline/track2-source-v1 or reviewed main
→ create a personal or exp/* branch
→ configure the environment
→ prepare and audit data
→ reproduce the unchanged baseline
→ record baseline metrics and logs
→ freeze the baseline reproduction commit
→ create one branch per research change
→ inspect git diff
→ run a small sanity check
→ run the formal experiment
→ record configuration, logs, metrics, and checkpoint path
→ update experiment documentation
→ commit and push the experiment branch
→ merge only after review and validation
```

Repository collaboration documents:

- [AGENTS.md](AGENTS.md) — rules for human and automated contributors.
- [EXPERIMENTS.md](EXPERIMENTS.md) — experiment index and required metadata.
- [docs/experiment-template.md](docs/experiment-template.md) — copyable
  per-experiment record.
- [docs/TRACK2_CHALLENGE_GUIDE_ZH.md](docs/TRACK2_CHALLENGE_GUIDE_ZH.md) —
  Track 2 reproduction guide.
- [UPSTREAM.md](UPSTREAM.md) — official baseline provenance and upstream sync.

### Storage boundaries

The Git repository stores source code, configurations, documentation, and small
metadata only. Keep the following outside normal Git history:

| Asset | Recommended location | Git policy |
|---|---|---|
| Training/challenge data | External shared storage | Never commit |
| Pretrained and trained weights | External model storage or ignored local directory | Never commit |
| Checkpoints and training logs | External experiment root | Never commit |
| Enhanced audio and submissions | External output root | Never commit |
| Experiment records and small metric summaries | `EXPERIMENTS.md`, `docs/experiments/`, or another reviewed documentation path | Commit after verification |

The current training code writes under `Experiments/` by default. That directory
is ignored, but on the server the preferred long-term output root and storage
quota are **TODO**. Do not hard-code a personal server path into tracked files.

### Environment variables and Hugging Face access

There is no `HF_ACCESS` variable in the codebase. `HF_ACCESS` is only a status
label meaning “has the user obtained access to the required Hugging Face
repository?” Hugging Face libraries use their standard authentication:

```bash
huggingface-cli login
# or provide HF_TOKEN through the shell/job secret mechanism
export HF_TOKEN=<token>
```

Never put a real token in Git, YAML configuration, shell scripts, notebooks, or
experiment records. An optional [.env.example](.env.example) documents variable
names only; `.env` is ignored and is not loaded automatically by this project.

Common runtime overrides include:

```bash
export DATA_ROOT=/external/path/to/Real-World-AVSE
export SAVE_DIR=/external/path/to/outputs
export ENROLL_CKPT=/external/path/to/enroll_dev.pt
```

The exact server paths are **TODO**. Verify them after the server repository,
data, and storage layout are available.

### Baseline reproduction checklist

1. Confirm the target branch and a clean `git status`.
2. Confirm the parent is `baseline/track2-source-v1` or its reviewed `main`
   commit and record `git rev-parse HEAD`.
3. Audit server GPU, driver, CUDA, Python, PyTorch, CPU, RAM, and disk.
4. Install the isolated environment using the instructions below.
5. Confirm challenge registration and data access; do not assume `dev` assets.
6. Confirm the training data source, license, and speaker-disjoint splits.
7. Log in to Hugging Face, accept the Track 2 model repository's
   contact-sharing condition, and verify that `config.json` and
   `model.safetensors` can be downloaded.
8. Run `tools/check_track2_setup.py`.
9. Run the one-item-per-scene smoke test.
10. Run the complete baseline only after the smoke test succeeds.
11. Record the run in `EXPERIMENTS.md` and a copied experiment template.

Useful commands after the required assets exist:

```bash
# Environment and optional training-asset checks
python tools/check_track2_setup.py --check-env
python tools/check_track2_setup.py --check-training-assets

# Challenge data audit (DATA_ROOT is TODO until the team receives the data)
python tools/check_track2_setup.py --data-root "$DATA_ROOT"

# Minimal Track 2 inference check
DATA_ROOT="$DATA_ROOT" GPU=0 bash scripts/smoke_track2.sh
```

The current scripts contain upstream machine defaults and fixed GPU lists.
Treat those as reference defaults, not as the team server configuration.
Server-specific cleanup is a separate code change and must not be mixed into a
documentation-only baseline record.

---

This repository hosts the official baseline system for the **Real-World AVSE Challenge** at ISCSLP 2026. It provides an end-to-end audio-visual speech enhancement (AVSE) pipeline — data preparation, an AV-ConvTasNet baseline model, training scripts, and a comprehensive offline evaluation suite — for two complementary tracks that push AVSE from the "clean video + additive mixing" setting toward real-world deployment.

---

## 1. Background

Audio-only speech enhancement still struggles under low SNR, strong reverberation, and overlapping speakers. A speaker's lip movements and facial cues are tied to speech content and are immune to acoustic noise, which makes **audio-visual** speech enhancement one of the most promising directions in the field. Yet a large gap remains between academic benchmarks and real deployment:

- Mainstream datasets follow a *clean-video + additive-mixture* paradigm that misses the natural speech overlap, device variability, and acoustic complexity of real rooms.
- The visual stream in practice is often imperfect: occlusion, side views, motion blur, lighting changes, low resolution, dropped frames, and even total loss. Model robustness under these conditions is rarely studied systematically.

This challenge tackles both gaps through two tracks:

| Track | Theme | What it tests |
|-------|-------|---------------|
| **Track 1** | Real-World Mixed Scenarios | Multi-speaker audio-visual data captured naturally, with speech mixed organically rather than synthetically — a realistic test of robustness and practicality. |
| **Track 2** | Visual Degradation | Visually degraded samples (occlusion, low resolution, frame freeze, missing/dropped frames, and audio-visual desynchronization) built on public data, to systematically probe how models hold up when the visual modality becomes unreliable. |

---

## 2. Challenge Schedule

| Date (2026) | Milestone |
|-------------|-----------|
| Jun 22 | Challenge registration opens |
| **Jun 23** | Release of baseline system, training-data references, and **development set** (Track 1 & 2) |
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

**Training data (VoxCeleb2).** The configs read pre-built manifests from `DataPreProcess/vox2/{tr,cv,tt}/`. Build them once from a local VoxCeleb2 mixture set (`tr/cv/tt` each holding `mix/s1/s2` 16 kHz wavs) plus the whole-face crops:

```bash
python DataPreProcess/process_vox2.py \
  --in_audio_dir  /path/to/vox2/wav16k/min \
  --in_visual_dir /path/to/vox2/faces \
  --visual_ext .mp4 \
  --out_dir DataPreProcess/vox2
```

This writes `mix.json` / `s1.json` / `s2.json` per split (see [DataPreProcess/process_vox2.py](DataPreProcess/process_vox2.py) for the expected layout and filename convention).

**Lip-reading backbone.** Training initialises the frozen video branch from `video_pretrain/frcnn_128_512.backbone.pth.tar` (≈ 259 MB). Download it manually from Google Drive and place it at that path — it is **not** needed for evaluation, since the trained video weights are baked into every checkpoint:

> **Lip-reading backbone:** <https://drive.google.com/file/d/13-T3nBnf21-lMKrV_XbH6Lf4vK2xU7lS/view> → `video_pretrain/frcnn_128_512.backbone.pth.tar`

`download_models.sh` (step 5) checks for this file and prints the link if it is missing.

```bash
# Track 1 (clean video)
python train.py --conf_dir configs/track1_av_convtasnet.yml
# Track 2 (visual degradation)
python train.py --conf_dir configs/track2_av_convtasnet.yml
```

**Skip training — use the released baseline checkpoints.** Both baseline models are published on HuggingFace with the standard [`PyTorchModelHubMixin`](https://huggingface.co/docs/huggingface_hub/guides/integrations) layout (`config.json` + `model.safetensors`). The model page is public, but downloading the gated files requires a logged-in account to accept the contact-sharing condition. Once access is accepted, the model loads from a **repo id alone** and the files download into the Hugging Face cache on first use:

| Track | HuggingFace repo |
|-------|------------------|
| Track 1 | [`JusperLee/Real-World-AVSE-Baseline-Track1`](https://huggingface.co/JusperLee/Real-World-AVSE-Baseline-Track1) |
| Track 2 | [`JusperLee/Real-World-AVSE-Baseline-Track2`](https://huggingface.co/JusperLee/Real-World-AVSE-Baseline-Track2) |

```bash
# One-time: accept the model-page condition in a browser, then authenticate
huggingface-cli login            # or provide HF_TOKEN through a secret mechanism

# Track 2 baseline — weights auto-download on first use
python eval_real.py \
  --conf_dir configs/track2_av_convtasnet.yml \
  --ckpt JusperLee/Real-World-AVSE-Baseline-Track2 \
  --data_root "$DATA_ROOT" \
  --track track2 --scene both --split dev \
  --metrics all --mode both --save_dir enhanced_out
```

Or load it directly in Python — the trained video encoder is bundled in the weights, so no lip-reading backbone file is needed:

```python
from look2hear.models import AV_ConvTasNet
model = AV_ConvTasNet.from_pretrained(
    "JusperLee/Real-World-AVSE-Baseline-Track2"
).eval()
```

For offline use, keep `config.json` and `model.safetensors` together in one
external directory and pass the directory itself as `--ckpt`:

```bash
export TRACK2_MODEL_DIR=/external/path/to/Real-World-AVSE-Baseline-Track2

python eval_real.py \
  --conf_dir configs/track2_av_convtasnet.yml \
  --ckpt "$TRACK2_MODEL_DIR" \
  --data_root "$DATA_ROOT" \
  --track track2 --scene both --split dev \
  --metrics none --mode enhance --save_dir enhanced_out/smoke-local \
  --gpus 0 --limit 1
```

The team snapshot was downloaded from Hugging Face revision
`3777a458d13e8dc98877f69e542d6a8a7441835b`:

| File | Size | SHA256 |
|---|---:|---|
| `config.json` | 279 bytes | `de1819318f7c3fb79914314475ab4b223973af9104d42600e533cdb557bcf5f8` |
| `model.safetensors` | 100,176,860 bytes | `600d1632346e9b85e64f2842000f41d2cdb9f6b1afca8011c5decdc5ce9e78c1` |

Keep these files outside Git and provide the location through the shell or job
configuration. `--ckpt` also accepts a local Lightning `*.ckpt` or serialized
`best_model.pth`.

---

## 4. Dataset Layout

Each track has a `dev` and a `test` split, and each split has two **scenes**:

- **`mix`** — REAL recordings of two people talking at once. The `mix.wav` is a real noisy mixture; there is **no clean ground truth** → no-reference metrics only.
- **`remix`** — SYNTHETIC mixtures (two clean single-speaker clips summed), so they can **ship clean `s1.wav`/`s2.wav`** → reference-based metrics (SI-SDR/PESQ/STOI) *plus* the no-reference ones.

Every clip yields **two evaluation items** (one per target speaker, `s1` and `s2`); the model is run once per target with that speaker's lip video. Audio is 16 kHz mono; faces are 256×256 mp4 at 25 fps, aligned with the audio.

**Track composition.** Track 2 is **not** a separate, smaller corpus — it is the **entire Track 1 set with its target-speaker video degraded offline** (occlusion, low resolution, frame freeze, dropped frames, AV desync), **plus** Track 2's own additional far-field (3 m) recordings. So every Track 1 clip id appears in Track 2 with a re-encoded (degraded) `s{1,2}.mp4` but the **same `mix.wav`**, and Track 2 adds its 3 m material on top. Both tracks' manifests enumerate the full clip set for that track.

| split | scene | Track 1 | Track 2 (= Track 1 degraded + 3 m) |
|-------|-------|:-------:|:----------------------------------:|
| dev   | mix   | 1242    | 1527 |
| dev   | remix | 900     | 1098 |
| test  | mix   | 2472    | 2820 |
| test  | remix | 1785    | 2121 |

Each entry expands to two eval items (`s1`/`s2`), so e.g. Track 2 `dev` enumerates `(1527 + 1098) × 2 = 5250` items. When sharded across N GPUs, each shard's progress bar shows ≈ items/N — Track 2's bar is larger than Track 1's because Track 2 is a superset.

### What participants receive (`Real-World-AVSE/`)

The released package follows the standard challenge handout — **`dev` ships ground truth for self-evaluation; `test` is withheld** (only the model inputs):

```
Real-World-AVSE/track{1,2}/
├── dev/                              # full self-eval assets
│   ├── mix/<id>/        mix.wav  s1.mp4 s2.mp4  s1.pkl s2.pkl  s1.txt s2.txt
│   ├── remix/<id>/      mix.wav  s1.mp4 s2.mp4  s1.pkl s2.pkl  s1.txt s2.txt  s1.wav s2.wav   # clean GT
│   ├── mix_manifest.json            # mix_id, s1_speaker, s2_speaker
│   └── remix_manifest.json          # + s1_id, s2_id
└── test/                            # inputs only — GT withheld
    ├── mix/<id>/        mix.wav  s1.mp4 s2.mp4  s1.pkl s2.pkl
    └── remix/<id>/      mix.wav  s1.mp4 s2.mp4  s1.pkl s2.pkl   # NO clean wav, NO txt, NO manifest
```

| Asset | dev | test |
|-------|:---:|:---:|
| `mix.wav`, `s1/s2.mp4` (model inputs) | ✓ | ✓ |
| `s1/s2.pkl` precomputed face landmarks | ✓ | ✓ |
| `s1/s2.txt` transcripts (CER) | ✓ | ✗ |
| `remix` clean `s1/s2.wav` (SI-SDR/PESQ/STOI) | ✓ | ✗ |
| `mix/remix_manifest.json` (speaker labels) | ✓ | ✗ |

So **participants develop and self-score on `dev`** with the full metric suite; the organizers score the held-out `test`. On the released `test` (no transcripts/clean refs/manifest), CER and reference-based metrics cannot be computed locally, and `eval_real.py` has no manifest to enumerate clips or read speaker labels — test scoring is organizer-side, using organizer-held voiceprints (see §6.2).

> **Note:** `dev` and `test` use **disjoint speaker sets**, so a voiceprint built on one split does not transfer to the other — each split is self-contained for enrollment.

Each `s{1,2}.pkl` holds the **precomputed face landmarks** for that speaker's `s{1,2}.mp4`: a Python list with one entry per video frame, each entry either a `68×2` float32 array of facial landmark coordinates or `None` when no face was detected in that frame. They are derived from the released mp4 and carry no ground-truth audio information.

In this repository's current evaluation path, landmark alignment is enabled by
default. The reader computes one stable square crop per clip from the median
valid 68-point landmark extent, re-crops every frame to approximate the
VoxCeleb2 training-face composition, resizes to 96×96, center-crops to 88×88,
and normalizes with mean 0.421 and standard deviation 0.165. If the landmark
file is missing or contains no valid detections, it warns once and falls back
to whole-frame grayscale resize. Pass `--no_align_face` only for an explicit
A/B comparison; this changes the evaluation protocol and must be recorded.
([look2hear/datas/real_test_dataset.py](look2hear/datas/real_test_dataset.py),
[look2hear/datas/transform.py](look2hear/datas/transform.py))

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
| Lip-reading backbone (≈ 259 MB) | Video-branch init — **training only** | `video_pretrain/frcnn_128_512.backbone.pth.tar` |

The first four are evaluation-metric weights, fetched automatically. The lip-reading backbone is only needed to train from scratch and must be **downloaded manually** from Google Drive — get it from <https://drive.google.com/file/d/13-T3nBnf21-lMKrV_XbH6Lf4vK2xU7lS/view> and place it at `video_pretrain/frcnn_128_512.backbone.pth.tar`. `download_models.sh` checks for it and prints the link if missing.

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
# Enhance and score Track 2 dev using the verified local snapshot
ENROLL_CKPT=pretrained/enroll_dev.pt \
  bash run_eval_real.sh \
  "$TRACK2_MODEL_DIR" track2 both all 0,1,2,3 "" both

# Score already-enhanced dev wavs without loading the separation model
ENROLL_CKPT=pretrained/enroll_dev.pt \
  bash run_eval_real.sh "" track2 both all 0,1,2,3 "" eval

# Pipeline smoke test (no metric weights needed): use the dedicated script
DATA_ROOT="$DATA_ROOT" GPU=0 bash scripts/smoke_track2.sh
```

`run_eval_real.sh` positional args: `[CKPT] [TRACK] [SCENE] [METRICS] [GPUS] [LIMIT] [MODE]`.
Env overrides: `SPLIT` (default `dev`), `SAVE_DIR`, `ENROLL_CKPT`,
`DATA_ROOT`, `CONF_DIR`. When `CONF_DIR` is unset, the wrapper selects the
tracked Track 1 or Track 2 YAML from `TRACK`.
Unlike `eval_real.py`, whose `--mode` default is `both`, the wrapper defaults
its seventh positional argument to `eval`. Pass `"" both` after the GPU list
when no limit is desired and enhancement should run.

The wrapper also forces Hugging Face, Transformers, and ModelScope offline
mode. `download_models.sh` primes the evaluation-metric models but does not
currently download the gated AVSE baseline checkpoint. Prefer the verified
local snapshot directory, or prime/copy a verified Hugging Face cache before
passing a repo id.

Calling `eval_real.py` directly exposes the full flag set:

```bash
python eval_real.py \
  --conf_dir configs/track2_av_convtasnet.yml \
  --ckpt JusperLee/Real-World-AVSE-Baseline-Track2 \
  --data_root "$DATA_ROOT" \
  --track track2 --scene both --split dev --metrics all \
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
| `--ckpt` / `--conf_dir` | HF repo id, local `config.json` + `model.safetensors` directory, Lightning checkpoint, or serialized `.pth`; the YAML supplies sample rate/face size and Lightning rebuild information |
| `--limit` | cap clips per (track, scene) for smoke tests |

### 6.6 Outputs

Three CSVs are written next to `--out_csv` (default `<exp_dir>/results/real_metrics.csv`):

- `real_metrics.csv` — per-item rows (key, track, scene, speaker_id, all metrics, ref/hyp text).
- `real_metrics_<track>_<scene>.csv` — per-group per-item rows.
- `real_metrics_summary.csv` — mean per metric, overall and per (track, scene) group.

### 6.7 Submission format

Submit your enhanced audio in the **same directory layout that `--mode enhance` produces** — mirroring the corpus tree, one enhanced wav per target speaker:

```
<submission>/track{1,2}/<split>/{mix,remix}/<clip_id>/
├── s1.wav        # enhanced estimate for target speaker s1
└── s2.wav        # enhanced estimate for target speaker s2
```

For example, `track2/test/mix/000356/s1.wav` is your estimate of speaker **s1** in clip `000356`, produced from that speaker's lip video. Requirements:

- **One `s1.wav` and one `s2.wav` per clip directory**, for every clip in the evaluated split (both `mix` and `remix` scenes).
- **16 kHz, mono, WAV.** Float (`PCM_F`) or 16-bit PCM are both accepted; the baseline writes 32-bit float so re-scoring reproduces the in-memory numbers exactly.
- **Keep the `clip_id` directory names and the `s1`/`s2` tags unchanged** — scoring matches files to references and speaker labels by this path. Length should match the input mixture.

The simplest way to generate a correctly-structured submission is to run the baseline (or your model in the same harness) with `--mode enhance --save_dir <submission>`; the resulting `enhanced_out/`-style tree *is* the submission. The current `eval_real.py` enumerates clips from `mix_manifest.json` and `remix_manifest.json` for every split. If the participant test handout does not include those manifests, a separate organizer-provided manifest or a reviewed directory-enumeration change is required before this harness can generate the full test submission.

### 6.8 Baseline results (dev)

Reference scores from the released checkpoints on the **dev** split (`--split dev --metrics all`). SI-SDR / PESQ / STOI are reference-based, so they are reported on `remix` only (the `mix` scene has no clean ground truth); each track's `overall` row therefore carries the same reference-based values as its `remix` row. Speaker similarity (`spk_sim`) uses the dev voiceprints; CER is from Fun-ASR-Nano. The current repository defaults to landmark alignment; the exact checkpoint revision, metric-cache revisions, command, and alignment setting used to produce the table still need to be attached before the team treats it as a reproducible reference.

**Track 1**

| scope | n | SI-SDR | PESQ | STOI | UTMOS | DNSMOS p808 | DNSMOS sig | DNSMOS bak | DNSMOS ovr | CER | spk_sim |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **overall** | 4284 | −1.61 | 1.21 | 0.464 | 1.022 | 2.260 | 1.849 | 2.301 | 1.493 | 0.725 | 0.364 |
| track1 / mix | 2484 | — | — | — | 1.009 | 2.291 | 1.925 | 2.404 | 1.565 | 0.673 | 0.383 |
| track1 / remix | 1800 | −1.61 | 1.21 | 0.464 | 1.040 | 2.217 | 1.744 | 2.159 | 1.392 | 0.798 | 0.337 |

**Track 2**

| scope | n | SI-SDR | PESQ | STOI | UTMOS | DNSMOS p808 | DNSMOS sig | DNSMOS bak | DNSMOS ovr | CER | spk_sim |
|-------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **overall** | 5250 | −2.70 | 1.243 | 0.469 | 1.174 | 2.324 | 1.655 | 1.658 | 1.356 | 0.915 | 0.370 |
| track2 / mix | 3054 | — | — | — | 1.167 | 2.345 | 1.685 | 1.659 | 1.384 | 0.815 | 0.383 |
| track2 / remix | 2196 | −2.70 | 1.243 | 0.469 | 1.183 | 2.294 | 1.613 | 1.657 | 1.319 | 1.053 | 0.352 |

These are intentionally modest — the baseline highlights the real-world domain gap rather than a tuned system. CER is a fraction (lower is better); SI-SDR is in dB.

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
