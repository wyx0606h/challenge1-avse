# EXP-001: Official Track 2 baseline reproduction

## Summary

- **Experiment ID:** EXP-001
- **Name:** Official Track 2 baseline reproduction
- **Date:** 2026-06-25
- **Owner:** TODO / pending confirmation
- **Status:** Planned
- **Purpose:** Reproduce the released Track 2 checkpoint on the official dev
  split without changing model, data, metric, or submission behavior.
- **Parent baseline:** `baseline/track2-source-v1`
- **Parent commit:** `a3cce6c5c362ae0ce79d59494c52b36995a44c49`
- **Branch:** `exp/baseline-reproduction`
- **Run commit:** TODO — record the exact commit after protocol review

## Hypothesis

With the released
`JusperLee/Real-World-AVSE-Baseline-Track2` checkpoint, the official Track 2
dev data, default landmark alignment, and the documented metric models, the
pipeline will produce 5,250 complete target-speaker outputs and metrics close
to the published reference values under the same protocol.

## Difference from the source baseline

No model, loss, training-data, inference, or metric algorithm change is
planned. Repository-preparation changes on this branch clarify the executable
protocol and fix the evaluation wrapper so a clean checkout passes a tracked
`--conf_dir`.

Current protocol/documentation files changed before the formal run:

- `.env.example`
- `README.md`
- `docs/TRACK2_CHALLENGE_GUIDE_ZH.md`
- `run_eval_real.sh`
- this experiment record

## Checkpoint

- **Repository:** `JusperLee/Real-World-AVSE-Baseline-Track2`
- **Format:** `PyTorchModelHubMixin` (`config.json` +
  `model.safetensors`)
- **Access:** Completed after accepting the gated repository's contact-sharing
  condition.
- **Repository revision/commit:**
  `3777a458d13e8dc98877f69e542d6a8a7441835b`
- **`config.json`:** 279 bytes; SHA256
  `de1819318f7c3fb79914314475ab4b223973af9104d42600e533cdb557bcf5f8`
- **`model.safetensors`:** 100,176,860 bytes; SHA256
  `600d1632346e9b85e64f2842000f41d2cdb9f6b1afca8011c5decdc5ce9e78c1`
- **Storage:** Verified external directory supplied through
  `TRACK2_MODEL_DIR`; do not commit the files or a personal absolute path.
- **Video encoder:** Bundled in the checkpoint; the separate
  `video_pretrain/frcnn_128_512.backbone.pth.tar` file is not required for
  evaluation.

Do not record an access token in this document.

## Evaluation protocol

- **Configuration:** `configs/track2_av_convtasnet.yml`
- **Track/split/scenes:** Track 2 / dev / mix + remix
- **Expected clips:** 1,527 mix + 1,098 remix
- **Expected target items:** 5,250
- **Face processing:** Default `align_face=True`; use each clip's `.pkl`
  landmarks to compute a fixed crop, then resize 96, center-crop 88, and
  normalize with mean 0.421 / standard deviation 0.165.
- **Fallback:** Whole-frame grayscale resize when no usable landmarks exist;
  warnings and affected clips must be recorded.
- **Audio output:** 16 kHz mono, 32-bit float PCM, input-matched length.
- **Metrics:** SI-SDR/PESQ/STOI on remix; UTMOS, DNSMOS, CER, and speaker
  similarity on applicable items.
- **Random seed:** TODO — record all relevant library and data-pipeline seeds.

The `--no_align_face` option is not part of the baseline reproduction protocol.
Any no-alignment run is a separate A/B comparison.

## Data

- **Official Track 2 dev access:** TODO / pending confirmation
- **Dataset version/date:** TODO
- **License/status:** TODO
- **External data path:** TODO
- **Manifest checksums:** TODO
- **Speaker-disjoint policy:** Official dev is evaluation-only; do not mix it
  into training.

## Environment and hardware

Initial checkout-host observations on 2026-06-25:

- **Host:** `IASP`
- **Operating system:** Ubuntu 20.04-era Linux, kernel
  `5.15.0-67-generic`
- **CPU:** 2 × Intel Xeon Platinum 8352V, 144 logical CPUs total
- **RAM:** approximately 503 GiB
- **Repository filesystem:** approximately 213 GiB free at audit time and 99%
  used; not approved as the experiment artifact root
- **`/tmp`:** approximately 813 GiB free at audit time; retention policy TODO
- **FFmpeg:** 4.2.7
- **Python/Conda:** not available on the current shell PATH
- **GPU/driver:** `nvidia-smi` could not communicate with the NVIDIA driver
- **CUDA/PyTorch/Lightning:** TODO
- **Target server and scheduler/job limits:** TODO
- **Environment export/lock path:** TODO

## Commands

### Preflight

```bash
git status --short --branch
git rev-parse HEAD
python tools/check_track2_setup.py --check-env
python tools/check_track2_setup.py --data-root "$DATA_ROOT"
```

### Local checkpoint one-item-per-scene smoke test

```bash
DATA_ROOT=/external/path/to/Real-World-AVSE \
TRACK2_MODEL_DIR=/external/path/to/Real-World-AVSE-Baseline-Track2 \
python eval_real.py \
  --conf_dir configs/track2_av_convtasnet.yml \
  --ckpt "$TRACK2_MODEL_DIR" \
  --data_root "$DATA_ROOT" \
  --track track2 --scene both --split dev \
  --metrics none --mode enhance \
  --save_dir /external/path/to/outputs/EXP-001/smoke \
  --gpus 0 --limit 1
```

### Formal four-GPU enhance and score

`run_eval_real.sh` forces offline mode, so use the same verified Hugging Face
cache that was primed during the smoke test.

```bash
DATA_ROOT=/external/path/to/Real-World-AVSE \
TRACK2_MODEL_DIR=/external/path/to/Real-World-AVSE-Baseline-Track2 \
SAVE_DIR=/external/path/to/outputs/EXP-001/enhanced \
ENROLL_CKPT=/external/path/to/enroll_dev.pt \
bash run_eval_real.sh \
  "$TRACK2_MODEL_DIR" \
  track2 both all 0,1,2,3 "" both
```

Replace every placeholder path with the approved external path before running.

## Artifacts and verification

- **Log path:** TODO
- **Checkpoint snapshot path:** runtime `TRACK2_MODEL_DIR` on external storage;
  approved server mount TODO
- **Checkpoint checksum:** recorded in the Checkpoint section above
- **Enhanced output path:** TODO
- **Metric CSV path:** TODO
- **Saved configuration/hash:** TODO
- **Expected outputs:** 5,250
- **Produced outputs:** TODO
- **Missing/duplicate outputs:** TODO
- **NaN/Inf check:** TODO
- **Audio format/length check:** TODO
- **Runtime and peak GPU memory:** TODO

## Reference comparison

The scores currently printed in `README.md` are organizer/reference values, not
team reproduction results. Before comparison, record:

- exact repository commit and Hugging Face checkpoint revision;
- alignment setting and fallback count;
- metric package/model/cache revisions;
- enrollment checkpoint identity and checksum;
- complete command and environment snapshot.

## Current blockers

- Confirm the experiment owner.
- Confirm official challenge registration and Track 2 dev access.
- Confirm the dataset version, license/status, and external path.
- Restore or select a Python/Conda environment with working CUDA.
- Resolve why `nvidia-smi` cannot communicate with the driver.
- Select external roots for caches, enhanced audio, logs, metrics, and
  enrollment artifacts.
- Confirm the four GPUs and mixed RTX 4090/RTX 5090 compatibility.

## Result and conclusion

No run result is available. Keep `Status: Planned` until preflight and resource
requirements are satisfied. Do not mark this experiment `Completed` until the
formal command, artifact checks, complete metrics, and reference comparison
have all been verified.
