# Experiment Conventions

This document records the current shared local conventions for this checkout.
Update it when paths, environments, branch policy, or artifact-root policy
change. Do not duplicate these details across experiment reports unless a run
intentionally deviates from them.

## Repository and branches

- Repository: `/home/avse/workspace-goodparts`
- Current improvement branch for EXP-002 and EXP-003: `exp/experiment-cross`
- Immutable source baseline: `baseline/track2-source-v1`
- Reviewed stable branch: `main`

Documentation-only changes that describe shared server paths may be merged
across experiment branches after review. They must not change model, data,
loss, metric, training, inference, or submission behavior.

## Local data paths

Training data:

- Processed Chinese Lips Track 2 training root:
  `/home/avse/processed_Chineselips`
- Expected splits under that root: `tr`, `cv`, and `tt`
- Expected per-split manifests: `mix.json`, `s1.json`, and `s2.json`
- Dataset build and validation notes: `/home/avse/data_pipeline`

The dedicated EXP-002 and EXP-003 configs point directly at
`/home/avse/processed_Chineselips/{tr,cv,tt}`. If another experiment uses a
different data root, update only that experiment's copied config and record the
deviation in its report.

Challenge Track 2 dev evaluation data:

- Actual Track 2 dev contents root: `/home/avse/data/track2_dev`
- This directory directly contains `mix_manifest.json`, `remix_manifest.json`,
  `mix/`, and `remix/`.
- Existing tool-compatible `DATA_ROOT` wrapper:
  `/home/avse/experiments/track2_dev_official_baseline/data_root`
- In that wrapper, `track2/dev` is a symlink to
  `/home/avse/data/track2_dev`.

Use the wrapper path for tools that expect a corpus root containing
`track2/dev`, for example:

```bash
DATA_ROOT=/home/avse/experiments/track2_dev_official_baseline/data_root
python tools/check_track2_setup.py --data-root "$DATA_ROOT"
python eval_real.py --data_root "$DATA_ROOT" --track track2 --split dev
```

## Local environments and assets

Training/runtime Python:

- `/home/avse/avse_gpu_venv/bin/python`

Evaluation Python:

- `/home/avse/avse_eval_py311/bin/python`

Evaluation assets:

- Root: `/home/avse/avse-assets/evaluation`
- WeSpeaker checkpoint:
  `/home/avse/avse-assets/evaluation/wespeaker/cnceleb-resnet34-LM/model_5.pt`
- DNSMOS assets: `/home/avse/avse-assets/evaluation/dnsmos`
- FunASR model: `/home/avse/avse-assets/evaluation/funasr/Fun-ASR-Nano-2512`
- FunASR VAD model: `/home/avse/avse-assets/evaluation/funasr/fsmn-vad`
- UTMOSv2 assets: `/home/avse/avse-assets/evaluation/utmosv2`

Do not commit environment directories, model weights, datasets, enhanced audio,
raw logs, or metric artifacts.

## Artifact policy

The training script writes to repository-local `Experiments/` by default, which
is ignored by Git. For long runs, prefer an external artifact root under
`/home/avse/experiments/` and record exact log, checkpoint, prediction, and
metric paths in the experiment report.

Current known external evaluation workspace:

- `/home/avse/experiments/track2_dev_official_baseline`

The final shared artifact-root policy and storage quota are still TODO /
pending confirmation.

## Required preflight before experiments

Run and record:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD
git diff --stat
/home/avse/avse_gpu_venv/bin/python --version
/home/avse/avse_eval_py311/bin/python --version
nvidia-smi
```

Then verify:

```bash
/home/avse/avse_gpu_venv/bin/python tools/check_track2_setup.py --check-env
/home/avse/avse_gpu_venv/bin/python tools/check_track2_setup.py --check-training-assets
DATA_ROOT=/home/avse/experiments/track2_dev_official_baseline/data_root
/home/avse/avse_gpu_venv/bin/python tools/check_track2_setup.py --data-root "$DATA_ROOT"
```

Do not start expensive training until the branch, clean working tree, data
layout, environment, GPU allocation, run budget, and output paths are confirmed.
