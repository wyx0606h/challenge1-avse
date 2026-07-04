# Experiment Conventions

These conventions describe the current local baseline-reproduction setup.
Experiment reports should reference this file instead of repeating the same
paths.

- Model repository: `/home/avse/workspace-goodparts`
- Data pipeline: `/home/avse/data_pipeline`
- Current CPU smoke environment: `/home/avse/data_pipeline/.venv`
- Legacy GPU environment for enhancement and Python 3.8-compatible metrics:
  `/home/avse/avse_gpu_venv`
- Default Track 2 dev evaluation environment for UTMOS/CER and future metric
  reruns: `/home/avse/avse_eval_py311`
- Current smoke data: `/home/avse/data_pipeline/outputs/chinese_lips_baseline_smoke`
- Artifact root: `/home/avse/experiments/<experiment_name>`
- Stable branch: `main`
- Baseline reproduction and debugging branch: `exp/baseline-reproduction`
- Future `exp/*` branches may reuse `/home/avse/data_pipeline/.venv`; create a
  new environment only when dependencies conflict.
- For Track 2 baseline evaluation, prefer `/home/avse/avse_eval_py311` unless
  a task explicitly needs the older Python 3.8 environment. UTMOSv2 and the
  bundled Fun-ASR/Qwen3 path require the Python 3.11 environment.
- Visual degradation is controlled online by the Dataset. Small-data overfit
  experiments use `degrade_prob=0`.

Do not write checkpoints, generated WAV files, raw logs, or large predictions
inside the Git repository.
