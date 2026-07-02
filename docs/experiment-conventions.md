# Experiment Conventions

These conventions describe the current local baseline-reproduction setup.
Experiment reports should reference this file instead of repeating the same
paths.

- Model repository: `/home/avse/workspace-goodparts`
- Data pipeline: `/home/avse/data_pipeline`
- Current CPU smoke environment: `/home/avse/data_pipeline/.venv`
- Current GPU evaluation environment: `/home/avse/avse_gpu_venv`
- Current smoke data: `/home/avse/data_pipeline/outputs/chinese_lips_baseline_smoke`
- Artifact root: `/home/avse/experiments/<experiment_name>`
- Stable branch: `main`
- Baseline reproduction and debugging branch: `exp/baseline-reproduction`
- Future `exp/*` branches may reuse `/home/avse/data_pipeline/.venv`; create a
  new environment only when dependencies conflict.
- Visual degradation is controlled online by the Dataset. Small-data overfit
  experiments use `degrade_prob=0`.

Do not write checkpoints, generated WAV files, raw logs, or large predictions
inside the Git repository.
