# AGENTS.md

This file defines the working rules for Codex, other automated coding agents,
and human contributors in this repository.

## Project objective

The project has two ordered goals:

1. reproduce the official Real-World AVSE Challenge Track 2 baseline without
   changing its algorithmic behavior;
2. develop and evaluate research improvements, especially visual-reliability
   aware fusion, only after the baseline is reproducible.

Current phase: **repository organization and baseline preparation**.

The team has not yet claimed a baseline reproduction. Challenge `dev` access,
training data, licenses, final server paths, and the full server hardware
inventory are **TODO / pending confirmation**.

The repository source baseline is frozen by the immutable tag
`baseline/track2-source-v1`. This tag records source code and collaboration
workflow only; it does not claim that official metrics have been reproduced.

## Repository map

- `look2hear/`: model, data, loss, metric, system, and utility code.
- `configs/`: tracked baseline and experiment configurations.
- `DataPreProcess/`: manifest-building tools; generated manifests are ignored.
- `Fun-ASR/`: vendored code used by the ASR/CER evaluation path.
- `tools/`: repository, environment, and data validation tools.
- `scripts/`: small reproducibility and smoke-test scripts.
- `docs/`: team guides and experiment templates.
- `train.py`: training entry point.
- `test.py`: training-corpus test-split evaluation.
- `eval_real.py`: challenge inference, evaluation, and submission generation.
- `build_enrollment.py`: speaker-enrollment embedding generation.
- `Experiments/`: generated training outputs; ignored by Git.

Do not create a lowercase root `experiments/` directory. It is easily confused
with the generated `Experiments/` directory on case-insensitive systems. Store
small tracked experiment reports under `docs/experiments/` when needed.

## Baseline protection

- The official upstream reference commit is documented in `UPSTREAM.md`.
- `main` is the reviewed stable branch. Do not edit it directly.
- `baseline/track2-source-v1` is the immutable source baseline. Never move,
  delete, or recreate it at another commit.
- `codex/bootstrap-track2` is archival history for the one-time import. Do not
  use it as a rolling experiment branch.
- Do not overwrite or silently reinterpret the official baseline.
- Baseline reproduction and research changes must use separate branches.
- Create new experiments from `baseline/track2-source-v1` or the matching
  reviewed `main` commit.
- Every experiment must identify the baseline commit it starts from.
- A documentation cleanup must not change model, loss, data, metric, training,
  inference, or submission behavior.
- Do not report an experiment as completed until its command finished and the
  expected artifacts and metrics were verified.

Recommended branches:

```text
exp/baseline-reproduction
exp/reliability-gating
exp/audio-only-fallback
exp/consistency-loss
exp/av-synchronization
```

One branch should address one coherent objective. Unverified experiments must
not be merged into the stable branch. A successful experiment must be fully
recorded and reviewed before it may be considered for `main`; merging it never
changes the immutable baseline tag.

## What may be modified

Normal experiment branches may modify:

- a dedicated copied configuration in `configs/`;
- the model/data/loss code required by the experiment;
- focused tests or validation tools;
- `EXPERIMENTS.md` and the experiment's report;
- supporting documentation.

Treat the following as protected:

- official baseline configurations;
- baseline model/evaluation behavior;
- upstream history and `UPSTREAM.md`;
- metric definitions and data splits;
- existing experiment records belonging to another contributor.

If a protected file must change, explain why in the experiment protocol and
keep the change isolated and reviewable.

## Code, data, weights, and outputs

The Git repository contains only code, configurations, documentation, and
necessary small metadata.

Never commit:

- challenge or third-party datasets;
- organizer test assets;
- pretrained weights or trained checkpoints;
- enhanced audio, submissions, or large predictions;
- raw training logs and caches;
- `.env`, tokens, API keys, passwords, or private keys;
- personal absolute server paths.

Use external storage for data and run artifacts. Record artifact paths in the
experiment report, but do not add the artifacts themselves to Git.

## Before modifying code

Run and record:

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD
git diff --stat
```

Then:

1. confirm that the branch is not `main`;
2. confirm the branch starts from `baseline/track2-source-v1` or the intended
   reviewed `main` commit;
3. read the relevant configuration and entry point;
4. read `README.md`, `EXPERIMENTS.md`, and the experiment template;
5. identify the exact baseline commit and hypothesis;
6. identify data, environment, and hardware unknowns;
7. create or update the experiment record with `Status: Planned`;
8. avoid touching unrelated user changes.

If data, environment, hardware, license, account, or metric details are
unclear, ask the user or write `TODO / pending confirmation`. Never invent a
path, result, credential, dataset version, or completed run.

## Required experiment workflow

```text
Clone repository
→ create a personal or experiment branch
→ configure environment
→ prepare data
→ run the unchanged baseline
→ save baseline metrics and logs
→ commit the baseline reproduction node
→ create a new experiment branch
→ make one clearly scoped change
→ inspect git diff
→ run a small sanity check
→ run the formal experiment
→ save logs, configuration, metrics, and checkpoint path
→ update the experiment record
→ commit code
→ push the remote branch
→ consider merging only after validation and review
```

## Experiment naming and metadata

Use monotonically increasing IDs:

```text
EXP-001-baseline-reproduction
EXP-002-reliability-gating
EXP-003-audio-only-fallback
```

Each experiment must record:

- ID, name, date, purpose, hypothesis, and status;
- branch, commit, parent baseline, and modified files;
- complete configuration and command;
- data source, version, path, license status, and split policy;
- random seed;
- GPU model/count and environment versions;
- logs, checkpoint, predictions, and metric paths;
- complete validation metrics and baseline delta;
- failures, anomalies, conclusion, and next step.

Allowed status values:

```text
Planned
Running
Failed
Completed
```

`Completed` means the formal command finished, artifacts were checked, and
metrics were recorded. A crash, partial run, or unverified output is not
completed.

## Checks after modifying code

At minimum:

```bash
git status --short
git diff --check
git diff --stat
git diff
```

Then run checks proportional to the change:

- documentation-only: inspect rendered Markdown, links, and `git diff --check`;
- Python-only static change: compile/import the touched module where feasible;
- configuration/data-loader change: run the setup checker and a tiny fixture;
- inference change: run a limited smoke test before formal evaluation;
- training change: run a short sanity batch before a formal run.

Do not run expensive training unless the user has approved it and the data,
environment, resource budget, and output paths are confirmed.

## Git safety

- Never automatically merge into `main`.
- Never force push.
- Never rewrite shared history without explicit approval.
- Do not delete another contributor's experiment results.
- Do not use `git add -f` to bypass data or weight protections.
- Inspect `git diff` before staging and again before committing.
- Use clear commit messages, for example:

```text
docs: add experiment workflow
chore: organize repository structure
exp: reproduce official baseline
feat: add reliability gating
fix: correct validation metric calculation
```

## Secrets and Hugging Face

There is no project variable named `HF_ACCESS`. It is a status label, not a
credential. Use standard Hugging Face authentication such as
`huggingface-cli login` or a secret-provided `HF_TOKEN`.

Never write a real token to tracked files. `.env.example` may contain empty
variable names and documentation only. The project does not automatically load
`.env`; users must explicitly export variables or use their job scheduler's
secret mechanism.

## Research integrity

- Never fabricate metrics, logs, completed runs, ablations, or leaderboard
  results.
- Never label an organizer/reference score as a team reproduction result.
- Never use challenge test results as if they were local ground truth.
- Record negative and failed experiments instead of deleting them.
- Claims must be traceable to a commit, configuration, data version, and
  verified result artifact.
