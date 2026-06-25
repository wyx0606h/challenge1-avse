# Experiment Registry

This file is the team-level index of baseline reproductions and research
experiments. Detailed reports should be created from
`docs/experiment-template.md` and stored under `docs/experiments/` when the
first run is prepared.

Do not mark an experiment `Completed` until its formal command has finished,
artifacts have been checked, and complete metrics have been recorded.

## Status values

| Status | Meaning |
|---|---|
| Planned | Protocol exists, but the formal run has not started |
| Running | Formal execution is currently in progress |
| Failed | Execution ended unsuccessfully or produced invalid artifacts |
| Completed | Execution and artifact/metric verification both succeeded |

## Experiment index

| ID | Name | Status | Branch | Commit | Parent | Result summary | Detailed record |
|---|---|---|---|---|---|---|---|
| EXP-001 | Official Track 2 baseline reproduction | Planned | `exp/baseline-reproduction` | TODO | `baseline/track2-source-v1` | No team result yet | [`docs/experiments/EXP-001-baseline-reproduction.md`](docs/experiments/EXP-001-baseline-reproduction.md) |

`baseline/track2-source-v1` freezes the source code and collaboration workflow.
It is not an experiment result and does not claim reproduction of the official
metrics.

## Current blockers and TODOs

- TODO: challenge registration and `dev` access.
- TODO: confirm the purpose, contents, license, and storage location of all
  organizer-provided assets.
- TODO: identify legal training data and document its license.
- TODO: confirm training manifests and speaker-disjoint split policy.
- TODO: audit the target server:
  - GPU model, count, and VRAM;
  - NVIDIA driver, CUDA, Python, and PyTorch;
  - CPU, RAM, disk, shared storage, and job limits;
  - mixed RTX 4090/RTX 5090 DDP compatibility.
- TODO: confirm Hugging Face authorization for the selected baseline model.
- TODO: define external data, checkpoint, log, prediction, and submission roots.
- Source baseline commit: frozen by `baseline/track2-source-v1`.
- TODO: freeze the evaluated checkpoint identity and evaluation protocol,
  including face-alignment behavior and metric model versions.

## Required metadata for every experiment

Each experiment must include all fields below. Use `TODO` while planning and
replace it only with verified information.

### Identity

- Experiment ID
- Experiment name
- Date
- Owner
- Purpose
- Hypothesis
- Status

### Git provenance

- Branch
- Commit SHA
- Parent baseline/experiment
- Modified files
- Description of changes

### Configuration and data

- Configuration file and immutable copy/hash
- Data source and version
- Data path
- License status
- Train/validation/test split policy
- Random seed
- Complete run command

### Environment and hardware

- Host or scheduler job identifier
- GPU model and count
- NVIDIA driver and CUDA
- Python, PyTorch, torchaudio, torchvision, and Lightning
- CPU, RAM, disk, and relevant limits
- Environment export or lock-file path

### Artifacts and results

- Log path
- Checkpoint path and checksum
- Prediction/output path
- Metric output path
- Complete validation metrics
- Delta from the declared baseline
- Runtime and peak GPU memory
- Artifact completeness checks

### Interpretation

- Whether the run completed successfully
- Whether the result is worth retaining
- Failures and anomalies
- Conclusion
- Next step

## Baseline reproduction acceptance criteria

EXP-001 may become `Completed` only when:

1. the immutable source tag, exact Git commit, and unchanged baseline
   configuration are recorded;
2. the environment and hardware inventory are captured;
3. data access, counts, paths, and license/status are documented;
4. Hugging Face checkpoint identity or local checkpoint checksum is recorded;
5. the smoke test succeeds;
6. the expected full evaluation item count is verified;
7. output completeness and metric CSVs are checked;
8. random seed and face-alignment setting are recorded;
9. results are compared with the official reference under the same protocol;
10. logs, configuration, metrics, and external checkpoint paths are preserved.
