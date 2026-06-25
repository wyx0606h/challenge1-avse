# EXP-XXX: Experiment name

Copy this file for each experiment. Recommended destination:

```text
docs/experiments/EXP-XXX-short-name.md
```

Do not overwrite `TODO` with guesses. Do not set `Status: Completed` until the
formal run and artifact verification have both succeeded.

## Summary

- **Experiment ID:** EXP-XXX
- **Name:** TODO
- **Date:** TODO
- **Owner:** TODO
- **Status:** Planned
- **Purpose:** TODO
- **Parent baseline/experiment:** TODO
- **Branch:** TODO
- **Commit SHA:** TODO

Allowed status values: `Planned`, `Running`, `Failed`, `Completed`.

## Background

TODO: Describe the problem, prior evidence, and why this experiment is needed.

## Hypothesis

TODO: State a falsifiable hypothesis and the expected metric behavior.

## Difference from baseline

TODO: Describe exactly one coherent change. List anything intentionally kept
identical to the baseline.

## Modified files

| File | Change | Reason |
|---|---|---|
| TODO | TODO | TODO |

## Key code changes

TODO: Summarize the implementation. Link to relevant lines or commit after the
code exists.

## Configuration

- **Configuration file:** TODO
- **Configuration snapshot/hash:** TODO
- **Random seed:** TODO
- **Batch size:** TODO
- **Learning rate/scheduler:** TODO
- **Epochs or stopping rule:** TODO
- **Other changed parameters:** TODO

Include the complete configuration or a path to an immutable copy:

```yaml
# TODO
```

## Data

- **Dataset/source:** TODO
- **Version/date:** TODO
- **License/status:** TODO
- **Data path:** TODO
- **Manifest/checksum:** TODO
- **Train split:** TODO
- **Validation split:** TODO
- **Test split:** TODO
- **Speaker-disjoint policy:** TODO
- **Augmentation/degradation policy:** TODO

Do not include credentials, private download links, organizer test contents, or
data files in this document.

## Environment

- **Host/job ID:** TODO
- **Operating system:** TODO
- **GPU model/count:** TODO
- **GPU VRAM:** TODO
- **NVIDIA driver:** TODO
- **CUDA:** TODO
- **Python:** TODO
- **PyTorch:** TODO
- **torchaudio/torchvision:** TODO
- **PyTorch Lightning:** TODO
- **CPU/RAM:** TODO
- **Disk/storage:** TODO
- **Environment export path:** TODO

Suggested capture commands:

```bash
git rev-parse HEAD
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
python -m pip freeze
```

## Commands

### Preflight

```bash
git status --short --branch
python tools/check_track2_setup.py --check-env
# TODO: add data/training-asset checks applicable to this experiment
```

### Sanity check

```bash
# TODO
```

### Formal run

```bash
# TODO: exact copy-pasteable command
```

### Evaluation

```bash
# TODO
```

## Runtime and resource usage

- **Start/end time:** TODO
- **Wall-clock time:** TODO
- **GPU-hours:** TODO
- **Peak GPU memory:** TODO
- **CPU/RAM observations:** TODO
- **Interrupted/restarted:** TODO

## Artifacts

- **Log path:** TODO
- **Checkpoint path:** TODO
- **Checkpoint checksum:** TODO
- **Prediction path:** TODO
- **Metric CSV/path:** TODO
- **Saved configuration path:** TODO

These paths should point to external storage or ignored directories. Do not
commit large artifacts.

## Results

### Artifact validation

- Expected samples/items: TODO
- Produced samples/items: TODO
- Missing/duplicate outputs: TODO
- NaN/Inf check: TODO
- Audio format/length check: TODO

### Metrics

| Scope | Metric | Baseline | Experiment | Delta | Direction | Verified |
|---|---|---:|---:|---:|---|---|
| TODO | TODO | TODO | TODO | TODO | Higher/Lower is better | No |

No result is available while this section contains only `TODO`.

## Failures and anomalies

TODO: Record crashes, invalid samples, divergence, unexpected metric changes,
restarts, and exploratory changes made after the protocol was written.

## Conclusion

TODO: State whether the hypothesis was supported and why. Separate evidence
from interpretation.

## Retention decision

- **Worth retaining:** TODO — Yes / No / Inconclusive
- **Reason:** TODO
- **Candidate for merge:** TODO — Yes / No

## Next step

TODO
