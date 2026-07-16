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
| Paused | Formal execution was intentionally stopped with resumable artifacts retained |
| Failed | Execution ended unsuccessfully or produced invalid artifacts |
| Completed | Execution and artifact/metric verification both succeeded |

## Experiment index

| ID | Name | Status | Branch | Commit | Parent | Result summary | Detailed record |
|---|---|---|---|---|---|---|---|
| EXP-001 | Official Track 2 baseline reproduction | Planned | `exp/baseline-reproduction` | TODO | `baseline/track2-source-v1` | No team result yet | TODO |
| EXP-002 | Reliability-gated visual fusion | Completed | `exp/experiment-cross` | `9cbbcc3780babe793014cb43b31aaca6fa22182b` | `baseline/track2-source-v1` | Negative result: full Track 2 dev metrics are worse than archived baseline on SI-SDR, PESQ, STOI, UTMOS, CER, and speaker similarity; DNSMOS background improves only partially | `docs/experiments/EXP-002-reliability-gated-fusion.md` |
| EXP-003 | Cross-attention visual fusion | Planned | `exp/experiment-cross` | TODO | `baseline/track2-source-v1` | No result yet | `docs/experiments/EXP-003-cross-attention-fusion.md` |
| EXP-004 | Frozen-baseline reliability-gate fusion | Paused | `exp/reliability-gate-frozen-fusion` | TODO | `EXP-002` / `baseline/track2-source-v1` | Paused on 2026-07-11 to free GPU for dev-domain adaptation; `last.ckpt` retained at epoch 30 for resume | `docs/experiments/EXP-004-frozen-baseline-reliability-gate.md` |
| EXP-005 | Baseline fine-tune on dev-matched Chinese-LiPS v1 | Failed | current worktree | TODO | official Track 2 baseline checkpoint | Invalid full-dev metrics: checkpoint produced severe waveform gain explosions on official dev aligned visual inputs due to adapted internal visual-branch BN stats | `docs/experiments/EXP-005-baseline-chineselips-devmatched-v1.md` |
| EXP-006 | Baseline fine-tune on dev-matched Chinese-LiPS v1 with frozen internal visual branch BN | Completed | current worktree | `9cbbcc3780babe793014cb43b31aaca6fa22182b` | EXP-005 / official Track 2 baseline checkpoint | Valid positive result: freezing `av_model.video` prevents gain explosions; dev-matched v1 improves remix objective metrics, UTMOS, speaker similarity, and overall CER over the official baseline | `docs/experiments/EXP-006-baseline-chineselips-devmatched-freeze-avvideo-bn.md` |
| EXP-007 | ReliabilityGate fine-tune on dev-matched Chinese-LiPS v1 with frozen visual branch | Completed | current worktree | `9cbbcc3780babe793014cb43b31aaca6fa22182b` | EXP-006 / official Track 2 baseline checkpoint | Positive result versus the official baseline: improves 9/10 reported remix metrics and 5/7 reported mix metrics; only remix CER and mix DNSMOS SIG/BAK regress | `docs/experiments/EXP-007-reliabilitygate-chineselips-devmatched-freeze-avvideo.md` |
| EXP-008 | Hierarchical visual temporal fusion v1/v2 | Planned | `exp/hierarchical-visual-fusion` | `78f87bbaa4f996e18304e144677815fea398982a` | EXP-006 / `2aeb35a65f15fc0c215abc5c651f4f7579edd5d4` | v1 aggregates V-TCN layers 1/3/5 at the existing fusion point; v2 adds stage-matched reliability-gated residual injection; implementation/static config checks passed, server forward and formal results pending | `docs/experiments/EXP-008-hierarchical-visual-fusion.md` |
| EXP-009 | Pretrained visual representations v1/v2 | Planned | `exp/pretrained-visual-representations` | TODO | EXP-008 V1 implementation / `3c41486b29bee98aa20c20d5fab54875864b9cd3` | v1 replaces the legacy ResNet with a frozen AV-HuBERT visual frontend; v2 adds a gated contextual residual for local-articulation plus longer-context visual evidence; `degrade_prob=0.0` is locked for the first comparison | `docs/experiments/EXP-009-pretrained-visual-representations.md` |

`baseline/track2-source-v1` freezes the source code and collaboration workflow.
It is not an experiment result and does not claim reproduction of the official
metrics.

## Current blockers and TODOs

- TODO: challenge registration and `dev` access.
- TODO: confirm the purpose, contents, license, and storage location of all
  organizer-provided assets.
- Current local path conventions are recorded in
  `docs/experiment-conventions.md`, including:
  - training root: `/home/avse/processed_Chineselips`;
  - Track 2 dev contents: `/home/avse/data/track2_dev`;
  - tool-compatible `DATA_ROOT` wrapper:
    `/home/avse/experiments/track2_dev_official_baseline/data_root`;
  - evaluation assets: `/home/avse/avse-assets/evaluation`;
  - training Python: `/home/avse/avse_gpu_venv/bin/python`;
  - evaluation Python: `/home/avse/avse_eval_py311/bin/python`.
- TODO: identify legal training data and document its license/status.
- TODO: confirm training manifests and speaker-disjoint split policy.
- TODO: audit the target server:
  - GPU model, count, and VRAM;
  - NVIDIA driver, CUDA, Python, and PyTorch;
  - CPU, RAM, disk, shared storage, and job limits;
  - mixed RTX 4090/RTX 5090 DDP compatibility.
- TODO: confirm Hugging Face authorization for the selected baseline model.
- TODO: define final shared checkpoint, log, prediction, and submission roots
  and storage quota. Use external ignored storage such as `/home/avse/experiments/`
  until that policy is finalized.
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
