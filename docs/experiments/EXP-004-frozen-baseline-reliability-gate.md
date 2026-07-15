# EXP-004: Frozen-baseline reliability-gate fusion

## Summary

- **Experiment ID:** EXP-004
- **Name:** Frozen-baseline reliability-gate fusion for Track 2 AV-ConvTasNet
- **Date:** 2026-07-10
- **Owner:** TODO
- **Status:** Paused
- **Purpose:** Test whether reliability-gated visual fusion improves Track 2 robustness when the warm-started baseline is protected from drift.
- **Parent baseline/experiment:** `EXP-002` implementation on top of `baseline/track2-source-v1`
- **Branch:** `exp/reliability-gate-frozen-fusion`
- **Commit SHA:** TODO

## Background

EXP-002 trained the reliability-gate model with most AV-ConvTasNet parameters still trainable. The full Track 2 dev evaluation was a negative result: objective metrics, UTMOS, CER, and speaker similarity regressed relative to the archived baseline. Checkpoint inspection also showed the gate stayed almost fully open, so the model did not learn the intended visual-reliability behavior.

## Hypothesis

Freezing the warm-started baseline parameters and training only the new fusion module may preserve the baseline separator while letting the gate adapt visual injection. This should reduce the risk that full fine-tuning damages target-speaker recovery before the gate has learned useful reliability behavior.

## Difference from EXP-002

EXP-004 keeps the same `ReliabilityGatedFusion` architecture but changes the training policy:

- Freeze all model parameters after warm-start.
- Re-enable gradients only for parameters with names prefixed by `av_model.concat.`.
- Trainable tensors are limited to the reliability-gate fusion module:
  - `av_model.concat.audio_proj.*`
  - `av_model.concat.video_proj.*`
  - `av_model.concat.gate.*`
  - `av_model.concat.out_proj.*`

The lip-reading ResNet video encoder remains frozen as before.

## Module architecture

The baseline fusion block is:

```text
concat(audio_features, video_features) -> Conv1d -> fused_features
```

The reliability-gate fusion block is:

```text
audio_proj = Conv1d(audio_features)
video_proj = Conv1d(video_features)
gate_in = concat(audio_proj, video_proj, abs(audio_proj - video_proj))
visual_gate = Sigmoid(Conv1d(PReLU(Conv1d(gate_in))))
fused = audio_proj + visual_gate * video_proj
output = Conv1d(fused)
```

EXP-004 trains only this fusion block, not the upstream encoder or downstream separator.

## Modified files

| File | Change | Reason |
|---|---|---|
| `train.py` | Add optional `training.trainable_param_prefixes` freeze policy | Allow experiments that train only selected parameters |
| `configs/track2_av_convtasnet_reliability_gate_frozen_fusion.yml` | Dedicated frozen-fusion config | Keep EXP-002 config untouched |
| `EXPERIMENTS.md` | Add EXP-004 row | Track provenance |
| `docs/experiments/EXP-004-frozen-baseline-reliability-gate.md` | Planned protocol | Record hypothesis, command, and TODOs |

## Configuration

- **Configuration file:** `configs/track2_av_convtasnet_reliability_gate_frozen_fusion.yml`
- **Configuration snapshot/hash:** `sha256:2d8a015a5dc7a75cb241961648a1a79d863a7280f8b0c942e98e3ec1c510d710`
- **Random seed:** TODO / inherited script default unless confirmed
- **Batch size:** 8
- **DataLoader workers:** 0
- **Learning rate/scheduler:** Adam lr 0.0001, ReduceLROnPlateau
- **Epochs or stopping rule:** 500 epochs with early stopping patience 20
- **Trainable parameter policy:** `training.trainable_param_prefixes: ["av_model.concat."]`
- **Warm-start checkpoint:** `/home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth`
- **Warm-start policy:** migrate baseline concat projection into reliability-gate projections, initialize `out_proj` as identity, and initialize the gate near open as implemented in `train.py`.

## Data

- **Training data path:** `/home/avse/processed_Chineselips/tr`
- **Validation data path:** `/home/avse/processed_Chineselips/cv`
- **Test data path:** `/home/avse/processed_Chineselips/tt`
- **Challenge Track 2 dev contents:** `/home/avse/data/track2_dev`
- **Tool-compatible evaluation data root:** `/home/avse/workspace-goodparts/Experiments/track2_dev_uploaded_wrapper/data_root`
- **License/status:** TODO / pending confirmation
- **Split policy:** TODO / pending confirmation; previous validation report did not show an obvious conversion integrity failure.

## Environment

- **Training Python:** `/home/avse/avse_gpu_venv/bin/python`
- **Evaluation Python:** `/home/avse/avse_eval_py311/bin/python`
- **GPU allocation:** config requests `[4]`; GPU 7 was busy at training launch time.
- **Logger:** SwanLab disabled for this run because no API key was configured in the non-interactive environment.
- **Path conventions:** `docs/experiment-conventions.md`
- **CUDA/PyTorch inventory:** TODO / pending confirmation

## Commands

### Sanity check

```bash
/home/avse/avse_gpu_venv/bin/python -m compileall train.py look2hear/models/av_convtasnet.py

/home/avse/avse_gpu_venv/bin/python - <<'PY'
import yaml
import look2hear.models
from train import load_model_weights, apply_trainable_param_prefixes

conf = yaml.safe_load(open("configs/track2_av_convtasnet_reliability_gate_frozen_fusion.yml"))
video_cfg = conf.get("videonet", {}).get("videonet_config", {})
model = getattr(look2hear.models, conf["audionet"]["audionet_name"])(
    sample_rate=conf["datamodule"]["data_config"]["sample_rate"],
    video_relu_type=video_cfg.get("relu_type", "prelu"),
    video_pretrain=video_cfg.get("pretrain", None),
    **conf["audionet"]["audionet_config"],
)
load_model_weights(
    model,
    "/home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth",
)
apply_trainable_param_prefixes(model, conf["training"]["trainable_param_prefixes"])
print([name for name, p in model.named_parameters() if p.requires_grad])
PY
```

### Formal run

```bash
/home/avse/avse_gpu_venv/bin/python train.py \
  --conf_dir configs/track2_av_convtasnet_reliability_gate_frozen_fusion.yml \
  --warm_start /home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth
```

### Resume from pause

Training was intentionally stopped on 2026-07-11 18:22:32 +0800 to release GPU
4 for the official-dev domain adaptation experiment. This was not a failed run:
Lightning checkpoints were already present.

Resume command:

```bash
cd /home/avse/workspace-goodparts
/home/avse/avse_gpu_venv/bin/python train.py \
  --conf_dir configs/track2_av_convtasnet_reliability_gate_frozen_fusion.yml
```

Do not pass `--warm_start` when resuming this paused run. `train.py` will
auto-detect and resume from:

```text
/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-ReliabilityGate-FrozenFusion/last.ckpt
```

If a fresh restart is intended instead of a true resume, move or archive the
existing experiment directory first, then launch with the original warm-start
command.

## Results

Training started on 2026-07-10 in tmux session `exp004_frozen`.

- **Process command:** `train.py --conf_dir configs/track2_av_convtasnet_reliability_gate_frozen_fusion.yml --warm_start /home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth`
- **Log path:** `/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-ReliabilityGate-FrozenFusion/logs/train.tmux.log`
- **Initial verification:** training entered Epoch 0; Lightning reported 328 K trainable parameters and 24.9 M non-trainable parameters.
- **Trainable tensors:** 11 tensors under `av_model.concat.*`.
- **Paused status:** stopped on 2026-07-11 18:22:32 +0800 by terminating the
  active `train.py` process to free GPU 4 for the official-dev split
  adaptation experiment.
- **Resume artifact:** `/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-ReliabilityGate-FrozenFusion/last.ckpt`
  exists and matched the latest recorded checkpoint, `epoch=30.ckpt`, at the
  time of pause.

Do not mark completed until the formal command finishes, artifacts are checked, and full Track 2 dev metrics are recorded.
