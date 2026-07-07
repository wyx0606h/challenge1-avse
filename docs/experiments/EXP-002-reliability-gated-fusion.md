# EXP-002: Reliability-gated visual fusion

## Summary

- **Experiment ID:** EXP-002
- **Name:** Reliability-gated visual fusion for Track 2 AV-ConvTasNet
- **Date:** 2026-07-07
- **Owner:** TODO
- **Status:** Planned
- **Purpose:** Test whether a lightweight learned gate can reduce over-reliance on degraded visual features while preserving useful lip guidance.
- **Parent baseline/experiment:** `baseline/track2-source-v1` (`a3cce6c5c362ae0ce79d59494c52b36995a44c49`)
- **Branch:** `exp/experiment-cross`
- **Commit SHA:** TODO

## Background

Track 2 deliberately corrupts the target visual stream with occlusion, low resolution, frame freeze, dropped frames, and AV desynchronization. The baseline fusion module temporally aligns video features and directly concatenates them with audio features. This assumes that visual features are uniformly useful after alignment, which is risky when the visual stream is unreliable.

## Hypothesis

A per-time, per-channel visual reliability gate can learn to inject visual evidence only when it is useful. The expected behavior is smaller degradation under severe visual corruption and no major regression under milder corruption.

## Difference from baseline

Replace direct audio-video concatenation with `ReliabilityGatedFusion` in `look2hear/models/av_convtasnet.py`, selected by `fusion_type: "reliability_gate"`. The audio encoder, frozen video encoder, separator blocks, loss, optimizer, and data pipeline remain unchanged.

## Modified files

| File | Change | Reason |
|---|---|---|
| `look2hear/models/av_convtasnet.py` | Add reliability-gated fusion module and config switch | Learn visual trust under Track 2 degradation |
| `configs/track2_av_convtasnet_reliability_gate.yml` | Dedicated Track 2 config | Keep baseline config untouched |
| `EXPERIMENTS.md` | Add planned experiment row | Track provenance |
| `docs/experiments/EXP-002-reliability-gated-fusion.md` | Planned protocol | Record hypothesis and TODOs |

## Configuration

- **Configuration file:** `configs/track2_av_convtasnet_reliability_gate.yml`
- **Configuration snapshot/hash:** `sha256:535e1d63b8679eb937510215a247b9dd599663affce95a001f287bc33aaf7773`
- **Random seed:** TODO
- **Batch size:** 8
- **Learning rate/scheduler:** Adam lr 0.001, ReduceLROnPlateau
- **Epochs or stopping rule:** 500 epochs with early stopping patience 20
- **Other changed parameters:** `fusion_type: reliability_gate`, `fusion_gate_hidden: 128`
- **Warm-start checkpoint:** `/home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth`
- **Warm-start checksum:** `sha256:bc0980c00646c6fab34d0ee10a9eda54af772ea3a2e94b19150cd4b230e03576`
- **Warm-start policy:** load matching official Track 2 baseline weights with `strict=False`; the new reliability gate is randomly initialized and the old concat projection is ignored. The frozen video encoder weights come from the checkpoint, so `videonet_config.pretrain` is `null`.

## Data

- **Dataset/source:** Processed Chinese Lips Track 2 training data, TODO verify source provenance in `/home/avse/data_pipeline`
- **Version/date:** TODO
- **License/status:** TODO / pending confirmation
- **Training data path:** `/home/avse/processed_Chineselips`
- **Challenge Track 2 dev contents:** `/home/avse/data/track2_dev`
- **Tool-compatible evaluation `DATA_ROOT`:** `/home/avse/experiments/track2_dev_official_baseline/data_root`
- **Evaluation assets:** `/home/avse/avse-assets/evaluation`
- **Manifest/checksum:** TODO
- **Train split:** `/home/avse/processed_Chineselips/tr` TODO verify counts/checksums
- **Validation split:** `/home/avse/processed_Chineselips/cv` TODO verify counts/checksums
- **Test split:** `/home/avse/processed_Chineselips/tt` TODO verify counts/checksums
- **Config path compatibility:** dedicated config now points directly to `/home/avse/processed_Chineselips/{tr,cv,tt}`.
- **Speaker-disjoint policy:** TODO / pending confirmation
- **Augmentation/degradation policy:** Track 2 online visual degradation, TODO verify exact run settings

## Environment

- **Training Python:** `/home/avse/avse_gpu_venv/bin/python`
- **Evaluation Python:** `/home/avse/avse_eval_py311/bin/python`
- **Path conventions:** `docs/experiment-conventions.md`
- **GPU allocation:** TODO / pending confirmation
- **CUDA/PyTorch inventory:** TODO / pending confirmation

## Commands

### Sanity check

```bash
/home/avse/avse_gpu_venv/bin/python -m compileall look2hear/models/av_convtasnet.py
/home/avse/avse_gpu_venv/bin/python - <<'PY'
import torch, yaml
import look2hear.models

conf = yaml.safe_load(open("configs/track2_av_convtasnet_reliability_gate.yml"))
model = getattr(look2hear.models, conf["audionet"]["audionet_name"])(
    sample_rate=conf["datamodule"]["data_config"]["sample_rate"],
    video_relu_type=conf["videonet"]["videonet_config"].get("relu_type", "prelu"),
    video_pretrain=None,
    **conf["audionet"]["audionet_config"],
)
ckpt = "/home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth"
state = torch.load(ckpt, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
print("missing", missing)
print("unexpected", unexpected)
PY
```

### Formal run

```bash
/home/avse/avse_gpu_venv/bin/python train.py \
  --conf_dir configs/track2_av_convtasnet_reliability_gate.yml \
  --warm_start /home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth
```

## Results

### Sanity check

Passed on 2026-07-07 before formal training:

- Model file compiled successfully.
- Dataset setup found 15,115 train mixtures, 1,954 validation target items, and 3,878 test target items.
- Warm-start from `official_serialized.pth` produced the expected compatibility report: 11 missing reliability-gate parameters and 2 unexpected old concat parameters.
- Frozen video encoder check: 0 trainable video-encoder tensors.
- One GPU forward pass on a batch shaped `(1, 32000)` audio and `(1, 50, 88, 88)` mouth frames produced finite output shaped `(1, 32000)`.

No result yet. Do not mark completed until the formal run, artifacts, and metrics are verified.
