# EXP-002: Reliability-gated visual fusion

## Summary

- **Experiment ID:** EXP-002
- **Name:** Reliability-gated visual fusion for Track 2 AV-ConvTasNet
- **Date:** 2026-07-07
- **Owner:** TODO
- **Status:** Completed
- **Purpose:** Test whether a lightweight learned gate can reduce over-reliance on degraded visual features while preserving useful lip guidance.
- **Parent baseline/experiment:** `baseline/track2-source-v1` (`a3cce6c5c362ae0ce79d59494c52b36995a44c49`)
- **Branch:** `exp/experiment-cross`
- **Commit SHA:** `9cbbcc3780babe793014cb43b31aaca6fa22182b`

## Background

Track 2 deliberately corrupts the target visual stream with occlusion, low resolution, frame freeze, dropped frames, and AV desynchronization. The baseline fusion module temporally aligns video features and directly concatenates them with audio features. This assumes that visual features are uniformly useful after alignment, which is risky when the visual stream is unreliable.

## Hypothesis

A per-time, per-channel visual reliability gate can learn to inject visual evidence only when it is useful. The expected behavior is smaller degradation under severe visual corruption and no major regression under milder corruption.

## Difference from baseline

Replace direct audio-video concatenation with `ReliabilityGatedFusion` in `look2hear/models/av_convtasnet.py`, selected by `fusion_type: "reliability_gate"`. The audio encoder, frozen video encoder, separator blocks, loss, optimizer, and data pipeline remain unchanged.

## Module architecture

The reliability-gated module keeps the AV-ConvTasNet separator unchanged and only replaces the audio-video fusion block:

1. Audio features `a` and video features `v` are temporally aligned by interpolating `v` to the audio feature length.
2. `audio_proj` maps audio channels to the separator fusion width with a `1x1` convolution.
3. `video_proj` maps video channels to the same width with a `1x1` convolution.
4. The gate input is `[audio_proj, video_proj, abs(audio_proj - video_proj)]`.
5. A lightweight gate predicts a per-time, per-channel visual reliability mask:

```text
Conv1d(3 * out_channels -> fusion_gate_hidden)
PReLU
Conv1d(fusion_gate_hidden -> out_channels)
Sigmoid
```

6. The fused feature is computed as:

```text
fused = audio_proj + visual_gate * video_proj
output = out_proj(fused)
```

The residual audio path was intended to protect the model when visual evidence is degraded, while `abs(audio_proj - video_proj)` gives the gate a cheap mismatch cue for occlusion, frame freeze, low resolution, dropped frames, or AV desynchronization.

## Modified files

| File | Change | Reason |
|---|---|---|
| `look2hear/models/av_convtasnet.py` | Add reliability-gated fusion module and config switch | Learn visual trust under Track 2 degradation |
| `configs/track2_av_convtasnet_reliability_gate.yml` | Dedicated Track 2 config | Keep baseline config untouched |
| `EXPERIMENTS.md` | Add planned experiment row | Track provenance |
| `docs/experiments/EXP-002-reliability-gated-fusion.md` | Planned protocol | Record hypothesis and TODOs |

## Configuration

- **Configuration file:** `configs/track2_av_convtasnet_reliability_gate.yml`
- **Configuration snapshot/hash:** `sha256:6a7a4dec3a7e644f690906a778972b24d6d0df6aacef9bffcaa77b16a96209a1`
- **Random seed:** TODO
- **Batch size:** 8
- **DataLoader workers:** 0 for the restart after a worker-side collate failure; this favors stability over throughput.
- **Learning rate/scheduler:** Adam lr 0.0001, ReduceLROnPlateau
- **Epochs or stopping rule:** 500 epochs with early stopping patience 20
- **Other changed parameters:** `fusion_type: reliability_gate`, `fusion_gate_hidden: 128`
- **Warm-start checkpoint:** `/home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth`
- **Warm-start checksum:** `sha256:bc0980c00646c6fab34d0ee10a9eda54af772ea3a2e94b19150cd4b230e03576`
- **Warm-start policy:** load matching official Track 2 baseline weights with `strict=False`; when the checkpoint contains the old `av_model.concat.conv1d` and the config selects `reliability_gate`, migrate the old concat projection into `audio_proj` and `video_proj`, initialize `out_proj` as identity, and initialize the gate near 1. The frozen video encoder weights come from the checkpoint, so `videonet_config.pretrain` is `null`.

## Data

- **Dataset/source:** Processed Chinese Lips Track 2 training data, TODO verify source provenance in `/home/avse/data_pipeline`
- **Version/date:** TODO
- **License/status:** TODO / pending confirmation
- **Training data path:** `/home/avse/processed_Chineselips`
- **Challenge Track 2 dev contents:** `/home/avse/data/track2_dev`
- **Tool-compatible evaluation `DATA_ROOT`:** `/home/avse/workspace-goodparts/Experiments/track2_dev_uploaded_wrapper/data_root`
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
- **GPU allocation:** `[7]` for the next formal run attempt. Earlier `[3, 7]` DDP attempts conflicted with another active training process on GPU 3 and left an orphan GPU 7 rank after OOM.
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
- Environment check passed with 16 checks, 0 warnings, and 0 failures.
- Track 2 dev wrapper path resolved and official manifest counts matched, but the setup checker reported missing `s1.pkl/s2.pkl` files for 483 dev clips; this does not block training on `/home/avse/processed_Chineselips`, but it must be accounted for before formal dev evaluation.
- First formal attempt failed at epoch 0, batch 585/944 with PyTorch DataLoader worker error `Trying to resize storage that is not resizable`. Added an explicit Track 2 collate function that copies numpy mouth arrays into contiguous tensors before stacking, and reduced EXP-002 `num_workers` to 0 for the restart.
- Second formal attempt failed at epoch 0, batch 559/944 because a batch mixed 50-frame and 45-frame mouth tensors. Added Track 2 mouth-length normalization: sequences longer than the configured video length are truncated and shorter sequences are padded by repeating the final frame. Verified 11 train batches at batch size 8 all produced mouth tensors shaped `(8, 50, 88, 88)`.
- Third formal attempt failed immediately with CUDA OOM on GPU 3 because another active `track1_av_convtasnet.yml` training process was already using that GPU. The orphan EXP-002 rank on GPU 7 was stopped, and the next restart uses single GPU `[7]` to avoid DDP rank placement on the busy GPU.
- Fourth formal attempt ran on GPU 7 through epoch 1 and part of epoch 2 before manual interrupt. Validation loss worsened from `1.45061` at epoch 0 to `2.43684` at epoch 1. This run used randomly initialized reliability-gate fusion while loading the rest of the model from the baseline checkpoint, so it is treated as a failed partial run rather than a completed result.
- Conservative-init training produced the evaluated checkpoint:
  `/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-ReliabilityGate-ConservativeInit/best_model.pth`.
- Checkpoint metadata showed `best_model_score=-5.1524` at epoch 17; `last.ckpt` reached epoch 37 without improving the best validation score.
- Full Track 2 dev enhancement completed with `--no_align_face`, producing 5250 enhanced wavs.
- Enhanced wav validation passed: all 5250 files were readable, mono, 16 kHz, finite, and nonzero. 128 files differed from the input length by 4 to 16 frames; the objective metric aligns to the shorter length, so this was recorded but not treated as blocking.
- Full metrics completed for objective, DNSMOS, speaker similarity, ASR/CER, and UTMOS.

### Full Track 2 dev metrics

Artifacts:

- Enhanced wavs: `/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-ReliabilityGate-ConservativeInit-dev-eval/full/enhanced`
- Metrics: `/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-ReliabilityGate-ConservativeInit-dev-eval/metrics`
- Status summary: `/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-ReliabilityGate-ConservativeInit-dev-eval/metrics/eval_status.md`

| scope | n | si_sdr | pesq | stoi | utmos | dnsmos_p808 | dnsmos_sig | dnsmos_bak | dnsmos_ovr | cer | spk_sim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 5250 | -4.7859 | 1.1721 | 0.3961 | 1.0845 | 2.3683 | 1.7341 | 1.8675 | 1.4065 | 1.0099 | 0.3524 |
| track2/mix | 3054 |  |  |  | 1.0745 | 2.3592 | 1.7221 | 1.8619 | 1.4122 | 0.9464 | 0.3762 |
| track2/remix | 2196 | -4.7859 | 1.1721 | 0.3961 | 1.0984 | 2.3811 | 1.7509 | 1.8752 | 1.3985 | 1.0982 | 0.3194 |

### Comparison with archived baseline

Baseline metric root: `/home/avse/experiments/track2_dev_official_baseline/metrics`.

Positive delta is better for SI-SDR, PESQ, STOI, UTMOS, DNSMOS, and speaker similarity. Negative delta is better for CER.

| scope | metric | current | baseline | delta |
| --- | --- | ---: | ---: | ---: |
| track2/remix | si_sdr | -4.7859 | -2.8480 | -1.9379 |
| track2/remix | pesq | 1.1721 | 1.2561 | -0.0840 |
| track2/remix | stoi | 0.3961 | 0.4698 | -0.0737 |
| overall | utmos | 1.0845 | 1.1758 | -0.0913 |
| track2/mix | utmos | 1.0745 | 1.1842 | -0.1097 |
| track2/remix | utmos | 1.0984 | 1.1640 | -0.0656 |
| overall | cer | 1.0099 | 0.8812 | +0.1287 |
| track2/mix | cer | 0.9464 | 0.8922 | +0.0542 |
| track2/remix | cer | 1.0982 | 0.8658 | +0.2324 |
| overall | spk_sim | 0.3524 | 0.3702 | -0.0178 |
| track2/mix | spk_sim | 0.3762 | 0.3842 | -0.0080 |
| track2/remix | spk_sim | 0.3194 | 0.3506 | -0.0312 |
| overall | dnsmos_p808 | 2.3683 | 2.3703 | -0.0020 |
| overall | dnsmos_sig | 1.7341 | 1.7448 | -0.0107 |
| overall | dnsmos_bak | 1.8675 | 1.7035 | +0.1640 |
| overall | dnsmos_ovr | 1.4065 | 1.3955 | +0.0110 |
| track2/mix | dnsmos_p808 | 2.3592 | 2.4023 | -0.0431 |
| track2/mix | dnsmos_sig | 1.7221 | 1.8188 | -0.0967 |
| track2/mix | dnsmos_bak | 1.8619 | 1.7760 | +0.0859 |
| track2/mix | dnsmos_ovr | 1.4122 | 1.4486 | -0.0364 |
| track2/remix | dnsmos_p808 | 2.3811 | 2.3257 | +0.0554 |
| track2/remix | dnsmos_sig | 1.7509 | 1.6420 | +0.1089 |
| track2/remix | dnsmos_bak | 1.8752 | 1.6027 | +0.2725 |
| track2/remix | dnsmos_ovr | 1.3985 | 1.3218 | +0.0767 |

### Analysis

The hypothesis is not supported by this run. The fine-tuned reliability-gate checkpoint is worse than the archived baseline on the primary target-recovery and usability metrics:

- Reference metrics degrade on `track2/remix`: SI-SDR drops by 1.94 dB, PESQ drops by 0.0840, and STOI drops by 0.0737.
- CER worsens overall from 0.8812 to 1.0099, with the largest regression on `track2/remix`.
- Speaker similarity drops overall from 0.3702 to 0.3524.
- UTMOS drops overall from 1.1758 to 1.0845.
- DNSMOS background and overall scores improve mainly on `remix`, but this appears to reflect stronger background suppression or smoothing rather than better target recovery.

The converted Chinese Lips dataset passed the available strict validation checks: 18031 pair records, expected split counts, no obvious speaker-overlap issue in the recorded report, and mixture reconstruction error within PCM16 tolerance. No direct evidence currently points to a gross conversion failure.

The more likely issue is the training and fusion behavior. Checkpoint inspection found the learned reliability gate still biased almost fully open: `av_model.concat.gate.2.bias` had mean absolute value about 6.0, which corresponds to a sigmoid value near 0.9975. In practice, the module did not learn to suppress unreliable visual evidence. The run therefore behaved closer to always-on visual fusion while also changing the projection path, which can explain degraded target recovery, CER, and speaker identity despite modest DNSMOS background gains.

Conclusion: completed negative experiment. Do not merge this checkpoint or claim it as an improvement. The next experiment should either add explicit visual-reliability supervision/regularization, start with a stricter audio-protective gate initialization, or run an audio-only/concat ablation on the same converted data to separate data effects from fusion effects.
