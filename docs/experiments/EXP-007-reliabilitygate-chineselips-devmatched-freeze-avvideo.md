# EXP-007: ReliabilityGate fine-tune on dev-matched Chinese-LiPS v1

## Summary

- **Experiment ID:** EXP-007
- **Name:** ReliabilityGate AV-ConvTasNet on Chinese-LiPS Track2-dev-matched v1
  with frozen `av_model.video`
- **Status:** Completed
- **Purpose:** Move from data-only validation to a module-improvement test.
  EXP-006 showed that dev-matched ChineseLips v1 plus frozen visual branch is a
  usable baseline. EXP-007 keeps that data/training protection and changes only
  the audio-video fusion module from direct concat to ReliabilityGate.
- **Primary comparison:** EXP-006 full-dev results.

## Rationale

EXP-006 used the official baseline concat fusion:

```text
concat(audio_features, video_features) -> Conv1d -> fused_features
```

EXP-007 uses the existing ReliabilityGate module:

```text
audio_proj = Conv1d(audio_features)
video_proj = Conv1d(video_features)
gate = sigmoid(Conv1d(PReLU(Conv1d([audio, video, audio-video]))))
fused = audio_proj + gate * video_proj
output = Conv1d(fused)
```

The intended effect is to let the model reduce visual contribution when visual
evidence is unreliable, while keeping a strong audio path. This is especially
relevant because EXP-005 showed that the visual domain can destabilize official
dev inference if adapted too freely.

## Controlled Setup

- Architecture change: `fusion_type: reliability_gate`.
- Data: `/home/avse/processed_Chineselips_track2dev_match_v1`.
- Initial weight: official baseline serialized checkpoint
  `/home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth`.
- Warm-start migration: baseline concat weights are migrated into
  ReliabilityGate's `audio_proj` and `video_proj`; `out_proj` starts close to
  identity and the gate bias starts near open, following the existing EXP-002
  migration path.
- Frozen branch: all `av_model.video.*` parameters are frozen and
  `av_model.video` is forced to eval mode during training.
- `degrade_prob: 0.0`, matching EXP-006 to avoid adding another variable.
- Optimizer/lr/epochs/early stopping: same as EXP-006.

## Configuration

```text
configs/track2_av_convtasnet_reliability_gate_chineselips_devmatched_v1_degrade0_freeze_avvideo_2gpu.yml
```

Experiment directory:

```text
Experiments/Track2-AVConvTasNet-ReliabilityGate-ChineseLipsDevMatchedV1-Degrade0-FreezeAVVideoBN-2GPU
```

## Launch Plan

Run after config/model smoke checks:

```bash
cd /home/avse/workspace-goodparts
mkdir -p Experiments/Track2-AVConvTasNet-ReliabilityGate-ChineseLipsDevMatchedV1-Degrade0-FreezeAVVideoBN-2GPU/logs

CUDA_VISIBLE_DEVICES=5,6 \
/home/avse/avse_gpu_venv/bin/python train.py \
  --conf_dir configs/track2_av_convtasnet_reliability_gate_chineselips_devmatched_v1_degrade0_freeze_avvideo_2gpu.yml \
  --warm_start /home/avse/experiments/track2_chineselips_finetune_degrade0_lr_sweep/roundtrip/official_serialized.pth \
  2>&1 | tee Experiments/Track2-AVConvTasNet-ReliabilityGate-ChineseLipsDevMatchedV1-Degrade0-FreezeAVVideoBN-2GPU/logs/train.tmux.log
```

tmux session:

```text
exp007_reliability_gate_v1_freeze_video
```

## Evaluation Plan

After training:

1. export/verify `best_model.pth`;
2. run the three EXP-005/006 official-dev amplitude diagnostic samples;
3. run full dev enhancement;
4. run full amplitude validation before metrics;
5. compare against EXP-006 and official baseline:
   - remix objective: SI-SDR/PESQ/STOI;
   - no-reference/task metrics: UTMOS, DNSMOS, SPK, ASR.

## Runtime Notes

- 2026-07-14: EXP-007 config and plan created after EXP-006 full metrics
  completed.
- 2026-07-14: config/model smoke passed. The config parses correctly,
  `AV_ConvTasNet` instantiates with `ReliabilityGatedFusion`, and warm-start
  migration from the official baseline concat fusion runs as expected. The
  migration reports the expected missing new gate hidden-layer keys
  (`av_model.concat.gate.0.*`, `av_model.concat.gate.1.weight`) and unexpected
  old concat projection keys (`av_model.concat.conv1d.*`). `av_model.video`
  freeze check passed: 38 parameter tensors frozen, 0 trainable under that
  prefix, and `av_model.video.training == False`.
- 2026-07-14: launched formal training in tmux session
  `exp007_reliability_gate_v1_freeze_video` on physical GPUs 5 and 6.
  Startup log confirmed DDP with 2 processes, Track2 dev-matched v1 dataset
  counts (`15115` train mixtures, `1954` validation items, `3878` test items),
  13.2M trainable parameters and 12.0M non-trainable parameters. Lightning
  sanity checking passed and training entered epoch 0.
- 2026-07-14: epoch 0 completed successfully. Validation reported
  `val_loss = 1.76752`, Lightning marked it as the current best score, and saved
  `/home/avse/workspace-goodparts/Experiments/Track2-AVConvTasNet-ReliabilityGate-ChineseLipsDevMatchedV1-Degrade0-FreezeAVVideoBN-2GPU/epoch=0.ckpt`.
  The run then entered epoch 1. Host-side checks showed the tmux session still
  alive and physical GPUs 5/6 active.
- 2026-07-14: training remained healthy through epoch 38. Epoch 38 produced the
  current best validation score, `val_loss = -5.0814`, and saved
  `epoch=38.ckpt`; `last.ckpt` also records epoch 38 at global step 73671.
  Early stopping was reset by this improvement (`wait_count = 0`,
  `patience = 8`). At the latest check, epoch 39 was in progress on both GPUs.
- 2026-07-14: formal training completed normally after reaching
  `max_epochs = 80` (final epoch 79, global step 151120); early stopping did not
  trigger. The final best checkpoint is `epoch=76.ckpt` with
  `val_loss = -5.7601`. Epoch 79 finished at `val_loss = -5.7569`. The training
  script exported the selected weights to `best_model.pth` (101132382 bytes,
  SHA-256
  `4bc10522958af81e5ca1ad030e58a4190becaf4321a1bb84bc4c4597f1a6d5aa`).
  GPUs 5 and 6 were released after completion. Official-dev amplitude checks
  and full metric evaluation remain pending.
- 2026-07-15 00:11 CST: official-dev amplitude smoke passed with
  `align_face=True` and the exported `best_model.pth`. Diagnostic outputs were
  finite and nonzero: `track2/dev/mix/000242/s2` peak `0.1027`, RMS `0.0155`;
  `track2/dev/mix/000317/s2` peak `0.2288`, RMS `0.0291`;
  `track2/dev/remix/000008_000480/s1` peak `0.1083`, RMS `0.0180`. This clears
  the EXP-005 gain-explosion failure mode on the targeted samples.
- 2026-07-15 00:11 CST: launched the full official-dev evaluation in tmux
  session `exp007_full_dev_eval`. Enhancement is split across physical GPUs
  4, 5, and 6. The pipeline will validate all 5250 enhanced waveforms and stop
  before metrics if files are missing, nonfinite, all-zero, or have peak above
  `1.0`. If validation passes, it continues with objective, UTMOS, DNSMOS,
  speaker similarity, and ASR metrics in `/home/avse/avse_eval_py311`.

- 2026-07-15 00:14 CST: full official-dev enhancement completed successfully
  with 5250/5250 wavs. Full amplitude validation passed: missing `0`, nonfinite
  `0`, all-zero `0`, peak p50 `0.1050`, p90 `0.2157`, p99 `0.3123`, max
  `0.6085`, and `peak > 1` count `0`. There were 128 outputs with the same
  small length-mismatch behavior observed in EXP-006; this is non-fatal for the
  existing evaluator. The pipeline then entered the four-shard remix objective
  metric stage.
- 2026-07-15 00:19 CST: the four-shard remix objective evaluation completed
  and merged over all 2196 items: SI-SDR `-1.0002`, PESQ `1.3703`, and STOI
  `0.5581`. These remain clearly above the official baseline (`-2.8480`,
  `1.2561`, `0.4698`), but are below EXP-006 concat fusion (`-0.5850`,
  `1.4002`, `0.5785`) by `-0.4152` SI-SDR, `-0.0299` PESQ, and `-0.0204`
  STOI. The pipeline then entered three-shard UTMOS evaluation; final module
  interpretation remains pending all no-reference and task metrics.
- 2026-07-15 01:07 CST: three-shard UTMOS completed over all 5250 items.
  EXP-007 scored `1.5639` overall, `1.4802` on mix, and `1.6803` on remix.
  Relative to EXP-006, the gains are `+0.3196`, `+0.2650`, and `+0.3955`;
  relative to the official baseline, the gains are `+0.3881`, `+0.2960`, and
  `+0.5163`. This is a strong perceptual-quality signal in favor of
  ReliabilityGate despite its lower remix SI-SDR/PESQ/STOI than EXP-006. The
  pipeline then entered 12-shard DNSMOS evaluation.
- 2026-07-15 02:33 CST: the full evaluation pipeline completed successfully.
  DNSMOS, speaker similarity, and ASR all produced complete 5250-item merged
  summaries; the `FULL_EVAL_COMPLETE` marker was written and GPUs 4/5/6 were
  released. Total wall time from pipeline launch was about 2 hours 22 minutes.

## Full Dev Evaluation

Evaluation root:

```text
/home/avse/experiments/track2_exp007_reliability_gate_v1_full_dev
```

Enhanced wavs:

```text
/home/avse/experiments/track2_exp007_reliability_gate_v1_full_dev/enhanced
```

Metrics:

```text
/home/avse/experiments/track2_exp007_reliability_gate_v1_full_dev/metrics
```

Pipeline log:

```text
/home/avse/experiments/track2_exp007_reliability_gate_v1_full_dev/logs/full_eval_driver.log
```

Launch command:

```bash
tmux new-session -d -s exp007_full_dev_eval \
  -c /home/avse/workspace-goodparts \
  "bash /home/avse/experiments/track2_exp007_reliability_gate_v1_full_dev/scripts/run_full_eval.sh"
```

### Remix Results

`remix` contains clean target references, so both objective and no-reference
metrics are available. There are 2196 target-speaker evaluation items. Delta is
EXP-007 minus the official baseline; higher is better except for CER.

| metric | official baseline | EXP-007 proposed | delta | result |
|---|---:|---:|---:|---|
| SI-SDR | -2.8480 | **-1.0002** | +1.8478 | improved |
| PESQ | 1.2561 | **1.3703** | +0.1142 | improved |
| STOI | 0.4698 | **0.5581** | +0.0883 | improved |
| UTMOS | 1.1640 | **1.6803** | +0.5163 | improved |
| DNSMOS p808 | 2.3257 | **2.4529** | +0.1272 | improved |
| DNSMOS SIG | 1.6420 | **1.8399** | +0.1979 | improved |
| DNSMOS BAK | 1.6027 | **1.7539** | +0.1512 | improved |
| DNSMOS OVR | 1.3218 | **1.4383** | +0.1165 | improved |
| speaker similarity | 0.3506 | **0.4380** | +0.0874 | improved |
| CER (lower is better) | **0.8658** | 0.8864 | +0.0206 | regressed |

### Mix Results

`mix` has no clean target waveform. SI-SDR, PESQ, and STOI are therefore not
defined, and the comparison uses no-reference quality, speaker similarity, and
ASR metrics over 3054 target-speaker evaluation items.

| metric | official baseline | EXP-007 proposed | delta | result |
|---|---:|---:|---:|---|
| UTMOS | 1.1842 | **1.4802** | +0.2960 | improved |
| DNSMOS p808 | 2.4023 | **2.4814** | +0.0791 | improved |
| DNSMOS SIG | **1.8188** | 1.8032 | -0.0156 | regressed |
| DNSMOS BAK | **1.7760** | 1.7519 | -0.0241 | regressed |
| DNSMOS OVR | 1.4486 | **1.4631** | +0.0145 | improved |
| speaker similarity | 0.3842 | **0.4649** | +0.0807 | improved |
| CER (lower is better) | 0.8922 | **0.8117** | -0.0805 | improved |

## Interpretation

Against the official baseline, the complete EXP-007 proposed recipe is a clear
positive result on both official-dev domains:

- on remix, 9 of 10 reported metrics improve; only CER regresses slightly;
- on mix, 5 of 7 reported metrics improve; DNSMOS SIG and BAK regress slightly;
- the strongest gains are remix UTMOS (`+0.5163`), mix UTMOS (`+0.2960`),
  remix SI-SDR (`+1.8478`), and mix CER (`-0.0805`, lower is better);
- full output-amplitude validation rules out the EXP-005 gain-explosion failure.

These paper-facing tables intentionally compare only the official baseline and
the complete proposed recipe. Consequently, they support the claim that the
full EXP-007 method improves the baseline, but do not isolate the contribution
of ReliabilityGate from the dev-matched data and frozen-visual-branch training
choices. Earlier experiments remain internal ablations for that attribution.
