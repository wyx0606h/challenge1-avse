# EXP-009: Pretrained visual representations v1/v2

## Summary

- **Experiment ID:** EXP-009
- **Name:** AV-HuBERT local and contextual visual representations for Track 2
- **Date:** 2026-07-16
- **Owner:** TODO
- **Status:** Planned
- **Purpose:** Replace the weak legacy lip encoder while preserving the
  already-positive AV-ConvTasNet audio path and EXP-008 V1 hierarchical fusion.
- **Direct parent code:** `3c41486b29bee98aa20c20d5fab54875864b9cd3`
- **Parent architecture:** EXP-008 V1 implementation
- **Primary result baseline:** verified EXP-008 V1 run with
  `degrade_prob: 0.0`; until available, EXP-006 is a secondary reference only.
- **Immutable source baseline:** `baseline/track2-source-v1`
- **Branch:** `exp/pretrained-visual-representations`
- **Initial protocol commit:** `245aab3`
- **Audio-first protocol amendment:** `a2ec7b0`
- **Implementation commit:** TODO
- **Formal-run commit SHA:** TODO

This record defines two ordered versions under one coherent question: whether
stronger pretrained visual speech representations improve challenge metrics
when the separator, data recipe, and degradation probability are controlled.
V1 tests the representation replacement itself. V2 is evaluated only after V1
and tests whether a small contextual residual complements local lip features
when visual injection is anchored by the audio representation.

## Post-protocol audio-first design review

After the initial protocol commit, a collaborator raised a valid concern:
Track 2 deliberately degrades video, so the model should protect and strengthen
the audio path instead of blindly increasing visual dependence.

The review changes V2 but does not turn EXP-009 into an audio-only experiment:

- this challenge runs the same mixture once per target speaker and distinguishes
  the desired output through that target's lip video; a single-output,
  audio-only model has no deterministic `s1`/`s2` selection signal unless a
  separate enrollment or multi-output separation mechanism is introduced;
- “visual-only AV-HuBERT” means only that the external representation extractor
  receives video; the complete AVSE model still retains the Conv-TasNet audio
  encoder and separator;
- V1 remains the clean representation-replacement test;
- V2 is amended to use an audio-anchored residual fusion, so the aligned audio
  feature is the base path and local/context visual evidence is a bounded
  increment rather than an unconditional replacement;
- audio SSL is retained as a separate future hypothesis. WavLM is a stronger
  first candidate than forcing AV-HuBERT's audio branch into this waveform
  Conv-TasNet because WavLM includes denoising in pretraining:
  <https://arxiv.org/abs/2110.13900>.

This review was recorded before the implementation commit. It is a protocol
amendment, not a result.

## Decision on `degrade_prob`

The first V1/V2 runs are locked to:

```yaml
degrade_prob: 0.0
```

This is based on the complete EXP-007 dev comparison supplied on 2026-07-16.
Changing `degrade_prob` from 0.0 to 1.0 mainly improved CER but reduced all four
DNSMOS measures:

| Scope | Metric | prob=0.0 | prob=1.0 | Delta |
|---|---|---:|---:|---:|
| remix, n=2196 | SI-SDR | -1.0002 | -0.9996 | +0.0006 |
| remix, n=2196 | PESQ | 1.3703 | 1.3735 | +0.0032 |
| remix, n=2196 | STOI | 0.5581 | 0.5612 | +0.0031 |
| remix, n=2196 | UTMOS | 1.6803 | 1.6749 | -0.0054 |
| remix, n=2196 | DNSMOS p808 | 2.4529 | 2.4459 | -0.0070 |
| remix, n=2196 | DNSMOS SIG | 1.8399 | 1.8045 | -0.0354 |
| remix, n=2196 | DNSMOS BAK | 1.7539 | 1.7078 | -0.0461 |
| remix, n=2196 | DNSMOS OVR | 1.4383 | 1.4177 | -0.0206 |
| remix, n=2196 | SPK | 0.4380 | 0.4368 | -0.0012 |
| remix, n=2196 | CER | 0.8864 | 0.8765 | -0.0099 |
| mix, n=3054 | UTMOS | 1.4802 | 1.4967 | +0.0165 |
| mix, n=3054 | DNSMOS p808 | 2.4814 | 2.4703 | -0.0111 |
| mix, n=3054 | DNSMOS SIG | 1.8032 | 1.7757 | -0.0275 |
| mix, n=3054 | DNSMOS BAK | 1.7519 | 1.7097 | -0.0422 |
| mix, n=3054 | DNSMOS OVR | 1.4631 | 1.4463 | -0.0168 |
| mix, n=3054 | SPK | 0.4649 | 0.4637 | -0.0012 |
| mix, n=3054 | CER | 0.8117 | 0.7699 | -0.0418 |

The challenge ranks DNSMOS SIG/BAK/OVR/p808 as separate components, while CER
is one component. More importantly, changing both the visual representation
and online degradation would confound the architecture comparison. A later
`degrade_prob` experiment must therefore be separately identified and should
include at least 0.0, an intermediate probability, and 1.0.

## Background and literature synthesis

The current model uses a randomly initialized or task-specific ResNet visual
encoder followed by a frozen V-TCN. Replacing that encoder is motivated by the
following primary evidence:

- AV-HuBERT learns self-supervised audio-visual speech representations and
  supports visual-only inference at the standard 25 fps visual feature rate:
  <https://github.com/facebookresearch/av_hubert>.
- Multi-modal self-supervised AV-HuBERT embeddings have transferred to AVSE and
  AV speech separation with suitable fine-tuning:
  <https://arxiv.org/abs/2210.17456>.
- AV-HuBERT visual cues improved target speech extraction over traditional
  visual baselines in the mask-and-recover study:
  <https://arxiv.org/abs/2403.16078>.
- RAVEN specifically studies pretrained AVSR/ASD visual representations for
  real-time AVSE. Its official AV-HuBERT extractor uses the visual frontend,
  producing 768-dimensional features at 25 fps:
  <https://github.com/Bose/RAVEN/> and
  <https://www.isca-archive.org/interspeech_2025/ma25c_interspeech.pdf>.
- Analysis of audio-visual speaker extraction finds both identity and
  synchronization useful, with synchronization contributing more strongly:
  <https://arxiv.org/abs/2306.02625>.
- Robust AV systems benefit from measuring or gating unreliable visual
  evidence rather than always trusting it:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Hong_Watch_or_Listen_Robust_Audio-Visual_Speech_Recognition_With_Visual_Corruption_CVPR_2023_paper.html>.

The resulting design deliberately avoids replacing the full AVSE network. The
known-positive audio encoder, separator, decoder, loss, and evaluation path are
preserved. Only the representation delivered to the existing visual temporal
and fusion path changes.

## Hypotheses

### H1 / V1: frozen AV-HuBERT visual frontend

The AV-HuBERT visual frontend contains stronger local articulatory and
short-range motion features than the legacy ResNet. A frozen frontend followed
by a trainable normalization and 768-to-512 temporal adapter should improve the
controlled EXP-008 V1 result without destabilizing waveform amplitude.

Expected behavior:

- the external pretrained backbone remains frozen and in evaluation mode;
- only the adapter and existing trainable separator paths receive gradients;
- preprocessing and 25 fps timing remain compatible with the current data;
- SI-SDR/PESQ/STOI and the challenge perceptual metrics improve or remain
  stable, without a material SPK or CER regression.

### H2 / V2: local/context representation with audio-anchored fusion

AV-HuBERT transformer states contain longer-context speech information, but
the public LRS3 model is English-dominant while the current matched training
data and challenge speech are Chinese. Replacing local features completely
with the final contextual state may therefore overfit language-specific
patterns.

V2 first keeps the V1 frontend projection as the visual base and adds a small,
learnable, gated contextual residual:

```text
local = local_adapter(AV-HuBERT visual frontend)
context = context_adapter(AV-HuBERT visual transformer layer L)
gate = sigmoid(MLP([local, context, abs(local - context)]))
visual = local + tanh(scale) * gate * context
```

The default context layer is configurable and initially set to the final
encoder layer. The residual scale starts at 0.05 so initialization remains
close to V1. This tests the requested “lip shape plus semantic/contextual
information” idea without forcing the contextual representation to dominate.

At the audio-visual fusion point, V2 preserves the baseline concat projection
and gates only its visual input. The old `conv1d` key and shape remain
unchanged, so the EXP-008 V1 audio/visual projection weights load exactly:

```text
visual_gate = sigmoid(MLP([audio, visual, abs(audio - visual)]))
scaled_visual = tanh(visual_scale) * visual_gate * visual
fused = old_concat_projection([audio, scaled_visual])
```

When the visual scale closes, the old projection's learned audio slice and bias
remain active. Its visual scale starts at 0.05.

Expected behavior:

- the local/context visual representation is initially close to V1, while the
  second fusion scale deliberately starts audio-dominant;
- the contextual scale and gate become non-zero/non-constant if context helps;
- the audio path remains available when the visual gate closes;
- V2 improves CER or speech quality without the DNSMOS loss seen from
  `degrade_prob: 1.0`;
- contextual ablation after training produces a measurable change if the
  second representation is actually used.

## Ordered implementations

### V1: AV-HuBERT frontend replacement

```text
mouth frames
  -> frozen AV-HuBERT visual frontend (768 channels, 25 fps)
  -> trainable temporal LayerNorm + 1x1 adapter (512 channels)
  -> existing EXP-008 V1 V-TCN layers 1/3/5 aggregation
  -> existing audio-visual fusion and separator
```

### V2: local/context representation plus audio-anchored residual

```text
mouth frames
  -> frozen AV-HuBERT visual frontend -----------------> local adapter ----+
  -> frozen AV-HuBERT transformer through layer L ----> context adapter --+
                                                                        |
                  local + small gated context residual <----------------+
                                      |
                         existing EXP-008 V1 hierarchy
                                      |
audio pre-fusion state ---------------+--> audio-anchored visual residual
                                      |
                             remaining audio separator
```

An active-speaker-detection encoder is not included in these first two
versions. RAVEN suggests ASD features can complement AVSR features in
multi-speaker/low-SNR conditions, but adding a second external model would mix
two hypotheses and substantially increase dependency and deployment risk.

## Controlled differences

Changed:

- replace the legacy ResNet visual encoder with an external frozen AV-HuBERT
  visual representation;
- V1 adds only a trainable 768-to-512 adapter;
- V2 adds a contextual projection/gate and an audio-anchored visual residual;
- add dedicated V1/V2 configurations, asset checks, and diagnostics.

Intentionally unchanged:

- EXP-008 V1 hierarchical V-TCN layers `[1, 3, 5]`;
- audio encoder, audio TCN, fusion type, mask estimator, decoder, loss, and
  official evaluation code;
- Chinese-LiPS Track2-dev-matched v1 split and normalization;
- learning rate, stopping rule, batch size, and seed for the controlled run;
- `degrade_prob: 0.0`;
- official baseline configurations and previous experiment records.

## External assets and licensing

No external source checkout, pretrained weight, dataset, or generated feature
is committed to Git.

Required server-side assets:

- official AV-HuBERT repository checkout, supplied through `AVHUBERT_ROOT`;
- official `base_lrs3_433h.pt` visual speech recognition checkpoint, supplied
  through `AVHUBERT_CKPT`;
- a compatible AV-HuBERT/fairseq environment.

The AV-HuBERT repository and checkpoint are subject to Meta's AV-HuBERT
license agreement. License acceptance and the checkpoint checksum must be
confirmed before the formal run. The official checkpoint page is:
<https://facebookresearch.github.io/av_hubert/>.

The existing lip preprocessing uses the same commonly used AV-HuBERT/RAVEN
normalization (`mean=0.421`, `std=0.165`) and compatible cropped grayscale
mouth inputs. This must still be verified on the target server before training.

## Implemented files

| File | Change | Reason |
|---|---|---|
| `look2hear/videomodels/avhubert_videomodel.py` | lazy external loader, frozen frontend/context extraction, V1/V2 adapters and diagnostics | isolate optional dependency and pretrained logic |
| `look2hear/videomodels/__init__.py` | export `AVHubertVideoModel` | model registry |
| `look2hear/models/av_convtasnet.py` | opt-in encoder selection, adapter-gradient path, audio-anchored fusion, diagnostics | connect representations while preserving the default ResNet path |
| `configs/track2_av_convtasnet_pretrained_visual_v1.yml` | controlled V1 with frontend representation and concat | test H1 |
| `configs/track2_av_convtasnet_pretrained_visual_v2.yml` | dual representation plus audio-anchored gate | test amended H2 |
| `scripts/check_pretrained_visual_representations.py` | config-only and deferred server asset/tensor checks | validate without committing assets |
| `EXPERIMENTS.md` | EXP-009 registry row | provenance |
| this report | protocol and result placeholders | traceability |

## Configuration plan

Shared settings:

- warm start: the verified EXP-008 V1 checkpoint for the primary controlled
  comparison; if unavailable, an EXP-006 warm start is exploratory and cannot
  establish the isolated visual-representation delta;
- `visual_encoder_type: avhubert`;
- `hierarchical_fusion_version: v1`;
- `hierarchical_video_layers: [1, 3, 5]`;
- `E: 512`, `D: 5`, `audio_index: 1`, `R: 4`, `skip_con: false`;
- Adam learning rate `1e-5`, maximum 80 epochs, early-stop patience 8;
- seed `20260716`, batch size 4 on two visible GPUs;
- frozen external AV-HuBERT backbone;
- `degrade_prob: 0.0`.

V1:

```yaml
visual_feature_mode: frontend
visual_adapter_out: 512
```

V2:

```yaml
visual_feature_mode: dual
visual_context_layer: 12
visual_gate_hidden: 128
visual_context_residual_init: 0.05
visual_adapter_out: 512
fusion_type: audio_anchored_gate
fusion_gate_hidden: 128
fusion_visual_residual_init: 0.05
```

Configuration SHA-256 values:

- V1:
  `1766A8A34AC3756C78923EE30588D08C9A4179783C8C372D677DF04D4281C68A`
- V2:
  `0C4EFD644858F16BCC5AC3123B02964A5DE81D0F38E14D08AA3386AD7A46F9DB`

## Evaluation and acceptance criteria

Primary comparison:

1. EXP-008 V1, `degrade_prob=0.0`;
2. EXP-009 V1, same parent/data/optimization;
3. EXP-009 V2, same parent/data/optimization.

All official dev metrics must be retained separately for remix and mix where
defined: SI-SDR, PESQ, STOI, UTMOS, DNSMOS p808/SIG/BAK/OVR, SPK, and CER.

Required checks:

- 5250-item output completeness and waveform amplitude distribution;
- no NaN/Inf and no missing/duplicate utterances;
- V1 adapter gradient and frozen-backbone verification;
- V2 context and audio-visual residual scales plus both gate distributions;
- V2 contextual-off ablation;
- V2 visual-residual-off audio fallback check;
- zero-video and time-shifted-video sensitivity;
- runtime, peak memory, and real-time implications;
- separate mix/remix and visually degraded subset analysis when labels exist.

Retention rule:

- V1 is retained only if it improves the multi-metric challenge profile over
  EXP-008 V1 without a material amplitude, SPK, CER, or DNSMOS failure.
- V2 is retained only if it improves over V1 or produces a useful
  quality/intelligibility Pareto trade-off supported by the full metrics.
- A CER-only gain accompanied by broad DNSMOS loss is not sufficient.

## Commands

### Local protocol/static checks

```bash
git status --short --branch
git diff --check
python -m py_compile \
  look2hear/videomodels/avhubert_videomodel.py \
  look2hear/models/av_convtasnet.py \
  scripts/check_pretrained_visual_representations.py
python scripts/check_pretrained_visual_representations.py --config-only
```

Results on 2026-07-16:

- `py_compile`: passed for all touched Python files;
- EXP-009 config-only validation: passed;
- `git diff --check`: passed;
- local runtime import: not validated because the existing Windows Python has
  NumPy 2.0.2 with a PyTorch binary compiled against NumPy 1.x, which fails
  while importing `torch` with `_ARRAY_API not found`;
- no AV-HuBERT load, tensor forward, gradient check, or training was executed.

The local ABI mismatch is an environment limitation rather than a passed or
failed EXP-009 model check. The target-server command below remains mandatory.

### Deferred server asset and tensor checks

```bash
export AVHUBERT_ROOT=/external/path/to/av_hubert
export AVHUBERT_CKPT=/external/path/to/base_lrs3_433h.pt
/home/avse/avse_gpu_venv/bin/python \
  scripts/check_pretrained_visual_representations.py --with-avhubert
```

### Formal V1 run

```bash
export AVHUBERT_ROOT=/external/path/to/av_hubert
export AVHUBERT_CKPT=/external/path/to/base_lrs3_433h.pt
CUDA_VISIBLE_DEVICES=5,6 /home/avse/avse_gpu_venv/bin/python train.py \
  --conf_dir configs/track2_av_convtasnet_pretrained_visual_v1.yml \
  --warm_start <verified-exp008-v1-checkpoint>
```

### Formal V2 run

```bash
export AVHUBERT_ROOT=/external/path/to/av_hubert
export AVHUBERT_CKPT=/external/path/to/base_lrs3_433h.pt
CUDA_VISIBLE_DEVICES=5,6 /home/avse/avse_gpu_venv/bin/python train.py \
  --conf_dir configs/track2_av_convtasnet_pretrained_visual_v2.yml \
  --warm_start <verified-exp008-v1-checkpoint>
```

The exact external paths, checkpoint checksum, GPU allocation, and artifact
root remain TODO until the server is connected. No forward pass, download, or
formal training is part of this local implementation.

## Risks and falsification conditions

- The official AV-HuBERT stack uses an old fairseq/hydra environment and may
  conflict with the current training environment. A side environment,
  pre-extraction cache, or compatibility patch may be needed.
- Runtime extraction is simpler for the first controlled test but increases
  compute and memory. Offline feature extraction is a later engineering
  optimization, not part of H1/H2.
- The LRS3 checkpoint is English-dominant. V1's local frontend is expected to
  transfer more safely; V2 therefore uses contextual features only as a small
  gated residual.
- A pure audio-only fallback cannot reliably identify `s1` versus `s2` from the
  identical mixture. Closing the visual gate is therefore a robustness fallback,
  not a complete replacement for target conditioning.
- Adding WavLM/audio SSL may improve perceptual quality, but its waveform rate,
  representation rate, model size, and fusion placement form a separate
  experiment and are not mixed into EXP-009.
- Stronger lip-reading features may improve CER while oversuppressing noise or
  speech components, reproducing the DNSMOS trade-off. Full metrics determine
  retention.
- V1 is falsified if it does not beat the controlled EXP-008 V1 result or if
  the adapter is unused.
- V2 is falsified if it does not beat V1, its gate/scale collapses, or its
  contextual-off ablation is indistinguishable.

## Runtime, artifacts, results, and conclusion

- **Configuration hashes:** recorded above
- **Static syntax/config validation:** Passed on 2026-07-16
- **Local runtime import:** Blocked by pre-existing NumPy/PyTorch ABI mismatch
- **AV-HuBERT asset/tensor validation:** TODO on target server
- **Formal-run commit:** TODO
- **Checkpoint and checksum:** TODO
- **Logs, predictions, metric paths:** TODO
- **Runtime and peak memory:** TODO
- **Artifact validation:** TODO
- **Result:** No formal run has started
- **Conclusion:** Inconclusive / protocol only
- **Candidate for merge:** No while unverified
- **Next step:** perform the deferred server asset/tensor check, then run V1
  before deciding whether to allocate training time to V2.
