# EXP-008: Hierarchical visual temporal fusion v1/v2

## Summary

- **Experiment ID:** EXP-008
- **Name:** Hierarchical visual temporal fusion for Track 2 AV-ConvTasNet
- **Date:** 2026-07-15
- **Owner:** TODO
- **Status:** Planned
- **Purpose:** Test whether short-, medium-, and longer-context visual features from the existing V-TCN provide complementary guidance beyond its final output alone.
- **Direct parent code:** `2aeb35a65f15fc0c215abc5c651f4f7579edd5d4`
- **Parent result:** EXP-006, with EXP-007 retained as a secondary comparison
- **Immutable source baseline:** `baseline/track2-source-v1`
- **Branch:** `exp/hierarchical-visual-fusion`
- **Formal-run commit SHA:** TODO

This record defines two ordered implementations under one coherent research
question. V1 tests whether multiple visual temporal depths help at all. V2 is
evaluated only after V1 and tests whether assigning those depths to different
audio stages with independent reliability control adds further value.

## Background and prior evidence

The current separator passes ResNet lip embeddings through five kernel-3 V-TCN
blocks but exposes only the final V-TCN output to one audio-visual fusion point.
At 25 fps and dilation 1, blocks 1, 3, and 5 have theoretical temporal receptive
fields of 3, 7, and 11 frames, approximately 120, 280, and 440 ms. These depths
are treated as candidate local-articulation, dynamic phonetic/synchronization,
and longer phonetic-context features. They are not assumed to be proven
linguistic semantics.

The design is motivated by, but intentionally narrower than, prior work:

- MFFCN showed that layer-wise audio-visual fusion can outperform a single
  fusion point: <https://arxiv.org/abs/2101.05975>.
- Visual-cue analysis found that AVSE visual embeddings contain speech activity
  and fine-grained place-of-articulation information:
  <https://arxiv.org/abs/2004.12031>.
- Work decoupling AV speaker cues found both identity and synchronization useful,
  with synchronization having the larger effect:
  <https://arxiv.org/abs/2306.02625>.
- RAVEN found AVSR and active-speaker embeddings contribute differently across
  acoustic conditions: <https://www.isca-archive.org/interspeech_2025/ma25c_interspeech.html>.
- Multi-scale adaptive modal weighting has appeared in recent AVSE challenge
  systems, so generic multi-layer concatenation alone is not treated as novel:
  <https://www.isca-archive.org/avsec_2025/ren25_avsec.html>.

The experiment-specific question is whether a baseline-compatible hierarchy
inside this AV-ConvTasNet improves the already positive EXP-006 recipe without
unfreezing the unstable internal visual branch.

## Hypotheses

### H1 / V1: multi-depth visual aggregation

V-TCN layers 1 and 3 contain complementary short- and medium-context evidence
that is attenuated when only layer 5 is used. Adding trainable residual adapters
from layers 1 and 3 to the layer-5 representation should improve or preserve
EXP-006 objective metrics while retaining stable waveform amplitude.

Expected behavior:

- initial behavior remains close to layer-5-only fusion because shallow
  residual scales start small;
- learned shallow scales become non-zero if those depths add useful evidence;
- ablating layer 1 or layer 3 after training produces a measurable metric or
  validation-loss change if the hierarchy is genuinely used.

### H2 / V2: stage-matched reliability-aware injection

A single aggregated fusion point cannot express that visual depths should be
used at different audio abstraction stages or rejected independently when
corrupted. Starting from V1, small gated residual conditioners should inject:

1. layer 1 before the existing central fusion;
2. layer 3 after the first post-fusion audio repeat;
3. layer 5 after the second post-fusion audio repeat.

V2 should improve the Track 2 aggregate or its most visually degraded subsets
over V1 without regressing EXP-006 amplitude safety. Each stage gate should be
finite, non-constant on real validation data, and respond differently to at
least some degradation types.

## Ordered implementation

### V1: baseline-compatible hierarchical aggregation

```text
ResNet output
  -> V-TCN block 1 ---- identity 1x1 adapter -- small residual scale --+
  -> V-TCN block 3 ---- identity 1x1 adapter -- small residual scale --+--> hierarchical visual feature
  -> V-TCN block 5 -------------------------------- layer-5 base ------+
                                                                  |
audio repeats before fusion -------------------------------------- concat -> remaining audio repeats
```

The existing final layer is the base path. Extra depth adapters are residual,
so setting their scales to zero exactly recovers the existing visual feature
and checkpoint-compatible fusion behavior.

### V2: V1 plus stage-matched gated residuals

```text
audio pre-fusion feature + Gate1(audio, V-TCN-1)
  -> existing V1 aggregate fusion
  -> post-fusion audio repeat 1 + Gate2(audio, V-TCN-3)
  -> post-fusion audio repeat 2 + Gate3(audio, V-TCN-5)
  -> remaining audio repeat(s) -> mask -> decoder
```

Each gate receives aligned audio, projected video, and their absolute
difference. It controls a residual visual update rather than replacing the
audio path. V2 requires `skip_con: false`, matching the EXP-006 configuration,
because repeat-boundary states are used for stage injection.

## Controlled differences

Changed:

- expose selected intermediate V-TCN outputs without modifying their block
  definitions or parameters;
- V1 adds a baseline-preserving multi-depth visual aggregator;
- V2 adds three scale-specific gated residual conditioners;
- add dedicated V1/V2 configurations, checks, and diagnostics.

Intentionally unchanged:

- frozen ResNet video encoder;
- frozen and force-eval `av_model.video` policy validated by EXP-006;
- audio encoder, mask estimator, decoder, loss, data split, evaluation path,
  and official-dev face alignment;
- EXP-006 learning rate, duration, batch size, seed, and dev-matched data recipe
  for the first controlled comparison;
- all official baseline configurations and existing experiment records.

## Planned modified files

| File | Planned change | Reason |
|---|---|---|
| `look2hear/models/av_convtasnet.py` | intermediate V-TCN capture, V1 aggregator, V2 stage gates | implement the hierarchy while preserving default behavior |
| `configs/track2_av_convtasnet_hierarchical_v1.yml` | dedicated V1 run | isolate H1 |
| `configs/track2_av_convtasnet_hierarchical_v2.yml` | dedicated V2 run | isolate H2 |
| `scripts/check_hierarchical_visual_fusion.py` | synthetic shapes, fallback, gradients, config checks | reject unsafe implementations before a server run |
| `EXPERIMENTS.md` | EXP-008 registry row | provenance |
| this report | protocol, commands, results placeholders | research traceability |

## Configuration plan

Shared V1/V2 settings are copied from the successful EXP-006 recipe:

- warm start: verified EXP-006 `best_model.pth` or serialized equivalent;
- data: Chinese-LiPS Track2-dev-matched v1 paths from
  `docs/experiment-conventions.md`;
- `D: 5`, selected layers `[1, 3, 5]` using one-based numbering;
- `audio_index: 1`, `R: 4`, `X: 8`, `skip_con: false`;
- freeze and force-eval `av_model.video`;
- Adam learning rate `1e-5`, maximum 80 epochs, early-stop patience 8;
- seed `20260714`, batch size 4 on two visible GPUs;
- first controlled comparison keeps `degrade_prob: 0.0`.

V1-specific:

```yaml
hierarchical_fusion_version: v1
hierarchical_video_layers: [1, 3, 5]
hierarchical_residual_init: 0.05
```

V2-specific:

```yaml
hierarchical_fusion_version: v2
hierarchical_video_layers: [1, 3, 5]
hierarchical_residual_init: 0.05
hierarchical_gate_hidden: 128
hierarchical_stage_residual_init: 0.05
```

`degrade_prob: 0.0` is not claimed to train comprehensive degradation routing.
If V2 is structurally positive, mixed online degradation becomes a separately
recorded follow-up rather than an untracked change to EXP-008.

## Evaluation and locked comparisons

Primary controlled baseline: EXP-006 full Track 2 dev.

| Scope | Metric | EXP-006 reference | V1 | V2 | Direction |
|---|---|---:|---:|---:|---|
| remix, n=2196 | SI-SDR | -0.5850 | TODO | TODO | higher |
| remix, n=2196 | PESQ | 1.4002 | TODO | TODO | higher |
| remix, n=2196 | STOI | 0.5785 | TODO | TODO | higher |
| overall | UTMOS | 1.2443 | TODO | TODO | higher |
| overall | DNSMOS p808 | 2.4873 | TODO | TODO | higher |
| overall | speaker similarity | 0.4673 | TODO | TODO | higher |
| overall | CER | 0.8725 | TODO | TODO | lower |

Required diagnostics:

- full 5250-item output completeness and amplitude distribution;
- validation loss and peak GPU memory;
- V1 residual scales and per-level ablation;
- V2 gate mean/std by stage;
- clean, masked/occluded, low-resolution, frozen, missing-frame, and desync
  subset metrics when degradation labels are available;
- zero-video, frozen-video, and time-shifted-video sensitivity checks.

No single proxy metric is sufficient for retention. A candidate that improves
SI-SDR while causing amplitude explosions, major CER regression, or missing
outputs is invalid.

## Commands

### Preflight and synthetic checks

```bash
git status --short --branch
python -m compileall look2hear/models/av_convtasnet.py train.py
python scripts/check_hierarchical_visual_fusion.py
```

### V1 formal run

```bash
CUDA_VISIBLE_DEVICES=5,6 /home/avse/avse_gpu_venv/bin/python train.py \
  --conf_dir configs/track2_av_convtasnet_hierarchical_v1.yml \
  --warm_start <verified-exp006-checkpoint>
```

### V2 formal run

For a fair architecture comparison, first warm-start V2 from the same verified
EXP-006 checkpoint. A later V1-to-V2 continuation must be labeled exploratory.

```bash
CUDA_VISIBLE_DEVICES=5,6 /home/avse/avse_gpu_venv/bin/python train.py \
  --conf_dir configs/track2_av_convtasnet_hierarchical_v2.yml \
  --warm_start <verified-exp006-checkpoint>
```

Formal training is not authorized or performed by this implementation change.
The GPU allocation, checkpoint checksum, output root, and run budget must be
confirmed immediately before launch.

## Risks and falsification conditions

- Intermediate V-TCN depth may not correspond to the proposed articulatory and
  phonetic hierarchy. Non-zero adapters alone do not prove that interpretation.
- Because EXP-006 freezes `av_model.video`, V1/V2 can only re-use fixed depth
  features; this is intentional for stability but may limit the upper bound.
- Repeated visual injection can amplify unreliable video. Residual scales and
  the unchanged audio path are safety mechanisms, not guarantees.
- With `degrade_prob: 0.0`, V2 gate specialization may be weak. Failure under
  that setting does not rule out a mixed-degradation curriculum.
- V1 is falsified if it does not improve the locked metrics or if ablating its
  shallow levels has no measurable effect after a valid run.
- V2 is falsified if it does not improve over V1, its gates collapse to
  constants, or it introduces amplitude/output-completeness failures.

## Runtime, artifacts, results, and conclusion

- **Configuration hashes:** TODO after implementation
- **Formal-run commit:** TODO
- **Checkpoint and checksum:** TODO
- **Logs, predictions, metric paths:** TODO
- **Runtime and peak memory:** TODO
- **Artifact validation:** TODO
- **Result:** No formal run has started
- **Conclusion:** Inconclusive / protocol only
- **Candidate for merge:** No while unverified
- **Next step:** implement V1/V2, pass synthetic and warm-start smoke checks,
  then request/confirm the server run budget before formal training.
