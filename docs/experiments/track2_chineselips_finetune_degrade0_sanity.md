# Track 2 Chinese-LiPS Fine-Tune Sanity, degrade_prob=0

## Summary

- **Date:** 2026-07-06
- **Branch:** `exp/baseline-reproduction`
- **Purpose:** first short fine-tuning sanity run from the official Track 2
  baseline checkpoint on the converted Chinese-LiPS training set, with visual
  degradation disabled.
- **Status:** training and dev enhancement completed, but dev objective metrics
  regressed badly. Treat this as a failed fine-tuning sanity, not an improved
  model.

## Inputs

- **Repository:** `/home/avse/workspace-goodparts`
- **Training data:** `/home/avse/processed_Chineselips`
- **Official Track 2 weights:**
  `/home/avse/avse-assets/models/Real-World-AVSE-Baseline-Track2`
- **Dev data root used for eval:**
  `/home/avse/experiments/track2_dev_official_baseline/data_root`
- **Evaluation assets:** `/home/avse/avse-assets/evaluation`
- **Experiment output:**
  `/home/avse/experiments/track2_chineselips_finetune_degrade0_sanity`

The converted Chinese-LiPS data has `tr/cv/tt/{mix.json,s1.json,s2.json}` and
was previously validated with the official Track 2 Dataset/DataLoader and model
forward/loss/backward smoke.

## Environment

- **Virtualenv:** `/home/avse/avse_gpu_venv`
- **GPU:** GPU 4, RTX 4090
- **Training entry:** `scripts/finetune_track2_sanity.py`
- **Evaluation entry:** `eval_real.py`

This run used the Python 3.8 GPU environment because it is sufficient for
training, enhancement, and objective metrics. Python 3.11 remains the preferred
environment for full UTMOS/CER evaluation.

## Training Setup

- Start point: official local `config.json + model.safetensors`.
- Model: official `AV_ConvTasNet.from_pretrained`.
- Dataset:
  - train: `Track2DynamicDataset`, `/home/avse/processed_Chineselips/tr`
  - validation: `Track2StaticDataset`, `/home/avse/processed_Chineselips/cv`
- Loss:
  - train: `PITLossWrapper(pairwise_neg_snr)`
  - validation: `PITLossWrapper(pairwise_neg_sisdr)`
- `degrade_prob=0.0`
- `segment=2.0`
- `normalize_audio=false`
- `batch_size=4`
- `num_workers=2`
- `lr=1e-5`
- `max_steps=500`
- seed: `20260706`

Command:

```bash
CUDA_VISIBLE_DEVICES=4 /home/avse/avse_gpu_venv/bin/python \
  /home/avse/workspace-goodparts/scripts/finetune_track2_sanity.py \
  --dataset-root /home/avse/processed_Chineselips \
  --pretrained-dir /home/avse/avse-assets/models/Real-World-AVSE-Baseline-Track2 \
  --output-dir /home/avse/experiments/track2_chineselips_finetune_degrade0_sanity/train_500step \
  --max-steps 500 --batch-size 4 --num-workers 2 \
  --val-every 100 --val-batches 50 \
  --degrade-prob 0.0 --lr 1e-5 --overwrite
```

Training artifacts:

- `train_500step/best_model.pth`
- `train_500step/final_model.pth`
- `train_500step/metrics.csv`
- `train_500step/run_config.yml`
- `train_500step/summary.json`

Loss summary:

| step | validation loss |
|---:|---:|
| 100 | 3.561521 |
| 200 | 3.055647 |
| 300 | 2.653464 |
| 400 | 2.756550 |
| 500 | 2.499934 |

Final training summary:

- last train loss: `-4.592978`
- best validation loss: `2.499934`
- elapsed: `177.9 s`
- finite output/loss/gradient checks: passed

## Dev Enhancement

Command:

```bash
/home/avse/avse_gpu_venv/bin/python eval_real.py \
  --conf_dir configs/track2_av_convtasnet.yml \
  --ckpt /home/avse/experiments/track2_chineselips_finetune_degrade0_sanity/train_500step/best_model.pth \
  --data_root /home/avse/experiments/track2_dev_official_baseline/data_root \
  --track track2 --scene both --split dev \
  --metrics none --mode enhance \
  --save_dir /home/avse/experiments/track2_chineselips_finetune_degrade0_sanity/dev_eval/enhanced \
  --gpus 4
```

Output validation:

| item | value |
|---|---:|
| expected outputs | 5250 |
| produced outputs | 5250 |
| mix / remix | 3054 / 2196 |
| sample rate | 16000 |
| non-finite outputs | 0 |
| all-zero outputs | 0 |
| length match / mismatch | 5122 / 128 |
| length delta range | -16 to 0 samples |
| max peak | 5144.527344 |
| min RMS | 0.000673 |

The output count and finite checks passed, but the maximum waveform peak is far
above the official baseline run (`0.650432`). This indicates output scale
instability after the short fine-tune.

## Dev Objective Metrics

Command:

```bash
/home/avse/avse_gpu_venv/bin/python eval_real.py \
  --conf_dir configs/track2_av_convtasnet.yml \
  --ckpt /home/avse/experiments/track2_chineselips_finetune_degrade0_sanity/train_500step/best_model.pth \
  --data_root /home/avse/experiments/track2_dev_official_baseline/data_root \
  --track track2 --scene both --split dev \
  --metrics objective --mode eval \
  --save_dir /home/avse/experiments/track2_chineselips_finetune_degrade0_sanity/dev_eval/enhanced \
  --out_csv /home/avse/experiments/track2_chineselips_finetune_degrade0_sanity/dev_eval/metrics/objective.csv \
  --gpus 4
```

| scope | n | SI-SDR | PESQ | STOI |
|---|---:|---:|---:|---:|
| this run, remix | 2196 | -25.7544 | 1.2107 | 0.3131 |
| official baseline, remix | 2196 | -2.848 | 1.256 | 0.470 |
| delta | -- | -22.9064 | -0.0453 | -0.1569 |

Artifacts:

- `dev_eval/enhanced/`
- `dev_eval/audio_validation.csv`
- `dev_eval/audio_validation_summary.json`
- `dev_eval/metrics/objective.csv`
- `dev_eval/metrics/objective_summary.csv`

## Interpretation

This experiment proves the following:

- the converted Chinese-LiPS training data can drive the official Track 2
  model/loss/optimizer loop from the official checkpoint;
- checkpoints saved by the fine-tuning script can be loaded by `eval_real.py`;
- full Track 2 dev enhancement can be generated from the fine-tuned checkpoint.

It does not prove that `degrade_prob=0.0` fine-tuning improves the official
baseline. The opposite happened on dev: objective quality regressed sharply and
the generated wavs show severe output-scale instability.

Full no-reference metrics such as DNSMOS, UTMOS, CER, and speaker similarity
were not run for this failed checkpoint because the objective gate already
showed a clear regression and the output amplitude is abnormal. Running the
remaining expensive metrics would not change the go/no-go conclusion.

## Next Steps

- Add a shorter learning-rate sweep before any full training attempt.
- Compare `best_model.pth` and `final_model.pth` output scale on a small dev
  subset before full dev enhancement.
- Try freezing parts of the model or using a much smaller LR.
- Re-enable Track 2-style visual degradation for robust training after the
  basic fine-tune stability issue is understood.
- Consider adding an output-amplitude validation gate immediately after
  training smoke.
