#!/usr/bin/env bash
set -euo pipefail

EXP=${EXP:-/home/avse/experiments/track2_chineselips_finetune_degrade0_full_dev}
REPO=${REPO:-/home/avse/workspace-goodparts}
PY38=${PY38:-/home/avse/avse_gpu_venv/bin/python}
PY311=${PY311:-/home/avse/avse_eval_py311/bin/python}
DATA_ROOT=${DATA_ROOT:-/home/avse/experiments/track2_dev_official_baseline/data_root}
CKPT=${CKPT:-$EXP/checkpoints/lr_1em06_steps_20_best_model.pth}
SAVE_DIR=${SAVE_DIR:-$EXP/full/enhanced}
METRICS_DIR=${METRICS_DIR:-$EXP/metrics}
LOG_DIR=${LOG_DIR:-$EXP/logs}
CONFIG_DIR=${CONFIG_DIR:-$EXP/config}
GPU_ENHANCE=${GPU_ENHANCE:-5}
GPU_METRICS=${GPU_METRICS:-6}
EXPECTED_ITEMS=${EXPECTED_ITEMS:-5250}
SPK_SHARDS=${SPK_SHARDS:-64}

mkdir -p "$EXP/full" "$METRICS_DIR" "$LOG_DIR" "$CONFIG_DIR"
cd "$REPO"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

is_summary_complete() {
  local summary="$1"
  [ -f "$summary" ] && awk -F, -v expected="$EXPECTED_ITEMS" \
    'NR == 2 && $1 == "overall" && $2 == expected { ok=1 } END { exit ok ? 0 : 1 }' \
    "$summary"
}

validate_enhanced() {
  "$PY38" - "$SAVE_DIR" "$DATA_ROOT" "$EXP/full/audio_validation_summary.json" "$EXP/full/audio_validation.csv" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

repo = Path("/home/avse/workspace-goodparts")
sys.path.insert(0, str(repo))
from look2hear.datas.real_test_dataset import build_items

save_dir = Path(sys.argv[1])
data_root = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
csv_path = Path(sys.argv[4])

items = build_items(root=str(data_root), tracks=("track2",), split="dev", scenes=("mix", "remix"))
rows = []
missing = []
nonfinite = 0
all_zero = 0
length_match = 0
deltas = []
sample_rates = set()
channels = set()
peak_max = 0.0
rms_min = None
scene_counts = {}

for meta in items:
    scene_counts[meta["scene"]] = scene_counts.get(meta["scene"], 0) + 1
    wav_path = save_dir / meta["track"] / meta["split"] / meta["scene"] / meta["clip_id"] / f"{meta['spk_tag']}.wav"
    if not wav_path.exists():
        missing.append(meta["key"])
        continue
    enhanced, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    source, _ = sf.read(meta["wav_path"], dtype="float32", always_2d=True)
    finite = bool(np.isfinite(enhanced).all())
    nonzero = bool(np.any(enhanced != 0))
    peak = float(np.max(np.abs(enhanced))) if enhanced.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(enhanced)))) if enhanced.size else 0.0
    delta = int(enhanced.shape[0] - source.shape[0])
    sample_rates.add(int(sr))
    channels.add(int(enhanced.shape[1]))
    nonfinite += 0 if finite else 1
    all_zero += 0 if nonzero else 1
    length_match += 1 if delta == 0 else 0
    deltas.append(delta)
    peak_max = max(peak_max, peak)
    rms_min = rms if rms_min is None else min(rms_min, rms)
    rows.append({
        "key": meta["key"],
        "scene": meta["scene"],
        "sample_rate": int(sr),
        "channels": int(enhanced.shape[1]),
        "samples": int(enhanced.shape[0]),
        "input_samples": int(source.shape[0]),
        "length_delta": delta,
        "finite": finite,
        "nonzero": nonzero,
        "peak": peak,
        "rms": rms,
    })

summary = {
    "expected": len(items),
    "produced": len(rows),
    "missing": len(missing),
    "mix": scene_counts.get("mix", 0),
    "remix": scene_counts.get("remix", 0),
    "sample_rates": sorted(sample_rates),
    "channels": sorted(channels),
    "nonfinite": nonfinite,
    "all_zero": all_zero,
    "length_match": length_match,
    "length_mismatch": len(deltas) - length_match,
    "length_delta_min": min(deltas) if deltas else None,
    "length_delta_max": max(deltas) if deltas else None,
    "peak_max": peak_max,
    "rms_min": rms_min,
    "missing_keys": missing[:20],
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "key", "scene", "sample_rate", "channels", "samples",
        "input_samples", "length_delta", "finite", "nonzero", "peak", "rms",
    ])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(summary, ensure_ascii=False), flush=True)
sys.exit(0 if summary["produced"] == summary["expected"] and summary["missing"] == 0 and summary["nonfinite"] == 0 and summary["all_zero"] == 0 else 2)
PY
}

save_config() {
  git rev-parse HEAD > "$CONFIG_DIR/git-commit.txt"
  git status --short --branch > "$CONFIG_DIR/git-status.txt"
  cp configs/track2_av_convtasnet.yml "$CONFIG_DIR/track2_av_convtasnet.yml"
  "$PY38" -V > "$CONFIG_DIR/python38-version.txt" 2>&1
  "$PY311" -V > "$CONFIG_DIR/python311-version.txt" 2>&1
  "$PY38" -m pip freeze > "$CONFIG_DIR/pip-freeze-py38.txt"
  "$PY311" -m pip freeze > "$CONFIG_DIR/pip-freeze-py311.txt"
}

run_enhance() {
  if validate_enhanced; then
    log "enhanced wavs already complete: $SAVE_DIR"
    return
  fi
  log "running full Track 2 dev enhancement with $CKPT"
  CUDA_VISIBLE_DEVICES="$GPU_ENHANCE" "$PY38" eval_real.py \
    --conf_dir configs/track2_av_convtasnet.yml \
    --ckpt "$CKPT" \
    --data_root "$DATA_ROOT" \
    --track track2 --scene both --split dev \
    --metrics none --mode enhance \
    --save_dir "$SAVE_DIR" \
    --gpus "$GPU_ENHANCE" \
    2>&1 | tee "$LOG_DIR/full_enhance.log"
  validate_enhanced
}

run_sharded_metric() {
  local metric="$1"
  local shards="$2"
  local py="$3"
  local out_csv="$METRICS_DIR/${metric}.csv"
  local summary="$METRICS_DIR/${metric}_summary.csv"

  if is_summary_complete "$summary"; then
    log "[skip] $metric already complete: $summary"
    return
  fi

  log "running $metric in $shards shards"
  for shard in $(seq 0 $((shards - 1))); do
    local shard_stem="${out_csv%.csv}.shard${shard}of${shards}"
    local shard_csv="${shard_stem}.csv"
    local shard_summary="${shard_stem}_summary.csv"
    if [ -f "$shard_csv" ] && [ -f "$shard_summary" ]; then
      log "[skip] $metric shard $shard exists: $shard_csv"
      continue
    fi
    DNSMOS_DIR=/home/avse/avse-assets/evaluation/dnsmos \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 \
        CUDA_VISIBLE_DEVICES="$GPU_METRICS" "$py" eval_real.py \
        --conf_dir configs/track2_av_convtasnet.yml \
        --data_root "$DATA_ROOT" \
        --track track2 --scene both --split dev \
        --metrics "$metric" --mode eval \
        --save_dir "$SAVE_DIR" \
        --out_csv "$out_csv" \
        --num_shards "$shards" --shard_id "$shard" \
        --gpus "$GPU_METRICS" \
        --wespeaker_ckpt /home/avse/avse-assets/evaluation/wespeaker/cnceleb-resnet34-LM/model_5.pt \
        --enroll_ckpt /home/avse/experiments/track2_dev_official_baseline/metrics/enroll_dev.pt \
        --funasr_model /home/avse/avse-assets/evaluation/funasr/Fun-ASR-Nano-2512 \
        --funasr_remote_code Fun-ASR/model.py \
        2>&1 | tee "$LOG_DIR/${metric}_shard_${shard}.log"
  done

  log "merging $metric shards"
  CUDA_VISIBLE_DEVICES="$GPU_METRICS" "$py" eval_real.py \
    --conf_dir configs/track2_av_convtasnet.yml \
    --data_root "$DATA_ROOT" \
    --track track2 --scene both --split dev \
    --metrics "$metric" --mode eval \
    --save_dir "$SAVE_DIR" \
    --out_csv "$out_csv" \
    --num_shards "$shards" --merge_shards \
    --gpus "$GPU_METRICS" \
    2>&1 | tee "$LOG_DIR/${metric}_merge.log"
}

run_single_metric() {
  local metric="$1"
  local py="$2"
  local out_csv="$METRICS_DIR/${metric}.csv"
  local summary="$METRICS_DIR/${metric}_summary.csv"

  if is_summary_complete "$summary"; then
    log "[skip] $metric already complete: $summary"
    return
  fi

  log "running $metric"
  DNSMOS_DIR=/home/avse/avse-assets/evaluation/dnsmos \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODELSCOPE_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES="$GPU_METRICS" "$py" eval_real.py \
      --conf_dir configs/track2_av_convtasnet.yml \
      --data_root "$DATA_ROOT" \
      --track track2 --scene both --split dev \
      --metrics "$metric" --mode eval \
      --save_dir "$SAVE_DIR" \
      --out_csv "$out_csv" \
      --gpus "$GPU_METRICS" \
      --wespeaker_ckpt /home/avse/avse-assets/evaluation/wespeaker/cnceleb-resnet34-LM/model_5.pt \
      --enroll_ckpt /home/avse/experiments/track2_dev_official_baseline/metrics/enroll_dev.pt \
      --funasr_model /home/avse/avse-assets/evaluation/funasr/Fun-ASR-Nano-2512 \
      --funasr_remote_code Fun-ASR/model.py \
      2>&1 | tee "$LOG_DIR/${metric}.log"
}

{
  date
  log "experiment: $EXP"
  log "checkpoint: $CKPT"
  save_config
  run_enhance
  run_sharded_metric objective 4 "$PY38"
  run_sharded_metric dnsmos 16 "$PY38"
  run_sharded_metric spk "$SPK_SHARDS" "$PY38"
  run_single_metric utmos "$PY311"
  run_single_metric asr "$PY311"
  date
  log "full dev metrics complete"
} 2>&1 | tee -a "$LOG_DIR/run_all_metrics.log"
