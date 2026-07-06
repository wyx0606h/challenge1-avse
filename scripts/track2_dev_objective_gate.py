#!/usr/bin/env python3
"""Run a small Track 2 dev objective gate for a local checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from look2hear.datas.real_test_dataset import build_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf-dir", type=Path, default=REPO_ROOT / "configs/track2_av_convtasnet.yml")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", default="4")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--track", default="track2", choices=["track1", "track2"])
    parser.add_argument("--scene", default="remix", choices=["mix", "remix", "both"])
    parser.add_argument("--baseline-si-sdr", type=float, default=-2.848)
    parser.add_argument("--min-si-sdr-margin", type=float, default=2.0)
    parser.add_argument("--min-stoi", type=float, default=0.430)
    parser.add_argument("--max-peak", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-enhance", action="store_true")
    parser.add_argument("--skip-objective", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)
    (args.output_dir / "metrics").mkdir(exist_ok=True)

    enhanced_dir = args.output_dir / "enhanced"
    objective_csv = args.output_dir / "metrics/objective.csv"
    run_config = vars(args).copy()
    for key, value in list(run_config.items()):
        if isinstance(value, Path):
            run_config[key] = str(value)
    write_json(args.output_dir / "gate_config.json", run_config)

    if not args.skip_enhance:
        run_eval_real(
            args,
            mode="enhance",
            metrics="none",
            save_dir=enhanced_dir,
            out_csv=None,
            log_path=args.output_dir / "logs/enhance.log",
        )

    audio_summary = validate_audio(
        data_root=args.data_root,
        save_dir=enhanced_dir,
        output_dir=args.output_dir,
        track=args.track,
        scene=args.scene,
        split=args.split,
        limit=args.limit,
    )

    if not args.skip_objective:
        run_eval_real(
            args,
            mode="eval",
            metrics="objective",
            save_dir=enhanced_dir,
            out_csv=objective_csv,
            log_path=args.output_dir / "logs/objective.log",
        )

    objective_summary = read_objective_summary(objective_csv.with_name("objective_summary.csv"))
    gate_summary = build_gate_summary(args, audio_summary, objective_summary)
    write_json(args.output_dir / "gate_summary.json", gate_summary)
    print(json.dumps(gate_summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if gate_summary["passed"] else 2


def run_eval_real(
    args: argparse.Namespace,
    *,
    mode: str,
    metrics: str,
    save_dir: Path,
    out_csv: Optional[Path],
    log_path: Path,
) -> None:
    cmd = [
        str(args.python),
        str(REPO_ROOT / "eval_real.py"),
        "--conf_dir",
        str(args.conf_dir),
        "--ckpt",
        str(args.ckpt),
        "--data_root",
        str(args.data_root),
        "--track",
        args.track,
        "--scene",
        args.scene,
        "--split",
        args.split,
        "--metrics",
        metrics,
        "--mode",
        mode,
        "--save_dir",
        str(save_dir),
        "--gpus",
        args.gpus,
        "--limit",
        str(args.limit),
    ]
    if out_csv is not None:
        cmd.extend(["--out_csv", str(out_csv)])

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{mode} command failed with code {proc.returncode}; see {log_path}")


def validate_audio(
    *,
    data_root: Path,
    save_dir: Path,
    output_dir: Path,
    track: str,
    scene: str,
    split: str,
    limit: int,
) -> dict:
    scenes = ("mix", "remix") if scene == "both" else (scene,)
    items = build_items(root=str(data_root), tracks=(track,), split=split, scenes=scenes, limit=limit)
    rows = []
    missing = []
    sample_rates = set()
    channels = set()
    nonfinite = 0
    all_zero = 0
    length_match = 0
    deltas = []
    peak_max = 0.0
    rms_min = None
    scene_counts: Dict[str, int] = {}

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
            "sample_rate": sr,
            "channels": enhanced.shape[1],
            "samples": enhanced.shape[0],
            "input_samples": source.shape[0],
            "length_delta": delta,
            "finite": finite,
            "nonzero": nonzero,
            "peak": peak,
            "rms": rms,
        })

    csv_path = output_dir / "audio_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "key", "scene", "sample_rate", "channels", "samples",
            "input_samples", "length_delta", "finite", "nonzero", "peak", "rms",
        ])
        writer.writeheader()
        writer.writerows(rows)

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
    write_json(output_dir / "audio_validation_summary.json", summary)
    return summary


def read_objective_summary(path: Path) -> dict:
    if not path.exists():
        return {"summary_path": str(path), "missing": True}
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["scope"]] = row
    return {"summary_path": str(path), "missing": False, "rows": rows}


def build_gate_summary(args: argparse.Namespace, audio: dict, objective: dict) -> dict:
    target_scope = f"{args.track}/remix"
    rows = objective.get("rows", {})
    metrics = rows.get(target_scope) or rows.get("overall") or {}
    si_sdr = parse_float(metrics.get("si_sdr"))
    stoi = parse_float(metrics.get("stoi"))
    pesq = parse_float(metrics.get("pesq"))
    min_si_sdr = args.baseline_si_sdr - args.min_si_sdr_margin

    checks = {
        "all_expected_outputs_present": audio["produced"] == audio["expected"] and audio["missing"] == 0,
        "finite_outputs": audio["nonfinite"] == 0,
        "nonzero_outputs": audio["all_zero"] == 0,
        "peak_within_limit": audio["peak_max"] <= args.max_peak,
        "objective_available": si_sdr is not None and stoi is not None,
        "si_sdr_within_margin": si_sdr is not None and si_sdr >= min_si_sdr,
        "stoi_above_min": stoi is not None and stoi >= args.min_stoi,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "max_peak": args.max_peak,
            "min_si_sdr": min_si_sdr,
            "min_stoi": args.min_stoi,
        },
        "metrics": {
            "si_sdr": si_sdr,
            "pesq": pesq,
            "stoi": stoi,
        },
        "audio": audio,
        "objective": objective,
    }


def parse_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
