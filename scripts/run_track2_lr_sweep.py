#!/usr/bin/env python3
"""Run low-LR Track 2 fine-tuning sanity sweep with dev objective gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pretrained-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", default="5")
    parser.add_argument("--lrs", default="1e-6,3e-7,1e-7")
    parser.add_argument("--steps", default="20,50,100")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--gate-limit", type=int, default=50)
    parser.add_argument("--degrade-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "sweep_config.json", serializable(vars(args)))

    results = []
    started = time.perf_counter()
    for lr in parse_float_list(args.lrs):
        for steps in parse_int_list(args.steps):
            run_name = f"lr_{format_lr(lr)}_steps_{steps}"
            run_dir = args.output_dir / "runs" / run_name
            train_dir = run_dir / "train"
            gate_dir = run_dir / f"gate_limit{args.gate_limit}"
            train_dir.mkdir(parents=True, exist_ok=True)
            gate_dir.mkdir(parents=True, exist_ok=True)

            print(f"== {run_name}: train ==", flush=True)
            train_code = run_train(args, lr, steps, train_dir)
            train_summary = read_json(train_dir / "summary.json")
            gate_code = None
            gate_summary = None

            if train_code == 0 and (train_dir / "best_model.pth").is_file():
                print(f"== {run_name}: gate ==", flush=True)
                gate_code = run_gate(args, train_dir / "best_model.pth", gate_dir)
                gate_summary = read_json(gate_dir / "gate_summary.json")
            else:
                print(f"== {run_name}: gate skipped because training failed ==", flush=True)

            result = {
                "run": run_name,
                "lr": lr,
                "steps": steps,
                "train_returncode": train_code,
                "gate_returncode": gate_code,
                "train_summary": train_summary,
                "gate_summary": gate_summary,
            }
            results.append(result)
            write_json(args.output_dir / "sweep_summary.json", {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "results": results,
            })
            print(json.dumps(compact_result(result), indent=2, ensure_ascii=False), flush=True)

    summary = {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": results,
        "best_passed": choose_best_passed(results),
    }
    write_json(args.output_dir / "sweep_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


def run_train(args: argparse.Namespace, lr: float, steps: int, train_dir: Path) -> int:
    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts/finetune_track2_sanity.py"),
        "--dataset-root",
        str(args.dataset_root),
        "--pretrained-dir",
        str(args.pretrained_dir),
        "--output-dir",
        str(train_dir),
        "--max-steps",
        str(steps),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--val-every",
        str(steps),
        "--val-batches",
        str(args.val_batches),
        "--degrade-prob",
        str(args.degrade_prob),
        "--lr",
        str(lr),
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return run_logged(cmd, train_dir / "train.log", env_cuda=args.gpus)


def run_gate(args: argparse.Namespace, ckpt: Path, gate_dir: Path) -> int:
    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts/track2_dev_objective_gate.py"),
        "--ckpt",
        str(ckpt),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(gate_dir),
        "--gpus",
        args.gpus,
        "--limit",
        str(args.gate_limit),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return run_logged(cmd, gate_dir / "gate.log", env_cuda=None)


def run_logged(cmd, log_path: Path, env_cuda: Optional[str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = None
    if env_cuda is not None:
        import os

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env_cuda
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
    return proc.returncode


def choose_best_passed(results):
    passed = []
    for result in results:
        gate = result.get("gate_summary") or {}
        if not gate.get("passed"):
            continue
        metrics = gate.get("metrics") or {}
        passed.append((metrics.get("si_sdr", float("-inf")), result))
    if not passed:
        return None
    return max(passed, key=lambda x: x[0])[1]


def compact_result(result):
    gate = result.get("gate_summary") or {}
    train = result.get("train_summary") or {}
    return {
        "run": result["run"],
        "train_returncode": result["train_returncode"],
        "gate_returncode": result["gate_returncode"],
        "last_train_loss": train.get("last_train_loss"),
        "best_val_loss": train.get("best_val_loss"),
        "gate_passed": gate.get("passed"),
        "gate_metrics": gate.get("metrics"),
        "peak_max": (gate.get("audio") or {}).get("peak_max"),
    }


def parse_float_list(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("-", "m")


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def serializable(data):
    out = {}
    for key, value in data.items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


if __name__ == "__main__":
    raise SystemExit(main())
