#!/usr/bin/env python3
"""Fixed-batch Track 2 overfit runner for the smoke dataset."""

import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader

import look2hear.losses
import look2hear.models
from look2hear.datas.track2_datasets import Track2StaticDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/track2_smoke_overfit.yml",
        help="Path to the overfit YAML config.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override step count.")
    parser.add_argument("--device", default=None, help="Override device: auto/cpu/cuda.")
    parser.add_argument(
        "--sample_index",
        type=int,
        default=None,
        help="Zero-based DataLoader batch index to use as the first fixed batch.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override artifact directory outside the Git repository.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def ensure_finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf.")


def make_loss(config):
    loss_name = config["loss"]["loss_func"]
    sdr_name = config["loss"]["sdr_type"]
    loss_cls = getattr(look2hear.losses, loss_name)
    sdr_func = getattr(look2hear.losses, sdr_name)
    return loss_cls(sdr_func, **config["loss"].get("config", {}))


def batch_to_device(batch, device):
    mixture, target, mouth, filename = batch
    return (
        mixture.to(device=device, dtype=torch.float32),
        target.to(device=device, dtype=torch.float32),
        mouth.to(device=device, dtype=torch.float32),
        filename,
    )


def output_distance(estimate, target):
    return torch.mean((estimate - target) ** 2)


def write_wav(path, audio, sample_rate):
    audio_np = audio.detach().cpu().float().numpy()
    sf.write(path, audio_np, sample_rate)


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.steps is not None:
        config["steps"] = args.steps
    if args.device is not None:
        config["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.sample_index is not None:
        config["sample_index"] = args.sample_index

    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.config, output_dir / "config.yml")
    with open(output_dir / "resolved_config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    dataset = Track2StaticDataset(
        json_dir=config["data"]["train_dir"],
        n_src=1,
        sample_rate=int(config["data"]["sample_rate"]),
        segment=float(config["data"]["segment"]),
        normalize_audio=bool(config["data"]["normalize_audio"]),
        is_train=False,
        face_size=int(config["data"]["face_size"]),
        degrade_prob=float(config["data"]["degrade_prob"]),
        degrade_state="test",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        drop_last=True,
    )

    fixed_batches = []
    sample_index = int(config.get("sample_index", 0))
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")
    for batch_index, batch in enumerate(loader):
        if batch_index < sample_index:
            continue
        fixed_batches.append(batch_to_device(batch, device))
        if len(fixed_batches) >= int(config["fixed_batches"]):
            break
    if not fixed_batches:
        raise RuntimeError("No fixed overfit batch could be loaded.")

    video_cfg = config.get("videonet", {})
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=int(config["data"]["sample_rate"]),
        video_relu_type=video_cfg.get("relu_type", "prelu"),
        video_pretrain=video_cfg.get("pretrain"),
        **config["audionet"]["audionet_config"],
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["learning_rate"]),
    )
    loss_func = make_loss(config).to(device)

    sample_rate = int(config["data"]["sample_rate"])
    first_mix, first_target, first_mouth, first_name = fixed_batches[0]
    with torch.no_grad():
        initial_output = model(first_mix, first_mouth)
        ensure_finite("initial_output", initial_output)
        initial_loss = loss_func(initial_output, first_target)
        ensure_finite("initial_loss", initial_loss)
        initial_distance = output_distance(initial_output, first_target)
        ensure_finite("initial_distance", initial_distance)

    write_wav(output_dir / "initial_output.wav", initial_output[0], sample_rate)
    write_wav(output_dir / "target.wav", first_target[0], sample_rate)

    rows = []
    started = time.time()
    for step in range(1, int(config["steps"]) + 1):
        mixture, target, mouth, _ = fixed_batches[(step - 1) % len(fixed_batches)]
        optimizer.zero_grad(set_to_none=True)
        estimate = model(mixture, mouth)
        ensure_finite("estimate", estimate)
        loss = loss_func(estimate, target)
        ensure_finite("loss", loss)
        loss.backward()

        grad_norm_sq = torch.zeros((), device=device)
        for param in model.parameters():
            if param.grad is not None:
                ensure_finite("gradient", param.grad)
                grad_norm_sq = grad_norm_sq + param.grad.detach().pow(2).sum()
        grad_norm = torch.sqrt(grad_norm_sq)
        ensure_finite("grad_norm", grad_norm)

        if float(config.get("grad_clip", 0.0)) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()

        if step == 1 or step % int(config["log_interval"]) == 0 or step == int(config["steps"]):
            distance = output_distance(estimate.detach(), target)
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "mse_to_target": float(distance.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
            }
            rows.append(row)
            print(
                "step={step} loss={loss:.6f} mse={mse_to_target:.8f} "
                "grad_norm={grad_norm:.6f}".format(**row),
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        final_output = model(first_mix, first_mouth)
        ensure_finite("final_output", final_output)
        final_loss = loss_func(final_output, first_target)
        ensure_finite("final_loss", final_loss)
        final_distance = output_distance(final_output, first_target)
        ensure_finite("final_distance", final_distance)

    write_wav(output_dir / "final_output.wav", final_output[0], sample_rate)

    csv_path = output_dir / "loss.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "mse_to_target", "grad_norm"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "completed",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": int(config["seed"]),
        "steps": int(config["steps"]),
        "batch_size": int(config["batch_size"]),
        "fixed_batches": int(config["fixed_batches"]),
        "sample_index": sample_index,
        "degrade_prob": float(config["data"]["degrade_prob"]),
        "first_filename": list(first_name),
        "initial_loss": float(initial_loss.detach().cpu()),
        "final_loss": float(final_loss.detach().cpu()),
        "initial_mse_to_target": float(initial_distance.detach().cpu()),
        "final_mse_to_target": float(final_distance.detach().cpu()),
        "loss_delta": float((initial_loss - final_loss).detach().cpu()),
        "mse_delta": float((initial_distance - final_distance).detach().cpu()),
        "elapsed_seconds": time.time() - started,
        "artifacts": {
            "config": str(output_dir / "config.yml"),
            "resolved_config": str(output_dir / "resolved_config.json"),
            "loss_csv": str(csv_path),
            "initial_output_wav": str(output_dir / "initial_output.wav"),
            "final_output_wav": str(output_dir / "final_output.wav"),
            "target_wav": str(output_dir / "target.wav"),
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    report = "\n".join(
        [
            "# Baseline Smoke Overfit",
            "",
            f"- Status: {summary['status']}",
            f"- Device: {summary['device']}",
            f"- Steps: {summary['steps']}",
            f"- Initial loss: {summary['initial_loss']:.6f}",
            f"- Final loss: {summary['final_loss']:.6f}",
            f"- Initial MSE to target: {summary['initial_mse_to_target']:.8f}",
            f"- Final MSE to target: {summary['final_mse_to_target']:.8f}",
            f"- Loss delta: {summary['loss_delta']:.6f}",
            f"- MSE delta: {summary['mse_delta']:.8f}",
            f"- Artifacts: {output_dir}",
            "",
        ]
    )
    with open(output_dir / "report.md", "w") as f:
        f.write(report)

    if not summary["final_loss"] < summary["initial_loss"]:
        raise RuntimeError("Final loss did not improve over initial loss.")
    if not summary["final_mse_to_target"] < summary["initial_mse_to_target"]:
        raise RuntimeError("Final output MSE did not improve over initial output MSE.")


if __name__ == "__main__":
    main()
