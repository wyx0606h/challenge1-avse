#!/usr/bin/env python3
"""Short Track 2 baseline fine-tuning sanity run."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from look2hear.datas.track2_datasets import Track2DynamicDataset, Track2StaticDataset
from look2hear.losses.sdr import PITLossWrapper, pairwise_neg_sisdr, pairwise_neg_snr
from look2hear.models import AV_ConvTasNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pretrained-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--degrade-prob", type=float, default=0.0)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; use --overwrite to replace run logs")
    run(args)
    return 0


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    config = vars(args).copy()
    config["dataset_root"] = str(args.dataset_root)
    config["pretrained_dir"] = str(args.pretrained_dir)
    config["output_dir"] = str(args.output_dir)
    config["resolved_device"] = str(device)
    write_json(args.output_dir / "run_config.json", config)
    (args.output_dir / "run_config.yml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    train = Track2DynamicDataset(
        json_dir=str(args.dataset_root / "tr"),
        n_src=1,
        sample_rate=16000,
        segment=2.0,
        normalize_audio=False,
        is_train=True,
        face_size=96,
        degrade_prob=args.degrade_prob,
        degrade_state="train",
    )
    cv = Track2StaticDataset(
        json_dir=str(args.dataset_root / "cv"),
        n_src=1,
        sample_rate=16000,
        segment=2.0,
        normalize_audio=False,
        is_train=False,
        face_size=96,
        degrade_prob=args.degrade_prob,
        degrade_state="test",
    )
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        cv,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = AV_ConvTasNet.from_pretrained(str(args.pretrained_dir)).to(device)
    model.train()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_loss_func = PITLossWrapper(pairwise_neg_snr, pit_from="pw_mtx", threshold_byloss=False)
    val_loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx", threshold_byloss=False)

    fields = ["step", "phase", "loss", "lr", "elapsed_seconds"]
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

    best_val = None
    started = time.perf_counter()
    train_iter = iter(train_loader)
    last_train_loss = None
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        mixtures, targets, mouths, _ = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(mixtures, mouths)
        if not torch.isfinite(outputs).all():
            raise RuntimeError(f"non-finite model output at step {step}")
        loss = train_loss_func(outputs, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite train loss at step {step}: {loss}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite grad norm at step {step}: {grad_norm}")
        optimizer.step()
        last_train_loss = float(loss.detach().cpu().item())

        append_metric(args.output_dir / "metrics.csv", fields, {
            "step": step,
            "phase": "train",
            "loss": last_train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        if step == 1 or step % 10 == 0:
            print(f"step={step} train_loss={last_train_loss:.6f}", flush=True)

        if step == args.max_steps or step % args.val_every == 0:
            val_loss = validate(model, val_loader, val_loss_func, device, args.val_batches)
            append_metric(args.output_dir / "metrics.csv", fields, {
                "step": step,
                "phase": "val",
                "loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })
            print(f"step={step} val_loss={val_loss:.6f}", flush=True)
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                torch.save(model.cpu().serialize(), args.output_dir / "best_model.pth")
                model.to(device)

    torch.save(model.cpu().serialize(), args.output_dir / "final_model.pth")
    summary = {
        "all_ok": True,
        "max_steps": args.max_steps,
        "last_train_loss": last_train_loss,
        "best_val_loss": best_val,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "best_model": str(args.output_dir / "best_model.pth"),
        "final_model": str(args.output_dir / "final_model.pth"),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


@torch.no_grad()
def validate(model, loader, loss_func, device, limit_batches: int) -> float:
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if idx >= limit_batches:
            break
        mixtures, targets, mouths, _ = move_batch(batch, device)
        outputs = model(mixtures, mouths)
        if not torch.isfinite(outputs).all():
            raise RuntimeError(f"non-finite validation output at batch {idx}")
        loss = loss_func(outputs, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite validation loss at batch {idx}")
        losses.append(float(loss.detach().cpu().item()))
    model.train()
    if not losses:
        raise RuntimeError("validation produced no batches")
    return float(sum(losses) / len(losses))


def move_batch(batch, device):
    mixtures, targets, mouths, filenames = batch
    return mixtures.to(device), targets.to(device), mouths.to(device), filenames


def append_metric(path: Path, fields: list[str], row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow(row)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
