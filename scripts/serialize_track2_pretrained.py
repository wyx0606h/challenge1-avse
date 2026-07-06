#!/usr/bin/env python3
"""Serialize an official Track 2 Hub-format baseline checkpoint to .pth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from look2hear.models import AV_ConvTasNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; use --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = AV_ConvTasNet.from_pretrained(str(args.pretrained_dir))
    torch.save(model.serialize(), args.output)

    summary = {
        "pretrained_dir": str(args.pretrained_dir),
        "output": str(args.output),
        "parameters": sum(p.numel() for p in model.parameters()),
        "state_dict_keys": len(model.state_dict()),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
