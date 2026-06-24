"""Precompute per-speaker enrollment voiceprints for the speaker-similarity metric.

Speaker similarity (spk-sim) compares the enhanced speech against a per-speaker
*enrollment* voiceprint. Building those voiceprints means running WeSpeaker over
thousands of clean single-speaker clips (the remix scene's s1.wav/s2.wav), which
is slow and identical across eval runs. This script does it once and caches the
result to a small ``.pt`` file:

    {speaker_id: mean_embedding[256]}   e.g. {"A_L": tensor, "G_R": tensor, ...}

eval_real.py then loads it via ``--enroll_ckpt`` and skips the rebuild.

Typical use:
    # dev voiceprints (speakers A/G/R) -- can be released with the baseline so
    # participants score spk-sim on dev without re-extracting:
    python build_enrollment.py --split dev --out pretrained/enroll_dev.pt

    # test voiceprints (speakers B/D/E/...) -- organizer-side only, NOT released:
    python build_enrollment.py --split test --out pretrained/enroll_test.pt

Then:
    python eval_real.py --split dev --metrics spk --enroll_ckpt pretrained/enroll_dev.pt

Requires the WeSpeaker checkpoint (download_models.sh). Runs offline.
"""

import os
import argparse

import torch

from look2hear.datas.real_test_dataset import REAL_AVSE_ROOT
from look2hear.metrics.real_metrics import RealMetricsTracker
from eval_real import build_enrollment


def parse_args():
    p = argparse.ArgumentParser(description="Precompute enrollment voiceprints")
    p.add_argument("--data_root", default=REAL_AVSE_ROOT,
                   help="REAL-AVSE corpus root (holds track1/ and track2/)")
    p.add_argument("--track", default="both", choices=["track1", "track2", "both"])
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--wespeaker_ckpt",
                   default="pretrained/wespeaker_cnceleb_resnet34/model_5.pt")
    p.add_argument("--out", required=True,
                   help="Output .pt path for the {speaker_id: emb} voiceprints")
    p.add_argument("--gpus", default="9", help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap clean sources per (track) (smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tracks = ("track1", "track2") if args.track == "both" else (args.track,)

    # A spk-only tracker just to expose the WeSpeaker extractor.
    tracker = RealMetricsTracker(metrics=["spk"], device=device,
                                 wespeaker_ckpt=args.wespeaker_ckpt)

    print(f"Building enrollment for split={args.split}, tracks={tracks} -> {args.out}")
    build_enrollment(tracker, args.data_root, tracks, args.split, device,
                     limit=args.limit, save_path=args.out)
    print("Done.")


if __name__ == "__main__":
    main()
