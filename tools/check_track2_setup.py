#!/usr/bin/env python3
"""Validate a Real-World AVSE Track 2 checkout before training/evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DEV_COUNTS = {"mix": 1527, "remix": 1098}
COMMON_FILES = (
    "mix.wav",
    "s1.mp4",
    "s2.mp4",
    "s1.pkl",
    "s2.pkl",
    "s1.txt",
    "s2.txt",
)
REMIX_ONLY_FILES = ("s1.wav", "s2.wav")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    successes: list[str] = field(default_factory=list)

    def ok(self, message: str) -> None:
        self.successes.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def print(self) -> None:
        for message in self.successes:
            print(f"[ OK ] {message}")
        for message in self.warnings:
            print(f"[WARN] {message}")
        for message in self.errors:
            print(f"[FAIL] {message}")
        print(
            f"\nSummary: {len(self.successes)} passed, "
            f"{len(self.warnings)} warnings, {len(self.errors)} failed."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the Track 2 development set, Python/CUDA environment, and "
            "optional training assets used by the official baseline."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Dataset root containing track2/, or its parent containing Real-World-AVSE/.",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check PyTorch, CUDA, audio/video, and evaluation dependencies.",
    )
    parser.add_argument(
        "--check-training-assets",
        action="store_true",
        help="Check VoxCeleb2 manifests and the lip-reading initialization weight.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Accept a non-empty dev subset instead of enforcing official sample "
            "counts. Intended only for fixtures and local pipeline debugging."
        ),
    )
    args = parser.parse_args()
    if not (args.data_root or args.check_env or args.check_training_assets):
        parser.print_help()
        raise SystemExit(0)
    return args


def resolve_data_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    candidates = (root, root / "Real-World-AVSE", root / "REAL-AVSE")
    for candidate in candidates:
        if (candidate / "track2").is_dir():
            return candidate
    return root


def load_manifest(path: Path, report: Report) -> list[dict]:
    if not path.is_file():
        report.fail(f"Missing manifest: {path}")
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.fail(f"Cannot read manifest {path}: {exc}")
        return []
    if not isinstance(data, list):
        report.fail(f"Manifest must contain a JSON list: {path}")
        return []
    invalid = [index for index, entry in enumerate(data) if not isinstance(entry, dict)]
    if invalid:
        report.fail(f"{path} contains non-object entries at indices {invalid[:5]}")
        return []
    return data


def missing_files(directory: Path, names: Iterable[str]) -> list[str]:
    missing = []
    for name in names:
        path = directory / name
        if not path.is_file():
            missing.append(name)
        elif path.suffix.lower() != ".txt" and path.stat().st_size == 0:
            missing.append(f"{name} (empty)")
    return missing


def check_scene(
    dev_root: Path,
    scene: str,
    allow_partial: bool,
    report: Report,
) -> None:
    manifest_path = dev_root / f"{scene}_manifest.json"
    entries = load_manifest(manifest_path, report)
    if not entries:
        report.fail(f"No usable entries found for Track 2 dev/{scene}.")
        return

    expected = EXPECTED_DEV_COUNTS[scene]
    if allow_partial:
        report.ok(f"dev/{scene}: {len(entries)} manifest entries (partial mode)")
    elif len(entries) != expected:
        report.fail(
            f"dev/{scene}: expected {expected} entries, found {len(entries)}. "
            "Re-download/extract the official development set, or use "
            "--allow-partial only for an intentional subset."
        )
    else:
        report.ok(f"dev/{scene}: official count verified ({expected})")

    seen: set[str] = set()
    bad_entries = 0
    for index, entry in enumerate(entries):
        clip_id = entry.get("mix_id")
        if not isinstance(clip_id, str) or not clip_id:
            report.fail(f"{manifest_path}: entry {index} has no valid mix_id")
            bad_entries += 1
            continue
        if clip_id in seen:
            report.fail(f"{manifest_path}: duplicate mix_id {clip_id!r}")
            bad_entries += 1
            continue
        seen.add(clip_id)

        required = COMMON_FILES + (REMIX_ONLY_FILES if scene == "remix" else ())
        absent = missing_files(dev_root / scene / clip_id, required)
        if absent:
            report.fail(
                f"dev/{scene}/{clip_id}: missing or empty {', '.join(absent)}"
            )
            bad_entries += 1

    if bad_entries == 0:
        report.ok(f"dev/{scene}: all referenced clip assets are present")


def check_data(path: Path, allow_partial: bool, report: Report) -> None:
    root = resolve_data_root(path)
    dev_root = root / "track2" / "dev"
    if not dev_root.is_dir():
        report.fail(
            f"Track 2 dev directory not found at {dev_root}. Register at "
            "https://forms.gle/GYL1SHRpAdVPNanaA and extract the organizer package "
            "so DATA_ROOT/track2/dev exists."
        )
        return
    report.ok(f"Resolved challenge dataset root: {root}")
    for scene in ("mix", "remix"):
        check_scene(dev_root, scene, allow_partial, report)


def module_version(module: object) -> str:
    return str(getattr(module, "__version__", "version unknown"))


def check_environment(report: Report) -> None:
    modules = {
        "torch": "PyTorch",
        "torchaudio": "torchaudio",
        "torchvision": "torchvision",
        "pytorch_lightning": "PyTorch Lightning",
        "soundfile": "SoundFile",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "yaml": "PyYAML",
        "torchmetrics": "torchmetrics",
        "librosa": "librosa",
        "onnxruntime": "ONNX Runtime",
        "funasr": "FunASR",
        "modelscope": "ModelScope",
        "transformers": "Transformers",
    }
    imported: dict[str, object] = {}
    for name, label in modules.items():
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            report.fail(f"{label} import failed: {exc}")
        else:
            imported[name] = module
            report.ok(f"{label}: {module_version(module)}")

    torch = imported.get("torch")
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        if not cuda_available:
            report.fail(
                "PyTorch cannot access CUDA. GPU inference/training requires a "
                "compatible NVIDIA driver and the CUDA-enabled PyTorch build."
            )
        else:
            count = int(torch.cuda.device_count())
            names = [torch.cuda.get_device_name(index) for index in range(count)]
            report.ok(f"CUDA available: {count} device(s): {', '.join(names)}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        report.ok(f"ffmpeg found: {ffmpeg}")
    else:
        report.warn(
            "ffmpeg was not found on PATH. OpenCV may still decode MP4 files, "
            "but installing ffmpeg makes media debugging easier."
        )


def check_json_list(path: Path, label: str, report: Report) -> None:
    if not path.is_file():
        report.fail(f"Missing {label}: {path}")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.fail(f"Cannot read {label} {path}: {exc}")
        return
    if not isinstance(data, list) or not data:
        report.fail(f"{label} must be a non-empty JSON list: {path}")
    else:
        report.ok(f"{label}: {len(data)} entries")


def check_training_assets(report: Report) -> None:
    manifest_root = PROJECT_ROOT / "DataPreProcess" / "vox2"
    for split in ("tr", "cv", "tt"):
        for name in ("mix", "s1", "s2"):
            check_json_list(
                manifest_root / split / f"{name}.json",
                f"VoxCeleb2 {split}/{name} manifest",
                report,
            )

    video_weight = (
        PROJECT_ROOT / "video_pretrain" / "frcnn_128_512.backbone.pth.tar"
    )
    if not video_weight.is_file() or video_weight.stat().st_size == 0:
        report.fail(
            f"Missing lip-reading initialization weight: {video_weight}. "
            "Download the training-only backbone linked in the official README."
        )
    else:
        report.ok(
            f"Lip-reading initialization weight: "
            f"{video_weight.stat().st_size / (1024 ** 2):.1f} MiB"
        )


def main() -> int:
    args = parse_args()
    report = Report()

    if args.data_root:
        check_data(args.data_root, args.allow_partial, report)
    if args.check_env:
        check_environment(report)
    if args.check_training_assets:
        check_training_assets(report)

    report.print()
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
