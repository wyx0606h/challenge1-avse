"""Build VoxCeleb2 JSON manifests for the audio-visual enhancement datasets.

For each split (``tr`` / ``cv`` / ``tt``) this writes three files under
``<out_dir>/<split>/``:

* ``mix.json``  -- ``[[mix_wav_path, n_samples], ...]``
* ``s1.json``   -- ``[[s1_wav_path, s1_visual_path, n_samples], ...]``
* ``s2.json``   -- ``[[s2_wav_path, s2_visual_path, n_samples], ...]``

These are consumed by ``look2hear.datas.Vox2DataModule``. The visual path is a
``<clip>.mp4`` whole-face video (default) or a ``<clip>.npz`` mouth ROI,
depending on ``--in_visual_dir`` / ``--visual_ext``.

Mixture filenames encode both speakers' source clips, e.g.::

    id00015_7UUbvO6HjFs_00035_1.6995_id08041__Kr0g55wbIA_00284_-1.6995.wav
    \________ s1 clip ________/\_____ s2 clip with SNR ______/

The s1 clip id is ``<spk>_<vid>_<utt>`` taken before the first SNR token, and the
s2 clip id is the same triple after it. Each maps to ``<visual_dir>/<clip><ext>``.
"""

import os
import re
import json
import argparse

import soundfile as sf
from tqdm import tqdm

# A clip id is ``id#####_<11-char-video>_#####`` (video id may contain leading
# underscores, e.g. ``_Kr0g55wbIA``).
CLIP_RE = re.compile(r"(id\d+_.{11}_\d+)")


def clip_ids(mix_stem):
    """Return ``(s1_clip, s2_clip)`` parsed from a mixture filename stem."""
    found = CLIP_RE.findall(mix_stem)
    if len(found) != 2:
        raise ValueError(
            f"expected 2 clip ids in {mix_stem!r}, found {len(found)}: {found}"
        )
    return found[0], found[1]


def build_split(audio_root, visual_dir, visual_ext, out_dir, split):
    """Write ``mix/s1/s2`` JSON manifests for one split."""
    mix_dir = os.path.join(audio_root, split, "mix")
    s1_dir = os.path.join(audio_root, split, "s1")
    s2_dir = os.path.join(audio_root, split, "s2")

    mix_files = sorted(f for f in os.listdir(mix_dir) if f.endswith(".wav"))

    mix_infos, s1_infos, s2_infos = [], [], []
    skipped = 0
    for fn in tqdm(mix_files, desc=split):
        stem = fn[:-4]
        try:
            c1, c2 = clip_ids(stem)
        except ValueError:
            skipped += 1
            continue

        mix_path = os.path.join(mix_dir, fn)
        s1_path = os.path.join(s1_dir, fn)
        s2_path = os.path.join(s2_dir, fn)
        v1_path = os.path.join(visual_dir, c1 + visual_ext)
        v2_path = os.path.join(visual_dir, c2 + visual_ext)

        if not (
            os.path.isfile(s1_path) and os.path.isfile(s2_path)
            and os.path.isfile(v1_path) and os.path.isfile(v2_path)
        ):
            skipped += 1
            continue

        n = len(sf.SoundFile(mix_path))
        mix_infos.append([mix_path, n])
        s1_infos.append([s1_path, v1_path, n])
        s2_infos.append([s2_path, v2_path, n])

    out_split = os.path.join(out_dir, split)
    os.makedirs(out_split, exist_ok=True)
    for name, infos in [("mix", mix_infos), ("s1", s1_infos), ("s2", s2_infos)]:
        with open(os.path.join(out_split, name + ".json"), "w") as f:
            json.dump(infos, f)
    print(f"[{split}] wrote {len(mix_infos)} mixtures, skipped {skipped}")


def main(args):
    for split in ["tr", "cv", "tt"]:
        build_split(
            args.in_audio_dir, args.in_visual_dir, args.visual_ext,
            args.out_dir, split,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("VoxCeleb2 manifest builder")
    parser.add_argument(
        "--in_audio_dir",
        type=str,
        default="/gpfs-flash/hulab/public_datasets/audio_datasets/vox2/audio_10w/wav16k/min",
        help="Root with tr/cv/tt -> mix/s1/s2 wavs",
    )
    parser.add_argument(
        "--in_visual_dir",
        type=str,
        default="/gpfs-flash/hulab/public_datasets/audio_datasets/vox2/faces",
        help="Directory of <clip><visual_ext> visual files (faces .mp4 or mouths .npz)",
    )
    parser.add_argument(
        "--visual_ext",
        type=str,
        default=".mp4",
        help="Visual file extension: .mp4 (whole-face video) or .npz (mouth ROI)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="DataPreProcess/vox2",
        help="Output directory for the JSON manifests",
    )
    args = parser.parse_args()
    print(args)
    main(args)
