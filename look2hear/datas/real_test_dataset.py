"""Real-world AVSE test dataset, read directly from the REAL-AVSE corpus tree.

Layout (per track, per split)::

    REAL-AVSE/track{1,2}/{dev,test}/
        mix/   <id>/         mix.wav  s1.mp4 s1.txt s1.pkl  s2.mp4 s2.txt s2.pkl
        remix/ <s1id_s2id>/  mix.wav  s1.{mp4,wav,txt,pkl}  s2.{mp4,wav,txt,pkl}
        mix_manifest.json    remix_manifest.json

Two scene types, both yielding one eval item per *target speaker* (s1 and s2):

* ``mix``  -- REAL recordings of two people talking at once. ``mix.wav`` is the
  real noisy mixture; there is NO clean reference (no s1.wav/s2.wav), so only
  no-reference / text / speaker metrics apply.
* ``remix`` -- SYNTHETIC mixtures (two clean single-speaker clips summed). Ships
  ``s1.wav`` / ``s2.wav`` clean references, so SI-SDR / PESQ / STOI can be
  computed *in addition to* the no-reference metrics.

Each item dict carries:
    key        : unique id, e.g. "track1/test/mix/000591/s1"
    track      : "track1" / "track2"
    split      : "dev" / "test"
    scene      : "mix" / "remix"
    clip_id    : directory name (e.g. "000591" or "000129_000662")
    spk_tag    : "s1" / "s2"
    speaker_id : speaker label from the manifest (e.g. "D_L")
    text       : reference transcript (from s{1,2}.txt)
    wav_path   : the noisy mixture (mix.wav) -- model input
    face_path  : target speaker face mp4 (s{1,2}.mp4)
    lm_path    : per-frame 68-pt landmark pickle (s{1,2}.pkl), for face align
    ref_path   : clean reference wav (remix only; None for mix)

The whole-face ``.mp4`` (256x256, 25 fps) is decoded to grayscale, resized to
``face_size`` (96), then run through the val lip pipeline (CenterCrop 88 +
normalization). Audio is 16 kHz mono, read full-length.
"""

import os
import json
import pickle

import cv2
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .transform import get_preprocessing_pipelines

SR = 16000
FPS = 25

# Default REAL-AVSE corpus root (the dataset tree, not the project's preprocess).
REAL_AVSE_ROOT = "/gpfs/hulab/public_datasets/audio_datasets/REAL-AVSE"

# Face-alignment calibration, reverse-engineered from the vox2 ``faces/*.mp4``
# training crops (produced by detectface.py: MTCNN bbox -> face2head square ->
# resize 224). In those final crops the 68-pt landmark bounding box occupies
# ~78% of the frame width with its centre at (50%, 61%). REAL-AVSE faces are
# whole-scene 256x256 (face only ~55% wide, shifted up), so feeding them through
# the same resize96->CenterCrop88 path puts the mouth ROI at the wrong scale and
# position for the model. We re-crop REAL faces to match this composition.
_TARGET_FACE_FRAC = 0.78
_TARGET_CX, _TARGET_CY = 0.50, 0.61


def _read_face_gray(path, size):
    """Decode a whole-face ``.mp4`` to a ``(T, size, size)`` grayscale array.

    BGR -> gray -> resize, matching ``track1_datasets._read_mp4_gray`` so the
    in-frame face scale lines up with the training crops.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] != size or gray.shape[1] != size:
            gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(gray)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0)


def _fixed_face_box(landmarks):
    """One fixed square crop box ``(left, top, side)`` for a whole clip, or
    ``None`` if no frame has a valid detection.

    ``landmarks`` is a per-frame list of ``(68, 2)`` arrays (REAL-AVSE ``.pkl``,
    coords in the original frame, 1:1 with the video frames). Frames where
    detection failed are stored as ``None`` and skipped. Uses the *median*
    landmark extents across valid frames so the box is stable against per-frame
    jitter (REAL faces barely move: centre std ~6 px). The square side is set so
    the face spans ``_TARGET_FACE_FRAC`` of the crop, matching the vox2 training
    composition (cf. detectface.py ``face2head``), then re-centred so the face
    centre lands at ``(_TARGET_CX, _TARGET_CY)`` of the box.
    """
    valid = [np.asarray(f, np.float32) for f in landmarks
             if f is not None and np.asarray(f).shape == (68, 2)]
    if not valid:
        return None
    A = np.stack(valid)                                          # (Tv, 68, 2)
    x0 = np.median(A[:, :, 0].min(1)); x1 = np.median(A[:, :, 0].max(1))
    y0 = np.median(A[:, :, 1].min(1)); y1 = np.median(A[:, :, 1].max(1))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) / _TARGET_FACE_FRAC
    left = cx - _TARGET_CX * side
    top = cy - _TARGET_CY * side
    return left, top, side


def _crop_square(gray, left, top, side, size):
    """Crop a ``side``-wide square at ``(left, top)`` from ``gray``, resize to
    ``size``. Out-of-frame regions are edge-replicated (PIL-crop would pad black;
    replicate keeps the border statistics closer to a real face background).
    """
    h, w = gray.shape
    l, t = int(round(left)), int(round(top))
    s = int(round(side))
    # Pad enough so the requested square always lies inside the padded image.
    pl = max(0, -l); pt = max(0, -t)
    pr = max(0, l + s - w); pb = max(0, t + s - h)
    if pl or pt or pr or pb:
        gray = cv2.copyMakeBorder(gray, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
        l += pl; t += pt
    crop = gray[t:t + s, l:l + s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _read_face_gray_aligned(mp4_path, pkl_path, size):
    """Decode ``mp4_path`` and crop each frame to the landmark-aligned square,
    returning a ``(T, size, size)`` grayscale array matching the vox2 face scale.

    Returns ``None`` if the clip has no valid landmark detections, so the caller
    can fall back to the legacy whole-face resize.
    """
    landmarks = pickle.load(open(pkl_path, "rb"))
    box = _fixed_face_box(landmarks)
    if box is None:
        return None
    left, top, side = box
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(_crop_square(gray, left, top, side, size))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {mp4_path}")
    return np.stack(frames, axis=0)


def _read_text(path):
    """Read a transcript ``.txt`` (single line), or '' if missing."""
    if not os.path.isfile(path):
        return ""
    with open(path, "r") as f:
        return f.read().strip()


class RealTestDataset(Dataset):
    """REAL-AVSE eval items, expanded per target speaker over mix/remix scenes.

    Args:
        items: list of pre-built item dicts (see module docstring). Usually
            constructed via :func:`build_items`.
        face_size: resize faces to this size before the val lip pipeline (96).
        sample_rate: audio sample rate (must be 16 kHz).
        align_face: if True (default), re-crop each face to the vox2 training
            composition using the clip's ``.pkl`` landmarks; if False, fall back
            to the legacy whole-face resize (kept for A/B comparison).
    """

    def __init__(self, items, face_size=96, sample_rate=SR, align_face=True):
        super().__init__()
        if sample_rate != SR:
            raise ValueError(f"Only {SR} Hz is supported, got {sample_rate}")
        self.items = items
        self.face_size = int(face_size)
        self.sample_rate = sample_rate
        self.align_face = bool(align_face)
        self.lip_pipeline = get_preprocessing_pipelines()["val"]
        self._warned_no_lm = False
        print(f"RealTestDataset: {len(self.items)} eval items "
              f"(align_face={self.align_face})")

    def __len__(self):
        return len(self.items)

    def _read_mouth(self, face_path, lm_path=None):
        gray = None
        if self.align_face and lm_path and os.path.isfile(lm_path):
            gray = _read_face_gray_aligned(face_path, lm_path, self.face_size)
        if gray is None:                                         # no align / no landmarks
            if self.align_face and not self._warned_no_lm:
                print(f"[RealTestDataset] WARN: no usable landmarks ({lm_path}), "
                      f"falling back to whole-face resize for some clips")
                self._warned_no_lm = True
            gray = _read_face_gray(face_path, self.face_size)    # (T, 96, 96)
        mouth = self.lip_pipeline(gray)                          # (T, 88, 88)
        return mouth.astype(np.float32)

    def __getitem__(self, idx):
        meta = self.items[idx]
        wav = sf.read(meta["wav_path"], dtype="float32")[0]
        if wav.ndim > 1:                                         # stereo -> mono
            wav = wav.mean(axis=1)
        mix = torch.from_numpy(wav)
        mouth = torch.from_numpy(self._read_mouth(meta["face_path"],
                                                  meta.get("lm_path")))
        return mix, mouth, meta


def _expand_dir(entry, root, track, split, scene):
    """Build the two (s1, s2) item dicts for one manifest entry."""
    clip_id = entry["mix_id"]
    clip_dir = os.path.join(root, track, split, scene, clip_id)
    mix_wav = os.path.join(clip_dir, "mix.wav")
    items = []
    for tag in ("s1", "s2"):
        ref = os.path.join(clip_dir, f"{tag}.wav")
        items.append({
            "key": f"{track}/{split}/{scene}/{clip_id}/{tag}",
            "track": track,
            "split": split,
            "scene": scene,
            "clip_id": clip_id,
            "spk_tag": tag,
            "speaker_id": entry.get(f"{tag}_speaker", ""),
            "text": _read_text(os.path.join(clip_dir, f"{tag}.txt")),
            "wav_path": mix_wav,
            "face_path": os.path.join(clip_dir, f"{tag}.mp4"),
            "lm_path": os.path.join(clip_dir, f"{tag}.pkl"),
            # remix ships clean refs (s1.wav/s2.wav); mix does not.
            "ref_path": ref if (scene == "remix" and os.path.isfile(ref)) else None,
        })
    return items


def build_items(root=REAL_AVSE_ROOT, tracks=("track1", "track2"),
                split="test", scenes=("mix", "remix"), limit=None):
    """Read manifests under the corpus tree and expand into per-speaker items.

    Args:
        root: REAL-AVSE corpus root.
        tracks / scenes: which subsets to include.
        split: "dev" or "test".
        limit: optional cap on manifest entries per (track, scene) -- the number
            of *directories*; each contributes 2 items (s1, s2).

    Returns the flat item list for :class:`RealTestDataset`.
    """
    manifest_name = {"mix": "mix_manifest.json", "remix": "remix_manifest.json"}
    items = []
    for track in tracks:
        for scene in scenes:
            mpath = os.path.join(root, track, split, manifest_name[scene])
            if not os.path.isfile(mpath):
                print(f"[build_items] WARN: missing {mpath}, skipped")
                continue
            entries = json.load(open(mpath))
            if limit is not None:
                entries = entries[:limit]
            for e in entries:
                items.extend(_expand_dir(e, root, track, split, scene))
    return items


def build_enrollment_items(root=REAL_AVSE_ROOT, tracks=("track1", "track2"),
                           split="test", limit=None):
    """Collect clean single-speaker clips for speaker enrollment.

    Uses the ``remix`` scene's ``s1.wav`` / ``s2.wav`` (clean single-speaker
    sources) grouped by speaker label. Returns ``[(speaker_id, wav_path), ...]``.
    """
    out = []
    for track in tracks:
        mpath = os.path.join(root, track, split, "remix_manifest.json")
        if not os.path.isfile(mpath):
            continue
        entries = json.load(open(mpath))
        if limit is not None:
            entries = entries[:limit]
        for e in entries:
            clip_dir = os.path.join(root, track, split, "remix", e["mix_id"])
            for tag in ("s1", "s2"):
                wav = os.path.join(clip_dir, f"{tag}.wav")
                spk = e.get(f"{tag}_speaker", "")
                if spk and os.path.isfile(wav):
                    out.append((spk, wav))
    return out
