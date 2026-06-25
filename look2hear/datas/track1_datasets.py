"""VoxCeleb2 audio-visual speech enhancement datasets (n_src=1 enhancement).

This module is a port of the AV-TIGER ``dynamic_avspeech_dataset`` route, wired
to the VoxCeleb2 ``audio_10w`` corpus via the JSON manifests produced by
``preprocess/process_vox2.py``.

Layout of a manifest directory (one per split ``tr`` / ``cv`` / ``tt``)::

    <json_dir>/mix.json   -> [[mix_wav_path, n_samples], ...]
    <json_dir>/s1.json    -> [[s1_wav_path, s1_mouth_npz, n_samples], ...]
    <json_dir>/s2.json    -> [[s2_wav_path, s2_mouth_npz, n_samples], ...]

Two data routes, both yielding the 4-tuple ``(mixture[T], target[T],
mouth[Tv, 88, 88], filename)`` that ``AudioVisualLightningModule`` / ``test.py``
expect:

* **Train** (:class:`Vox2DynamicDataset`) -- dynamic on-the-fly remix. Each step
  draws two clean single-speaker source clips of *different* speakers, sums them
  into a fresh mixture, and returns one of the two as the target with its mouth
  video. Every step is a new pairing, so an "epoch" is just one pass over the
  manifest length.

* **Val / Test** (:class:`Vox2StaticDataset`) -- the pre-generated mixtures from
  ``mix.json``. Each mixture contributes both of its sources (``s1`` and ``s2``)
  as separate evaluation items, so SI-SNR is measured over both talkers.

The target speaker's mouth is a grayscale ``(T, 96, 96)`` uint8 array stored in
``<mouth>.npz`` under key ``data`` (25 fps), run through the lip-reading
preprocessing pipeline (random/center crop to 88x88 + LRW normalization).
"""

import os
import re
import json
import random

import cv2
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

from .transform import get_preprocessing_pipelines

SR = 16000
FPS = 25
SPEAKER_RE = re.compile(r"id\d+")  # VoxCeleb2 speaker token
_EPS = 1e-8


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    """Mean/variance normalize a waveform tensor along the last axis."""
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


def _read_mp4_gray(path, size):
    """Decode an ``.mp4`` whole-face clip to a ``(T, size, size)`` gray array.

    Frames are read as RGB, converted to grayscale, and resized to
    ``size x size`` so the in-frame face scale matches the 96x96 ``.npz`` mouths.
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


def _read_mp4_rgb(path):
    """Decode an ``.mp4`` whole-face clip to a ``(T, H, W, 3)`` uint8 RGB array."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0).astype(np.uint8)


def _load_visual_frames(path, face_size):
    """Load a visual clip as a ``(T, H, W)`` grayscale array.

    ``.npz`` files are pre-cropped mouth ROIs (key ``data``, already grayscale).
    ``.mp4`` files are whole-face videos decoded + grayscaled + resized to
    ``face_size``. Other extensions raise.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return np.load(path)["data"]
    if ext == ".mp4":
        return _read_mp4_gray(path, face_size)
    raise ValueError(f"{path}: unsupported visual format {ext!r}")


def _speaker_of(mouth_path):
    """Target speaker id (first ``id#####`` token) from a mouth ``.npz`` path."""
    m = SPEAKER_RE.search(os.path.basename(mouth_path))
    return m.group(0) if m else mouth_path


def _load_manifests(json_dir):
    """Read ``mix.json`` / ``s1.json`` / ``s2.json`` from ``json_dir``."""
    if not json_dir:
        raise ValueError("json_dir is None or empty")
    with open(os.path.join(json_dir, "mix.json"), "r") as f:
        mix_infos = json.load(f)
    sources_infos = []
    for src in ["s1", "s2"]:
        with open(os.path.join(json_dir, src + ".json"), "r") as f:
            sources_infos.append(json.load(f))
    return mix_infos, sources_infos


# ---------------------------------------------------------------------------
# Train dataset: dynamic on-the-fly remix
# ---------------------------------------------------------------------------

class Vox2DynamicDataset(Dataset):
    """Dynamic two-speaker remix over the VoxCeleb2 single-source clips.

    Each ``__getitem__`` draws two source clips of *different* speakers from the
    flat pool of all ``s1``/``s2`` entries, sums them into a fresh mixture, and
    returns one of the two as the target with its mouth video.

    Args:
        json_dir: Manifest directory holding ``mix.json`` / ``s1.json`` /
            ``s2.json`` (the train split, usually ``.../vox2/tr``).
        n_src: Kept for interface parity; only ``1`` (enhancement) is supported.
        sample_rate: Audio sample rate (must be 16 kHz).
        segment: Crop length in seconds. ``None`` disables cropping (full clip).
        normalize_audio: Mean/var normalize mix and target by the mix std.
        is_train: Selects the train video pipeline (random crop + flip).
    """

    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 1,
        sample_rate: int = SR,
        segment: float = 2.0,
        normalize_audio: bool = False,
        is_train: bool = True,
        face_size: int = 96,
    ):
        super().__init__()
        if sample_rate != SR:
            raise ValueError(f"Only {SR} Hz is supported, got {sample_rate}")
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.n_src = n_src
        self.face_size = int(face_size)
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if is_train else "val"
        ]

        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            self.fps_len = int(segment * FPS)
        self.test = self.seg_len is None

        mix_infos, sources_infos = _load_manifests(json_dir)
        # Flat pool of every clean single-speaker clip (s1 and s2 of every mix),
        # each tagged with its target speaker so we can avoid same-speaker pairs.
        self.sources = []
        for src_inf in sources_infos:
            for entry in src_inf:
                self.sources.append(entry)  # [wav_path, mouth_npz, n_samples]
        self.length = len(mix_infos)
        print(
            f"Vox2DynamicDataset: {self.length} mixtures, "
            f"{len(self.sources)} source clips (seg_len={self.seg_len})"
        )

    def __len__(self):
        return self.length

    def _read_audio(self, path):
        stop = None if self.test else self.seg_len
        return sf.read(path, start=0, stop=stop, dtype="float32")[0]

    def _read_mouth(self, path):
        frames = _load_visual_frames(path, self.face_size)
        mouth = self.lipreading_preprocessing_func(frames)
        if self.fps_len is not None:
            mouth = mouth[: self.fps_len]
        return mouth.astype(np.float32)

    def __getitem__(self, idx: int):
        # Draw two clips of different speakers (idx is ignored -- dynamic remix).
        while True:
            s1_entry = random.choice(self.sources)
            s2_entry = random.choice(self.sources)
            if _speaker_of(s1_entry[1]) != _speaker_of(s2_entry[1]):
                break

        s1 = torch.from_numpy(self._read_audio(s1_entry[0]))
        s2 = torch.from_numpy(self._read_audio(s2_entry[0]))
        # Align lengths before summing (clips can differ by a few samples).
        n = min(s1.shape[-1], s2.shape[-1])
        s1, s2 = s1[..., :n], s2[..., :n]
        mixture = s1 + s2

        entries = [s1_entry, s2_entry]
        sources = [s1, s2]
        select_idx = random.randint(0, 1)
        source = sources[select_idx]
        source_mouth = self._read_mouth(entries[select_idx][1])

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=_EPS, std=m_std)
            source = normalize_tensor_wav(source, eps=_EPS, std=m_std)

        filename = os.path.basename(entries[select_idx][0])
        return mixture, source, source_mouth, filename


# ---------------------------------------------------------------------------
# Val / Test dataset: pre-generated static mixtures
# ---------------------------------------------------------------------------

class Vox2StaticDataset(Dataset):
    """Pre-generated VoxCeleb2 mixtures for validation / test.

    Each mixture in ``mix.json`` contributes both of its clean sources as
    separate items: ``(mix, s1, s1_mouth)`` and ``(mix, s2, s2_mouth)``. Short
    utterances (< ``seg_len``) are dropped during training-style cropping; in
    test mode (``segment=None``) the full clip is returned.

    Args:
        json_dir: Manifest directory (the ``cv`` or ``tt`` split).
        n_src: Kept for interface parity; only ``1`` (enhancement) is supported.
        sample_rate: Audio sample rate (must be 16 kHz).
        segment: Crop length in seconds, or ``None`` for the full clip (test).
        normalize_audio: Mean/var normalize mix and target by the mix std.
        is_train: Forwarded to the video pipeline (False for val/test).
    """

    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 1,
        sample_rate: int = SR,
        segment: float = 2.0,
        normalize_audio: bool = False,
        is_train: bool = False,
        face_size: int = 96,
    ):
        super().__init__()
        if sample_rate != SR:
            raise ValueError(f"Only {SR} Hz is supported, got {sample_rate}")
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.n_src = n_src
        self.face_size = int(face_size)
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if is_train else "val"
        ]

        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            self.fps_len = int(segment * FPS)
        self.test = self.seg_len is None

        mix_infos, sources_infos = _load_manifests(json_dir)

        # One item per (mixture, source): (mix_path, src_path, mouth_npz).
        self.items = []
        drop_utt, drop_len = 0, 0
        for i in range(len(mix_infos)):
            mix_path, mix_len = mix_infos[i][0], mix_infos[i][1]
            if not self.test and mix_len < self.seg_len:
                drop_utt += 1
                drop_len += mix_len
                continue
            for src_inf in sources_infos:
                self.items.append((mix_path, src_inf[i][0], src_inf[i][1]))

        print(
            "Vox2StaticDataset: drop {} utts ({:.2f} h), {} items "
            "(seg_len={})".format(
                drop_utt, drop_len / sample_rate / 3600, len(self.items),
                self.seg_len,
            )
        )
        self.length = len(self.items)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        mix_path, src_path, mouth_path = self.items[idx]
        stop = None if self.test else self.seg_len

        mix_source = sf.read(mix_path, start=0, stop=stop, dtype="float32")[0]
        source = sf.read(src_path, start=0, stop=stop, dtype="float32")[0]

        source_mouth = self.lipreading_preprocessing_func(
            _load_visual_frames(mouth_path, self.face_size)
        )
        if self.fps_len is not None:
            source_mouth = source_mouth[: self.fps_len]
        source_mouth = source_mouth.astype(np.float32)

        mixture = torch.from_numpy(mix_source)
        source = torch.from_numpy(source)
        n = min(mixture.shape[-1], source.shape[-1])
        mixture, source = mixture[..., :n], source[..., :n]

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=_EPS, std=m_std)
            source = normalize_tensor_wav(source, eps=_EPS, std=m_std)

        filename = os.path.basename(src_path)
        return mixture, source, source_mouth, filename


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class Vox2DataModule(object):
    """Wire the VoxCeleb2 dynamic-train / static-eval datasets into loaders.

    Args mirror the AV-TIGER ``AVSpeechDyanmicDataModule``: ``train_dir`` uses
    the dynamic remix route, ``valid_dir`` / ``test_dir`` use the static route
    (test with full-length clips).
    """

    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 1,
        sample_rate: int = SR,
        segment: float = 2.0,
        normalize_audio: bool = False,
        batch_size: int = 4,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        face_size: int = 96,
    ) -> None:
        super().__init__()
        if train_dir is None or valid_dir is None or test_dir is None:
            raise ValueError("JSON DIR is None!")

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.face_size = face_size

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = Vox2DynamicDataset(
            json_dir=self.train_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            is_train=True,
            face_size=self.face_size,
        )
        self.data_val = Vox2StaticDataset(
            json_dir=self.valid_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            is_train=False,
            face_size=self.face_size,
        )
        self.data_test = Vox2StaticDataset(
            json_dir=self.test_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=None,  # full-length clips for test
            normalize_audio=self.normalize_audio,
            is_train=False,
            face_size=self.face_size,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=1,  # full-length clips -> one at a time
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test
