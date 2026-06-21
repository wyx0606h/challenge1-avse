"""Track2 audio-visual datasets: VoxCeleb2 with online visual degradation.

Standalone module (does not import from ``track1_datasets``). Track2 trains for
robustness to corrupted video. The audio route matches Track1 -- dynamic
two-speaker remix for train, static pre-generated mixtures for val/test, over
the VoxCeleb2 ``audio_10w`` corpus and its JSON manifests -- but before the
target face is grayscaled and cropped, the RGB whole-face frames are passed
through one of the visual degradations in
:mod:`look2hear.datas.visual_perturb`:

    0 FULL_MASK      blackout a block of frames (frame drop / zero padding)
    1 OCCLUSION      overlay a COCO object on the lips (needs landmarks)
    2 LOW_RESOLUTION gaussian noise / blur
    3 FRAME_FREEZE   freeze a block of frames (video stalls, audio continues)
    4 AV_DESYNC      shift the video relative to the audio

Only the target's visual stream is degraded; audio is untouched. Each
``__getitem__`` returns the 4-tuple ``(mixture[T], target[T], mouth[Tv, 88, 88],
filename)`` that ``AudioVisualLightningModule`` / ``test.py`` expect.

Manifest layout (one dir per split ``tr`` / ``cv`` / ``tt``), produced by
``DataPreProcess/process_vox2.py`` with faces (``.mp4``)::

    <json_dir>/mix.json -> [[mix_wav, n_samples], ...]
    <json_dir>/s1.json  -> [[s1_wav, s1_face_mp4, n_samples], ...]
    <json_dir>/s2.json  -> [[s2_wav, s2_face_mp4, n_samples], ...]

Landmarks (needed only for occlusion, type 1) are read from a ``.npz`` sibling
of the face ``.mp4`` -- the same clip id under a ``landmark/`` folder instead of
``faces/`` (shape ``(T, 1, 68, 2)`` or ``(T, 68, 2)``).
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
from .visual_perturb import (
    apply_degradation,
    generate_degradation_params,
    DegradationParams,
)

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


def _speaker_of(path):
    """Target speaker id (first ``id#####`` token) from a clip path."""
    m = SPEAKER_RE.search(os.path.basename(path))
    return m.group(0) if m else path


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


def _read_face_rgb(path):
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


def _load_visual_rgb(path):
    """Load the target's visual clip as a ``(T, H, W, 3)`` uint8 RGB array.

    ``.mp4`` faces are decoded directly. ``.npz`` mouth ROIs (grayscale ``data``)
    are tiled to 3 channels so the degradation pipeline (which expects RGB) and
    the downstream grayscale conversion still work.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mp4":
        return _read_face_rgb(path)
    if ext == ".npz":
        g = np.load(path)["data"]  # (T, H, W) uint8 grayscale
        return np.repeat(g[..., None], 3, axis=-1).astype(np.uint8)
    raise ValueError(f"{path}: unsupported visual format {ext!r}")


def _face_to_landmark_path(face_path):
    """Map a ``.../faces/<clip>.mp4`` path to ``.../landmark/<clip>.npz``."""
    return face_path.replace("/faces/", "/landmark/").rsplit(".", 1)[0] + ".npz"


def _load_landmarks(path):
    """Load landmarks as a ``(T, 68, 2)`` float32 array, or None if missing.

    Most files are a regular ``(T, 1, 68, 2)`` array (one detected face per
    frame). A minority are stored as an object array of per-frame detections
    whose face count varies (0 or >1 faces), so they cannot be stacked
    directly; for those we keep the first detection per frame and drop frames
    with no detection. If the result cannot be coerced to ``(T, 68, 2)`` the
    function returns ``None`` so the caller can fall back to a landmark-free
    degradation.
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        raw = np.load(path, allow_pickle=True)["data"]
    except Exception:
        return None

    lm = np.asarray(raw, dtype=object) if raw.dtype == object else np.asarray(raw)

    # Regular numeric array: (T, 1, 68, 2) -> (T, 68, 2), or already (T, 68, 2).
    if lm.dtype != object:
        lm = lm.astype(np.float32)
        if lm.ndim == 4 and lm.shape[1] >= 1:
            lm = lm[:, 0]
        return lm if lm.ndim == 3 and lm.shape[-1] == 2 else None

    # Object array of ragged per-frame detections: take the first face each
    # frame, skip frames with none.
    frames = []
    for fr in lm:
        fr = np.asarray(fr, dtype=np.float32)
        if fr.ndim == 3 and fr.shape[0] >= 1:   # (n_faces, 68, 2)
            frames.append(fr[0])
        elif fr.ndim == 2 and fr.shape[-1] == 2:  # (68, 2)
            frames.append(fr)
    if not frames:
        return None
    return np.stack(frames, axis=0)



def _rgb_to_gray_resize(frames, size):
    """Convert a ``(T, H, W, 3)`` RGB clip to ``(T, size, size)`` grayscale."""
    out = np.empty((frames.shape[0], size, size), dtype=np.uint8)
    for i, f in enumerate(frames):
        g = cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY)
        if g.shape[0] != size or g.shape[1] != size:
            g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        out[i] = g
    return out


def degrade_face_rgb(frames, landmark_path=None, state="train",
                     params=None, seed=None):
    """Apply one online visual degradation to a ``(T, H, W, 3)`` RGB face clip.

    Args:
        frames: ``(T, H, W, 3)`` uint8 RGB whole-face clip.
        landmark_path: ``.npz`` landmark path (only loaded for occlusion).
        state: ``"train"`` or ``"test"`` (occluder size policy).
        params: A fixed :class:`DegradationParams`; if ``None`` a random one is
            drawn with :func:`generate_degradation_params`.
        seed: Optional RNG seed for reproducible degradation (val/test).

    Returns:
        ``(degraded_frames[T, H, W, 3] uint8 RGB, params)``. Type 2 returns BGR
        from the underlying noise op; since the frames are grayscaled next, the
        channel order does not matter.
    """
    # Reproducibility: type 1/2 also draw from torch's global RNG (GaussianBlur
    # sigma) and python's random; seed both. The skimage noise instance is made
    # reproducible separately by passing seed through to random_noise.
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if params is None:
        params = generate_degradation_params(frames.shape[0], seed=seed)
    landmarks = None
    if int(params.deg_type) == 1:  # OCCLUSION needs landmarks
        landmarks = _load_landmarks(landmark_path)
        # occlude_sequence indexes landmarks by video-frame index, so it needs
        # at least as many landmark frames as video frames. Missing / too-short
        # landmarks -> fall back to noise/blur (type 2).
        if landmarks is None or landmarks.shape[0] < frames.shape[0]:
            landmarks = None
            params = DegradationParams(
                mask_start_frame=params.mask_start_frame,
                mask_len_frames=params.mask_len_frames,
                deg_type=2,
            )
    out = apply_degradation(
        frames, landmarks, params, state=state, seed=seed
    )
    return np.ascontiguousarray(out).astype(np.uint8), params


# ---------------------------------------------------------------------------
# Train dataset: dynamic remix + online visual degradation
# ---------------------------------------------------------------------------

class Track2DynamicDataset(Dataset):
    """Dynamic two-speaker remix with online degradation of the target face.

    Identical audio sampling to the Track1 dynamic dataset: every step draws two
    clean single-speaker clips of *different* speakers, sums them, and returns
    one as the target. The target's RGB face is degraded with probability
    ``degrade_prob`` before grayscale + crop.

    Args:
        json_dir: Manifest dir with ``mix.json`` / ``s1.json`` / ``s2.json``.
        n_src: Kept for interface parity; only ``1`` (enhancement) is supported.
        sample_rate: Audio sample rate (must be 16 kHz).
        segment: Crop length in seconds (``None`` = full clip).
        normalize_audio: Mean/var normalize mix and target by the mix std.
        is_train: Selects train video pipeline (random crop + flip).
        face_size: Resize faces to this size before grayscale + crop.
        degrade_prob: Probability a given sample gets a visual degradation.
        degrade_state: ``"train"`` / ``"test"`` occluder size policy.
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
        degrade_prob: float = 1.0,
        degrade_state: str = "train",
    ):
        super().__init__()
        if sample_rate != SR:
            raise ValueError(f"Only {SR} Hz is supported, got {sample_rate}")
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.n_src = n_src
        self.face_size = int(face_size)
        self.degrade_prob = float(degrade_prob)
        self.degrade_state = degrade_state
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
        self.sources = []
        for src_inf in sources_infos:
            for entry in src_inf:
                self.sources.append(entry)  # [wav, face_mp4, n_samples]
        self.length = len(mix_infos)
        print(
            f"Track2DynamicDataset: {self.length} mixtures, "
            f"{len(self.sources)} source clips, degrade_prob={self.degrade_prob}"
        )

    def __len__(self):
        return self.length

    def _read_audio(self, path):
        stop = None if self.test else self.seg_len
        return sf.read(path, start=0, stop=stop, dtype="float32")[0]

    def _read_mouth(self, face_path):
        frames = _load_visual_rgb(face_path)  # (T, H, W, 3) RGB
        if random.random() < self.degrade_prob:
            frames, _ = degrade_face_rgb(
                frames,
                landmark_path=_face_to_landmark_path(face_path),
                state=self.degrade_state,
            )
        gray = _rgb_to_gray_resize(frames, self.face_size)
        mouth = self.lipreading_preprocessing_func(gray)
        if self.fps_len is not None:
            mouth = mouth[: self.fps_len]
        return mouth.astype(np.float32)

    def __getitem__(self, idx: int):
        while True:
            s1_entry = random.choice(self.sources)
            s2_entry = random.choice(self.sources)
            if _speaker_of(s1_entry[1]) != _speaker_of(s2_entry[1]):
                break

        s1 = torch.from_numpy(self._read_audio(s1_entry[0]))
        s2 = torch.from_numpy(self._read_audio(s2_entry[0]))
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
# Val / Test dataset: static mixtures + reproducible visual degradation
# ---------------------------------------------------------------------------

class Track2StaticDataset(Dataset):
    """Pre-generated VoxCeleb2 mixtures with reproducible visual degradation.

    Each mixture contributes both sources (``s1``, ``s2``) as separate items.
    Degradation is applied with a per-item seed (derived from the item index),
    so val/test scores are deterministic across runs.

    Args mirror :class:`Track2DynamicDataset`; ``degrade_state`` defaults to
    ``"test"`` (fixed occluder size).
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
        degrade_prob: float = 1.0,
        degrade_state: str = "test",
    ):
        super().__init__()
        if sample_rate != SR:
            raise ValueError(f"Only {SR} Hz is supported, got {sample_rate}")
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.n_src = n_src
        self.face_size = int(face_size)
        self.degrade_prob = float(degrade_prob)
        self.degrade_state = degrade_state
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
            "Track2StaticDataset: drop {} utts ({:.2f} h), {} items, "
            "degrade_prob={}".format(
                drop_utt, drop_len / sample_rate / 3600, len(self.items),
                self.degrade_prob,
            )
        )
        self.length = len(self.items)

    def __len__(self):
        return self.length

    def _read_mouth(self, face_path, seed):
        frames = _load_visual_rgb(face_path)
        # Reproducible: same seed -> same degradation type/params each epoch.
        rng = random.Random(seed)
        if rng.random() < self.degrade_prob:
            frames, _ = degrade_face_rgb(
                frames,
                landmark_path=_face_to_landmark_path(face_path),
                state=self.degrade_state,
                seed=seed,
            )
        gray = _rgb_to_gray_resize(frames, self.face_size)
        mouth = self.lipreading_preprocessing_func(gray)
        if self.fps_len is not None:
            mouth = mouth[: self.fps_len]
        return mouth.astype(np.float32)

    def __getitem__(self, idx: int):
        mix_path, src_path, mouth_path = self.items[idx]
        stop = None if self.test else self.seg_len

        mix_source = sf.read(mix_path, start=0, stop=stop, dtype="float32")[0]
        source = sf.read(src_path, start=0, stop=stop, dtype="float32")[0]
        source_mouth = self._read_mouth(mouth_path, seed=idx)

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

class Track2DataModule(object):
    """Wire the Track2 dynamic-train / static-eval datasets into loaders.

    ``train_dir`` uses the dynamic remix route with online degradation;
    ``valid_dir`` / ``test_dir`` use the static route with reproducible
    (seeded) degradation. ``test`` runs full-length clips at batch size 1.
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
        degrade_prob: float = 1.0,
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
        self.degrade_prob = degrade_prob

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = Track2DynamicDataset(
            json_dir=self.train_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            is_train=True,
            face_size=self.face_size,
            degrade_prob=self.degrade_prob,
            degrade_state="train",
        )
        self.data_val = Track2StaticDataset(
            json_dir=self.valid_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            is_train=False,
            face_size=self.face_size,
            degrade_prob=self.degrade_prob,
            degrade_state="test",
        )
        self.data_test = Track2StaticDataset(
            json_dir=self.test_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=None,  # full-length clips for test
            normalize_audio=self.normalize_audio,
            is_train=False,
            face_size=self.face_size,
            degrade_prob=self.degrade_prob,
            degrade_state="test",
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
            batch_size=1,
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




