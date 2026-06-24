"""
Core visual perturbation functions for online video augmentation.

Supports three degradation types:
  0. FULL_MASK    — blackout frames (simulates frame drop / zero padding)
  1. OCCLUSION    — overlay a random COCO object on face landmarks
  2. LOW_RESOLUTION — Gaussian noise or Gaussian blur

Asset structure expected:
  assets/
    object_image_sr/       — JPEG images of COCO objects
    object_mask_x4/        — PNG masks corresponding to the objects
    object_image_sr_test/  — (optional) test-set objects
    object_mask_x4_test/   — (optional) test-set masks
"""

import os
import cv2
import random
import pickle
import numpy as np

import albumentations as A
from skimage.util import random_noise
import torchvision.transforms as T
import torch

# ---------------------------------------------------------------------------
# Paths — adjust ASSETS_DIR if you move the assets folder
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(_THIS_DIR, "assets")

DEFAULT_OBJECT_DIR = os.path.join(ASSETS_DIR, "object_image_sr")
DEFAULT_MASK_DIR = os.path.join(ASSETS_DIR, "object_mask_x4")

# Minimum frame count for valid sequences
_WINDOW_MARGIN = 12

# ---------------------------------------------------------------------------
# Seeding (original code seeds globally; we keep it optional)
# ---------------------------------------------------------------------------
random.seed(10)
np.random.seed(10)


# ===================================================================
#  Low-level helpers
# ===================================================================

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Alpha-blend ``img_overlay`` onto ``img`` at position (x, y).

    Args:
        img: Background image, shape (H, W, 3), float or uint8.
        img_overlay: Foreground overlay, shape (h, w, 3).
        x, y: Top-left corner of the overlay on the background.
        alpha_mask: Alpha mask, shape (h, w), values in [0, 1].
    """
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    img_crop = img[y1:y2, x1:x2].astype(np.float32)
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o].astype(np.float32)
    # Accept a 2D (h,w) mask or one already carrying a trailing channel axis
    # (h,w,1); normalise to (h,w) here, then add exactly one channel axis so it
    # broadcasts over the 3 colour channels.
    am = np.asarray(alpha_mask)
    if am.ndim > 2:
        am = am.reshape(am.shape[0], am.shape[1])
    alpha = am[y1o:y2o, x1o:x2o, np.newaxis].astype(np.float32)
    alpha_inv = 1.0 - alpha

    # Write the blended result back INTO img. img_crop is a float32 copy (astype
    # always copies), so we must assign to the img slice, not to img_crop.
    img[y1:y2, x1:x2] = (alpha * img_overlay_crop + alpha_inv * img_crop).astype(img.dtype)
    return img


def linear_interpolate(landmarks, start_idx, stop_idx):
    """Linearly interpolate missing landmarks between two known indices.

    Args:
        landmarks: ndarray of shape (T, N, 2).
        start_idx: Known left index.
        stop_idx: Known right index.

    Returns:
        ndarray with interpolated values filled in-place.
    """
    start = landmarks[start_idx]
    stop = landmarks[stop_idx]
    delta = stop - start
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks


def landmarks_interpolate(landmarks):
    """Fill in missing (None) landmark frames via linear interpolation.

    Corner frames at the beginning/end are filled with the nearest valid frame.

    Args:
        landmarks: List of ndarrays or None, length T.

    Returns:
        ndarray of shape (T, N, 2) with no None entries, or None on failure.
    """
    valid_idx = [i for i, lm in enumerate(landmarks) if lm is not None]
    if not valid_idx:
        return None

    landmarks = list(landmarks)  # don't mutate caller's list
    for idx in range(1, len(valid_idx)):
        if valid_idx[idx] - valid_idx[idx - 1] == 1:
            continue
        landmarks = linear_interpolate(
            landmarks, valid_idx[idx - 1], valid_idx[idx]
        )

    valid_idx = [i for i, lm in enumerate(landmarks) if lm is not None]
    if valid_idx:
        # Fill leading missing frames
        landmarks[:valid_idx[0]] = [landmarks[valid_idx[0]]] * valid_idx[0]
        # Fill trailing missing frames
        landmarks[valid_idx[-1]:] = [landmarks[valid_idx[-1]]] * (
            len(landmarks) - valid_idx[-1]
        )

    valid_idx = [i for i, lm in enumerate(landmarks) if lm is not None]
    assert len(valid_idx) == len(landmarks), "Not every frame has a landmark"
    return np.array(landmarks)


# ===================================================================
#  Occluder loading & augmentation
# ===================================================================

def get_occluder_augmentor():
    """Build the albumentations augmentation pipeline for occluder objects.

    Applies blur, JPEG compression, affine transforms, and brightness jitter.
    """
    # ImageCompression's kwarg changed across albumentations versions:
    # >=2.0 uses quality_range=(lo, hi); <2.0 uses quality_lower/quality_upper.
    try:
        _jpeg = A.ImageCompression(quality_range=(70, 100), p=0.5)
    except TypeError:
        _jpeg = A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5)

    return A.Compose([
        A.AdvancedBlur(),
        A.OneOf([
            _jpeg,
        ], p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            rotate=(-15, 15),
            shear=(-8, 8),
            fit_output=True,
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            p=0.5,
            brightness_limit=0.1,
            contrast_limit=0.1,
            brightness_by_max=False,
        ),
    ])


def get_occluders(object_dir=None, mask_dir=None, state="train"):
    """Randomly select and augment an occluder object from the asset pool.

    Args:
        object_dir: Path to object JPEG images.
        mask_dir: Path to corresponding mask PNGs.
        state: "train" or "test" — controls occluder size (train: 30–50 px,
               test: 40 px).

    Returns:
        tuple (filename, occluder_img, occluder_mask) where:
          - filename: str, name of the chosen object file
          - occluder_img: ndarray (size, size, 3) RGB
          - occluder_mask: ndarray (size, size) uint8
    """
    if object_dir is None:
        object_dir = DEFAULT_OBJECT_DIR
    if mask_dir is None:
        mask_dir = DEFAULT_MASK_DIR

    aug = get_occluder_augmentor()

    size = random.uniform(25, 50) if state == "train" else 40  # original uses 30-50 or 40
    if isinstance(size, float):
        size = int(size)

    occlude_imgs = os.listdir(object_dir)
    occlude_img_name = random.choice(occlude_imgs)
    occlude_mask_name = occlude_img_name.replace("jpeg", "png")

    ori_img = cv2.imread(os.path.join(object_dir, occlude_img_name), -1)
    if ori_img is None:
        raise FileNotFoundError(f"Cannot read {os.path.join(object_dir, occlude_img_name)}")
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(os.path.join(mask_dir, occlude_mask_name))
    if mask is None:
        raise FileNotFoundError(f"Cannot read {os.path.join(mask_dir, occlude_mask_name)}")
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.resize(mask, (ori_img.shape[1], ori_img.shape[0]),
                      interpolation=cv2.INTER_LANCZOS4)

    # Apply mask to get clean object
    occluder_img = cv2.bitwise_and(ori_img, ori_img, mask=mask)

    # Augment
    transformed = aug(image=occluder_img, mask=mask)
    occluder_img, mask = transformed["image"], transformed["mask"]

    occluder_img = cv2.resize(occluder_img, (size, size),
                              interpolation=cv2.INTER_LANCZOS4)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LANCZOS4)

    return occlude_img_name, occluder_img, mask


# ===================================================================
#  Degradation type 1: Object occlusion
# ===================================================================

def occlude_sequence(
    img_seq,
    landmarks,
    object_dir=None,
    mask_dir=None,
    freq=1,
    state="train",
    seed=None,
):
    """Apply object occlusion to a video sequence.

    Overlays random COCO objects onto face-landmark positions for a
    contiguous block of frames.

    Args:
        img_seq: ndarray (T, H, W, 3) in RGB, float or uint8.
        landmarks: ndarray (T, N, 2), face landmark coordinates.
        object_dir: Path to object images (default: assets/object_image_sr).
        mask_dir: Path to mask images (default: assets/object_mask_x4).
        freq: Number of occlusion segments. 1 = one contiguous block,
              >1 = multiple blocks spread across the video.
        state: "train" or "test" — passed to get_occluders.
        seed: Random seed for reproducibility.

    Returns:
        tuple (augmented_frames, occluder_filename) where:
          augmented_frames is ndarray (T, H, W, 3) in RGB.
    """
    if seed is not None:
        random.seed(seed)

    if object_dir is None:
        object_dir = DEFAULT_OBJECT_DIR
    if mask_dir is None:
        mask_dir = DEFAULT_MASK_DIR

    img_seq = img_seq.copy()
    T_len = img_seq.shape[0]

    if freq == 1:
        _, occluder_img, occluder_mask = get_occluders(
            object_dir, mask_dir, state=state
        )
        # Pick a random landmark point (lip region: indices 48–67 for 68-point model)
        start_pt_idx = random.randint(48, 67)
        offset_x, offset_y = 15, 15

        occ_len = random.randint(int(T_len * 0.3), int(T_len * 0.5))
        start_fr = random.randint(0, T_len - occ_len)

        for i in range(occ_len):
            fr = img_seq[start_fr + i]
            x, y = landmarks[start_fr + i, start_pt_idx]
            alpha = np.expand_dims(occluder_mask / 255.0, axis=2)
            img_seq[start_fr + i] = overlay_image_alpha(
                fr, occluder_img, int(x - offset_x), int(y - offset_y), alpha
            )
        occluder_filename = _
    else:
        for j in range(freq):
            _, occluder_img, occluder_mask = get_occluders(
                object_dir, mask_dir, state=state
            )
            start_pt_idx = random.randint(48, 67)
            offset_x, offset_y = 15, 15

            sub_len = T_len // freq
            try:
                occ_len = random.randint(int(T_len * 0.1), int(T_len * 0.5 * sub_len / (T_len / freq)))
                start_fr = random.randint(0, sub_len * j + sub_len - occ_len)
                if start_fr < sub_len * j:
                    raise ValueError("start frame before segment")
            except (ValueError, AssertionError):
                occ_len = sub_len // 2
                start_fr = sub_len * j

            for i in range(occ_len):
                fr = img_seq[start_fr + i]
                x, y = landmarks[start_fr + i, start_pt_idx]
                alpha = np.expand_dims(occluder_mask / 255.0, axis=2)
                img_seq[start_fr + i] = overlay_image_alpha(
                    fr, occluder_img, int(x - offset_x), int(y - offset_y), alpha
                )
            occluder_filename = _

    return img_seq, occluder_filename


# ===================================================================
#  Degradation type 2: Noise / blur (low resolution)
# ===================================================================

def _gaussian_noise(segment, var, seed=None):
    """skimage gaussian noise with version-portable, reproducible seeding.

    skimage renamed the RNG kwarg from ``seed`` (<0.21) to ``rng`` (>=0.21).
    Passing it makes the noise instance deterministic; ``seed=None`` keeps the
    original random behaviour on both versions.
    """
    try:
        return random_noise(segment, mode="gaussian", mean=0, var=var,
                            clip=True, rng=seed)
    except (TypeError, ValueError):
        return random_noise(segment, mode="gaussian", mean=0, var=var,
                            clip=True, seed=seed)


def occlude_sequence_noise(img_seq, freq=1, seed=None):
    """Apply Gaussian noise or Gaussian blur to a video sequence.

    For each degradation segment, randomly chooses:
      - 50% chance: Gaussian noise (var ∈ [0, 0.2])
      - 50% chance: Gaussian blur (kernel=7, sigma ∈ [0.1, 2.0])

    Args:
        img_seq: ndarray (T, H, W, 3) in RGB, values in [0, 255].
        freq: Number of degradation segments.
        seed: Random seed.

    Returns:
        ndarray (T, H, W, 3) in BGR (for consistency with original code).
    """
    if seed is not None:
        random.seed(seed)

    img_seq = img_seq.copy().astype(np.float32)
    T_len = img_seq.shape[0]

    if freq == 1:
        occ_len = random.randint(int(T_len * 0.1), int(T_len * 0.5))
        start_fr = random.randint(0, T_len - occ_len)

        segment = img_seq[start_fr:start_fr + occ_len]
        prob = random.random()
        if prob < 0.5:
            var = random.random() * 0.2
            segment = _gaussian_noise(segment, var, seed) * 255
        else:
            blur = T.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
            segment = (
                blur(torch.tensor(segment).permute(0, 3, 1, 2))
                .permute(0, 2, 3, 1)
                .numpy()
            )
        img_seq[start_fr:start_fr + occ_len] = segment
    else:
        for j in range(freq):
            sub_len = T_len // freq
            try:
                occ_len = random.randint(int(T_len * 0.1), int(T_len * 0.5))
                start_fr = random.randint(0, sub_len * j + sub_len - occ_len)
                if start_fr < sub_len * j:
                    raise ValueError("start frame before segment")
            except (ValueError, AssertionError):
                occ_len = sub_len // 2
                start_fr = sub_len * j

            segment = img_seq[start_fr:start_fr + occ_len]
            prob = random.random()
            if prob < 0.5:
                var = random.random() * 0.2
                segment = _gaussian_noise(segment, var, seed) * 255
            else:
                blur = T.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
                segment = (
                    blur(torch.tensor(segment).permute(0, 3, 1, 2))
                    .permute(0, 2, 3, 1)
                    .numpy()
                )
            img_seq[start_fr:start_fr + occ_len] = segment

    # Convert to BGR (matching original behavior)
    bgr_seq = np.array([cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
                         for im in img_seq])
    return bgr_seq


# ===================================================================
#  Degradation type 0: Full blackout mask
# ===================================================================

def apply_full_mask(img_seq, start_frame, mask_len, seed=None):
    """Set a contiguous block of frames to zero (blackout / frame drop).

    Simulates frame loss — the model sees all-black frames for the masked
    duration, equivalent to "padding to 0" in the temporal dimension.

    Args:
        img_seq: ndarray (T, H, W, 3), any dtype.
        start_frame: First frame index to mask.
        mask_len: Number of frames to set to zero.

    Returns:
        ndarray with masked frames set to 0.
    """
    img_seq = img_seq.copy()
    end_frame = min(start_frame + mask_len, img_seq.shape[0])
    img_seq[start_frame:end_frame] = 0
    return img_seq


# ===================================================================
#  Degradation type 3: Frame freeze / stutter
# ===================================================================

def apply_frame_freeze(img_seq, start_frame=None, freeze_len=None, seed=None):
    """Freeze a segment of video by repeating a single frame.

    Simulates video lag/stutter — the video freezes on one frame while the
    audio continues normally. This breaks the natural AV correspondence
    within the frozen segment.

    Args:
        img_seq: ndarray (T, H, W, 3) in RGB.
        start_frame: First frame of the freeze segment. If None, chosen randomly.
        freeze_len: Number of frames to freeze. If None, chosen randomly
                    (3 to max(3, 30% of T)).
        seed: Random seed.

    Returns:
        ndarray (T, H, W, 3) RGB with frozen segment.
    """
    if seed is not None:
        random.seed(seed)

    img_seq = img_seq.copy()
    T_len = img_seq.shape[0]

    if freeze_len is None:
        freeze_len = random.randint(3, max(3, int(T_len * 0.3)))
    if start_frame is None:
        start_frame = random.randint(0, max(0, T_len - freeze_len))

    # The frame that gets "stuck"
    frozen_frame = img_seq[start_frame].copy()
    for i in range(1, freeze_len):
        end = start_frame + i
        if end >= T_len:
            break
        img_seq[end] = frozen_frame

    return img_seq


# ===================================================================
#  Degradation type 4: Audio-visual desynchronization
# ===================================================================

def apply_av_desync(img_seq, shift_frames=None, max_shift=10, seed=None):
    """Shift the video stream relative to the audio stream.

    Simulates AV misalignment:
      - Positive shift: video is AHEAD of audio (duplicate first frames, truncate end).
      - Negative shift: video LAGS behind audio (drop first frames, duplicate last).

    Audio continues unchanged — only the video timeline is shifted.
    Total frame count is preserved.

    Args:
        img_seq: ndarray (T, H, W, 3) in RGB.
        shift_frames: Signed int, number of frames to shift.
                      Positive = video ahead, negative = video behind.
                      If None, chosen randomly within [-max_shift, max_shift].
        max_shift: Maximum shift in frames (used only if shift_frames is None).
                   Default 10 frames = 400ms at 25fps.
        seed: Random seed.

    Returns:
        tuple (shifted_frames, shift_amount) where:
          shifted_frames: ndarray (T, H, W, 3) RGB, same length as input.
          shift_amount: int, the actual shift applied.
    """
    if seed is not None:
        random.seed(seed)

    T_len = img_seq.shape[0]

    if shift_frames is None:
        shift_frames = random.randint(-max_shift, max_shift)
        # Avoid zero shift (no desync = no degradation)
        if shift_frames == 0:
            shift_frames = 3 if random.random() < 0.5 else -3

    shift_frames = int(shift_frames)
    result = img_seq.copy()

    if shift_frames > 0:
        # Video AHEAD: pad beginning with first frame, truncate end
        pad = np.tile(img_seq[0:1], (shift_frames, 1, 1, 1))
        result = np.concatenate([pad, img_seq])[:T_len]
    elif shift_frames < 0:
        # Video LAGS: drop beginning, pad end with last frame
        drop = abs(shift_frames)
        pad = np.tile(img_seq[-1:], (drop, 1, 1, 1))
        result = np.concatenate([img_seq[drop:], pad])
        if len(result) > T_len:
            result = result[:T_len]

    return result, shift_frames


# ===================================================================
#  Unified apply function
# ===================================================================

def apply_degradation(
    img_seq,
    landmarks,
    params,
    object_dir=None,
    mask_dir=None,
    state="train",
    seed=None,
):
    """Apply a degradation described by DegradationParams.

    Args:
        img_seq: ndarray (T, H, W, 3) RGB.
        landmarks: ndarray (T, N, 2) or None (not needed for type 0/2/3/4).
        params: DegradationParams with fields:
                mask_start_frame, mask_len_frames, deg_type, shift_frames.
        object_dir: Path to object images.
        mask_dir: Path to mask images.
        state: "train" or "test".
        seed: Random seed.

    Returns:
        ndarray of degraded frames (RGB for most types, BGR for type 2).
        For type 4 (AV_DESYNC), also returns the shift amount as second value.
    """
    if seed is not None:
        random.seed(seed)

    deg_type = params.deg_type
    start_fr = params.mask_start_frame
    length = params.mask_len_frames

    if deg_type == 0:
        return apply_full_mask(img_seq, start_fr, length)

    elif deg_type == 1:
        if landmarks is None:
            raise ValueError("landmarks required for occlusion (deg_type=1)")
        return occlude_sequence(
            img_seq, landmarks,
            object_dir=object_dir, mask_dir=mask_dir,
            state=state, seed=seed,
        )[0]

    elif deg_type == 2:
        return occlude_sequence_noise(img_seq, seed=seed)

    elif deg_type == 3:
        return apply_frame_freeze(img_seq, start_fr, length, seed=seed)

    elif deg_type == 4:
        shift = getattr(params, 'shift_frames', 0)
        if shift == 0:
            shift = length  # fallback: use mask_len_frames as shift magnitude
        return apply_av_desync(img_seq, shift_frames=shift, seed=seed)[0]

    else:
        raise ValueError(f"Unknown degradation type: {deg_type}")


# ===================================================================
#  Video loading helpers
# ===================================================================

def preprocess(video_pathname, landmarks_pathname):
    """Load and preprocess a video + its facial landmarks.

    Args:
        video_pathname: str, path to video file.
        landmarks_pathname: str (pickle file) or list/ndarray of landmarks.

    Returns:
        tuple (sequence, landmarks) where:
          sequence: ndarray (T, H, W, 3) RGB frames.
          landmarks: ndarray (T, N, 2) interpolated landmarks.
        Returns (None, None, None, None) on failure.
    """
    from .utils import crop_patch

    # Load landmarks
    if isinstance(landmarks_pathname, str):
        with open(landmarks_pathname, "rb") as f:
            landmarks = pickle.load(f)
    else:
        landmarks = landmarks_pathname

    lm = landmarks_interpolate(landmarks)
    if lm is None or len(lm) < _WINDOW_MARGIN:
        return None, None, None, None

    sequence = crop_patch(video_pathname, lm)
    if sequence is None or len(sequence) == 0:
        return None, None, None, None

    return sequence, np.array(lm)


def load_video(video_path, landmarks_path):
    """Convenience wrapper around preprocess().

    Args:
        video_path: str, path to video file.
        landmarks_path: str or list/ndarray of landmarks.

    Returns:
        tuple (sequence, landmarks) — same as preprocess().
    """
    return preprocess(video_path, landmarks_path)
