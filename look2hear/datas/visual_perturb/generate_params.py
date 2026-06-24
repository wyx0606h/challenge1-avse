"""
Parameter generation for visual degradation.

Generates random degradation parameters that describe WHAT type of
degradation to apply, WHERE in the video, and for HOW LONG.
These params are meant to be stored (e.g. in CSV) and then consumed
by the perturbation functions in perturb.py at training/inference time.
"""

import random
import csv
from dataclasses import dataclass
from enum import IntEnum


class DegradationType(IntEnum):
    """Types of visual degradation."""
    FULL_MASK = 0        # Blackout entire frame (simulates frame drop / padding to 0)
    OCCLUSION = 1        # Object occlusion on face landmarks
    LOW_RESOLUTION = 2   # Gaussian noise or blur
    FRAME_FREEZE = 3     # Frame freezing/stutter (one frame repeats, audio continues)
    AV_DESYNC = 4        # Audio-visual desynchronization (video shifted vs audio)


@dataclass
class DegradationParams:
    """Parameters for a single speaker's visual degradation.

    Fields vary by degradation type:
      FULL_MASK (0):    mask_start_frame, mask_len_frames
      OCCLUSION (1):    mask_start_frame, mask_len_frames
      LOW_RESOLUTION (2): mask_start_frame, mask_len_frames
      FRAME_FREEZE (3):  mask_start_frame, mask_len_frames (= freeze duration)
      AV_DESYNC (4):     shift_frames (positive = video ahead, negative = video lags)
    """
    mask_start_frame: int = 0   # Frame index where degradation begins
    mask_len_frames: int = 0    # Number of frames to degrade (or freeze duration)
    deg_type: int = 0           # DegradationType
    shift_frames: int = 0       # AV desync shift amount in frames (+ = video ahead)


# Weights for degradation type selection (same as in DEGRADATION_SUMMARY.md)
_DEGRADATION_WEIGHTS = {
    DegradationType.LOW_RESOLUTION: 0.35,
    DegradationType.OCCLUSION: 0.25,
    DegradationType.FRAME_FREEZE: 0.18,
    DegradationType.AV_DESYNC: 0.12,
    DegradationType.FULL_MASK: 0.10,
}


def generate_degradation_params(
    total_frames: int,
    seed: int = None,
) -> DegradationParams:
    """Generate random degradation parameters for a single speaker.

    Every call returns a degradation — no "clean" path.
    Degradation type chosen with weighted probabilities:
      LOW_RESOLUTION: 35%, OCCLUSION: 25%, FRAME_FREEZE: 18%,
      AV_DESYNC: 12%, FULL_MASK: 10%.

    Args:
        total_frames: Total number of frames in the video.
        seed: Optional random seed for reproducibility.

    Returns:
        DegradationParams with appropriate fields for the chosen type.
    """
    if seed is not None:
        random.seed(seed)

    types = list(_DEGRADATION_WEIGHTS.keys())
    weights = list(_DEGRADATION_WEIGHTS.values())
    deg_type = random.choices(types, weights=weights, k=1)[0]

    if deg_type == DegradationType.AV_DESYNC:
        # Shift: ±1 to ±10 frames (±40ms ~ ±400ms at 25fps)
        max_shift = 10
        shift = random.randint(-max_shift, max_shift)
        # Avoid zero shift
        if shift == 0:
            shift = 3 if random.random() < 0.5 else -3
        return DegradationParams(
            mask_start_frame=0,
            mask_len_frames=abs(shift),
            deg_type=int(deg_type),
            shift_frames=shift,
        )

    elif deg_type == DegradationType.FRAME_FREEZE:
        # Freeze duration: 3 to max(3, 30% of video) frames
        max_freeze = max(3, int(total_frames * 0.3))
        freeze_len = random.randint(3, max_freeze)
        start_fr = random.randint(0, max(0, total_frames - freeze_len))
        return DegradationParams(
            mask_start_frame=start_fr,
            mask_len_frames=freeze_len,
            deg_type=int(deg_type),
        )

    else:
        # Types 0, 1, 2: position-based degradation, 20%~80% of frames
        mask_ratio = round(random.uniform(0.2, 0.8), 2)

        mask_len_frames = int(mask_ratio * total_frames)
        mask_start_frame = random.randint(0, max(0, total_frames - mask_len_frames))

        return DegradationParams(
            mask_start_frame=mask_start_frame,
            mask_len_frames=mask_len_frames,
            deg_type=int(deg_type),
        )


def generate_degradation_params_for_mix(
    num_speakers: int,
    total_frames_list: list,
    seed: int = None,
) -> list:
    """Generate degradation params for a multi-speaker mixture.

    Args:
        num_speakers: Number of speakers in the mix.
        total_frames_list: List of frame counts per speaker.
        seed: Base random seed.

    Returns:
        List of DegradationParams, one per speaker.
    """
    if seed is not None:
        random.seed(seed)

    params_list = []
    for i in range(num_speakers):
        params = generate_degradation_params(
            total_frames=total_frames_list[i],
            seed=None,  # Don't reseed per speaker
        )
        params_list.append(params)
    return params_list
