"""
Visual Perturbation Module
===========================
Online video augmentation for robustness training:
  - Object occlusion (overlay COCO objects on face landmarks)
  - Gaussian noise / blur (low-resolution simulation)
  - Full blackout mask (frame dropping / zero padding)
  - Frame freeze / stutter (video freezes, audio continues)
  - AV desynchronization (video shifted vs audio)

Usage:
    from visual_perturb import (
        occlude_sequence,
        occlude_sequence_noise,
        apply_full_mask,
        apply_frame_freeze,
        apply_av_desync,
        apply_degradation,
        load_video,
        write_video,
        generate_degradation_params,
    )
"""

from .perturb import (
    overlay_image_alpha,
    get_occluder_augmentor,
    get_occluders,
    occlude_sequence,
    occlude_sequence_noise,
    apply_full_mask,
    apply_frame_freeze,
    apply_av_desync,
    apply_degradation,
    landmarks_interpolate,
    linear_interpolate,
    load_video,
    preprocess,
)

from .utils import (
    read_video,
    write_video,
    read_video_frames,
    crop_patch,
)

from .generate_params import (
    generate_degradation_params,
    DegradationType,
    DegradationParams,
)

__version__ = "1.1.0"
