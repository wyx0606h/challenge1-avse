"""
Example usage of the visual_perturb module.

Demonstrates all five degradation types:
  0. Full blackout mask (frame drop / zero padding)
  1. Object occlusion on face landmarks
  2. Gaussian noise / blur (low resolution)
  3. Frame freeze / stutter (video freezes, audio continues)
  4. AV desynchronization (video shifted vs audio)
"""

import numpy as np
from visual_perturb import (
    occlude_sequence,
    occlude_sequence_noise,
    apply_full_mask,
    apply_frame_freeze,
    apply_av_desync,
    apply_degradation,
    generate_degradation_params,
    load_video,
    write_video,
    read_video_frames,
    DegradationType,
    DegradationParams,
)


def example_full_mask():
    """Type 0: Set a block of frames to zero."""
    print("=" * 50)
    print("Type 0: Full blackout mask")
    print("=" * 50)

    # Simulate a 100-frame video (100, 112, 112, 3)
    fake_video = np.random.randint(0, 255, (100, 112, 112, 3), dtype=np.uint8)

    params = generate_degradation_params(total_frames=100, seed=42)
    print(f"Params: start={params.mask_start_frame}, "
          f"len={params.mask_len_frames}, type={DegradationType(params.deg_type).name}")

    result = apply_full_mask(fake_video, params.mask_start_frame, params.mask_len_frames)
    # Verify
    masked_region = result[params.mask_start_frame:params.mask_start_frame + params.mask_len_frames]
    assert np.all(masked_region == 0), "Full mask should set frames to 0!"
    print("✓ Full mask applied successfully\n")


def example_noise_blur():
    """Type 2: Apply Gaussian noise or blur."""
    print("=" * 50)
    print("Type 2: Noise / Blur (low resolution)")
    print("=" * 50)

    fake_video = np.random.randint(0, 255, (100, 112, 112, 3), dtype=np.uint8)

    # Noise mode
    result = occlude_sequence_noise(fake_video, freq=1, seed=123)
    print(f"Input shape: {fake_video.shape}, Output shape: {result.shape}")
    print(f"Output format: BGR (as converted by the function)")
    print("✓ Noise/blur applied successfully\n")


def example_occlusion():
    """Type 1: Object occlusion on landmarks."""
    print("=" * 50)
    print("Type 1: Object occlusion")
    print("=" * 50)

    fake_video = np.random.randint(0, 255, (100, 112, 112, 3), dtype=np.uint8)
    # Simulate 68-point face landmarks moving slightly each frame
    landmarks = np.array([
        [[40 + i * 0.1 + j * 3, 40 + j * 2] for j in range(68)]
        for i in range(100)
    ])

    result, obj_name = occlude_sequence(
        fake_video, landmarks, freq=1, state="train", seed=456
    )
    print(f"Occluder used: {obj_name}")
    print(f"Output shape: {result.shape}")
    print("✓ Object occlusion applied successfully\n")


def example_unified_api():
    """Demonstrate the unified apply_degradation() API."""
    print("=" * 50)
    print("Unified API: apply_degradation()")
    print("=" * 50)

    fake_video = np.random.randint(0, 255, (100, 112, 112, 3), dtype=np.uint8)
    landmarks = np.array([
        [[40 + i * 0.1 + j * 3, 40 + j * 2] for j in range(68)]
        for i in range(100)
    ])

    all_types = [
        DegradationType.FULL_MASK,
        DegradationType.OCCLUSION,
        DegradationType.LOW_RESOLUTION,
        DegradationType.FRAME_FREEZE,
        DegradationType.AV_DESYNC,
    ]
    for deg_type in all_types:
        params = DegradationParams(
            mask_start_frame=20,
            mask_len_frames=30,
            deg_type=int(deg_type),
        )
        result = apply_degradation(fake_video, landmarks, params, seed=42)
        print(f"  {deg_type.name:20s} → shape {result.shape}")
    print("✓ Unified API works for all types\n")


def example_frame_freeze():
    """Type 3: Freeze a video segment by repeating one frame."""
    print("=" * 50)
    print("Type 3: Frame freeze / stutter")
    print("=" * 50)

    # Create a video where each frame has a different value so we can detect freezing
    fake_video = np.zeros((100, 112, 112, 3), dtype=np.uint8)
    for i in range(100):
        fake_video[i, :, :, 0] = i  # Red channel = frame index

    result = apply_frame_freeze(fake_video, start_frame=30, freeze_len=10, seed=42)

    # Verify: frames 30-39 should all be identical to frame 30
    frozen_block = result[30:40]
    ref = result[30]
    is_frozen = np.all(frozen_block == ref)
    print(f"Frames 30-39 frozen = {is_frozen}")

    # Frames before and after should differ
    before_differs = result[29, 0, 0, 0] != result[30, 0, 0, 0]
    after_differs = result[40, 0, 0, 0] != result[39, 0, 0, 0]
    print(f"Edge continuity: before ok={before_differs}, after ok={after_differs}")
    print("✓ Frame freeze applied successfully\n")


def example_av_desync():
    """Type 4: Shift video relative to audio."""
    print("=" * 50)
    print("Type 4: AV desynchronization")
    print("=" * 50)

    # Create a video where red channel = frame index
    fake_video = np.zeros((100, 112, 112, 3), dtype=np.uint8)
    for i in range(100):
        fake_video[i, :, :, 0] = i

    # Test: video AHEAD by 5 frames
    result_ahead, shift_ahead = apply_av_desync(fake_video, shift_frames=5, seed=42)
    # First 5 frames should all be frame 0 (duplicated)
    print(f"Video AHEAD by {shift_ahead} frames:")
    print(f"  Frame 0 red value: {result_ahead[0, 0, 0, 0]}")
    print(f"  Frame 4 red value: {result_ahead[4, 0, 0, 0]}")
    print(f"  Frame 5 red value: {result_ahead[5, 0, 0, 0]} (should be 0 = original frame 0)")
    print(f"  Frame 50 red value: {result_ahead[50, 0, 0, 0]} (should be 45 = original frame 45)")
    # Frame[5] should be original frame[0] (shifted)
    # Frame[50] should be original frame[45]
    assert result_ahead[5, 0, 0, 0] == 0, "Frame 5 should be original frame 0"
    assert result_ahead[50, 0, 0, 0] == 45, "Frame 50 should be original frame 45"

    # Test: video LAGS by 5 frames
    result_lag, shift_lag = apply_av_desync(fake_video, shift_frames=-5, seed=42)
    # Last 5 frames should all be frame 99 (duplicated)
    print(f"\nVideo LAGS by {abs(shift_lag)} frames:")
    print(f"  Frame 95 red value: {result_lag[95, 0, 0, 0]} (should be 99 = last frame)")
    print(f"  Frame 0 red value: {result_lag[0, 0, 0, 0]} (should be 5 = original frame 5)")
    assert result_lag[0, 0, 0, 0] == 5, "Frame 0 should be original frame 5"
    assert result_lag[95, 0, 0, 0] == 99, "Frame 95 should be original frame 99"
    print("✓ AV desync applied successfully\n")


def example_with_real_video():
    """Example: load a real video, apply degradation, save result."""
    print("=" * 50)
    print("Real video example (run with actual paths)")
    print("=" * 50)
    print("""
    # Load video and landmarks
    frames, landmarks = load_video("input.mp4", "landmarks.pkl")

    # Generate random degradation params
    params = generate_degradation_params(len(frames))

    # Apply degradation
    degraded = apply_degradation(frames, landmarks, params)

    # Save result (convert RGB -> BGR for video writing)
    import cv2
    degraded_bgr = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in degraded])
    write_video(degraded_bgr, "output_degraded.mp4")
    print("Saved degraded video to output_degraded.mp4")
    """)


if __name__ == "__main__":
    example_full_mask()
    example_noise_blur()
    example_occlusion()
    example_frame_freeze()
    example_av_desync()
    example_unified_api()
    example_with_real_video()
    print("All examples completed!")
