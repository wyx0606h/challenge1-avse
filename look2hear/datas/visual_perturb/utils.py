"""
Video perturbation utilities: read/write video files, load frames.
"""

import cv2
import numpy as np


def read_video(filename):
    """Generator that yields frames from a video file (BGR format).

    Args:
        filename: str, path to the video file.

    Yields:
        ndarray: BGR frame, shape (H, W, 3).
    """
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break
    cap.release()


def read_video_frames(filename):
    """Read all frames from a video file into a numpy array.

    Args:
        filename: str, path to the video file.

    Returns:
        ndarray: shape (T, H, W, 3) in RGB format.
    """
    frames = []
    for frame in read_video(filename):
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.array(frames)


def write_video(frames, filename, fps=25):
    """Write a sequence of frames to a video file.

    Args:
        frames: ndarray, shape (T, H, W, 3) in BGR format.
        filename: str, output path.
        fps: int, frames per second (default 25).
    """
    if len(frames.shape) != 4:
        raise ValueError(f"Expected 4D array (T, H, W, C), got shape {frames.shape}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frames.shape[1], frames.shape[2]
    output = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    for frame in frames:
        output.write(frame)
    output.release()


def crop_patch(video_pathname, landmarks):
    """Read video frames and return as RGB sequence.

    Args:
        video_pathname: str, path to the video file.
        landmarks: list, landmark data (kept for API compatibility).

    Returns:
        ndarray: shape (T, H, W, 3) in RGB.
    """
    frames = []
    for frame in read_video(video_pathname):
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return np.array(frames)
