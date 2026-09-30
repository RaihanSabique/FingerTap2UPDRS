"""Streaming video reader with per-frame timestamps (never holds the whole clip in memory)."""
import cv2
import numpy as np


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def iter_frames(path, max_side=None, max_seconds=None):
    """Yield (index, time_s, BGR frame). OpenCV applies the phone's rotation flag itself.

    Timestamps come from the demuxer so variable-frame-rate phone clips keep true timing;
    if the backend reports nothing useful we fall back to index / fps.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last_t = 0, -1.0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= last_t or (i > 0 and t == 0):
            t = i / fps
        last_t = t
        if max_seconds is not None and t > max_seconds:
            break
        if max_side:
            h, w = img.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        yield i, t, img
        i += 1
    cap.release()
