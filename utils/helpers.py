"""
utils/helpers.py
================
General-purpose helper functions used across the system:

- FPS calculation
- Dlib shape → NumPy array conversion
- Frame annotation (text overlays, status panels)
- Screenshot saving
- Low-light detection
- Camera reconnect utility
"""

import time
import os
import logging
from datetime import datetime
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── FPS Calculator ─────────────────────────────────────────────────────────

class FPSCounter:
    """
    Rolling-average FPS counter.

    Parameters
    ----------
    window : int  Number of recent frame timestamps to average over.
    """

    def __init__(self, window: int = 30):
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """
        Call once per processed frame.

        Returns
        -------
        float  Current FPS (0.0 if fewer than 2 samples so far).
        """
        now = time.perf_counter()
        self._timestamps.append(now)

        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        """Most recent FPS estimate without advancing the counter."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ── Dlib ↔ NumPy conversion ────────────────────────────────────────────────

def shape_to_np(shape, dtype: type = np.int32) -> np.ndarray:
    """
    Convert a Dlib ``full_object_detection`` (68-point shape) to a
    NumPy array of shape (68, 2).

    Parameters
    ----------
    shape : dlib.full_object_detection
    dtype : NumPy dtype for the output array.

    Returns
    -------
    np.ndarray  Shape (68, 2)  — columns are [x, y].
    """
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# ── Frame annotation helpers ───────────────────────────────────────────────

def put_text(frame: np.ndarray,
             text: str,
             pos: tuple,
             color: tuple = (255, 255, 255),
             scale: float = 0.6,
             thickness: int = 2) -> None:
    """
    Draw anti-aliased text with a dark shadow for readability.

    Parameters
    ----------
    frame     : np.ndarray  BGR image (modified in place).
    text      : str
    pos       : tuple       (x, y) top-left origin of the text.
    color     : tuple       BGR foreground colour.
    scale     : float       Font scale.
    thickness : int         Line thickness.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Shadow (slight offset in dark colour for contrast)
    cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1),
                font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Foreground
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def draw_status_panel(frame: np.ndarray,
                      ear: float,
                      mar: float,
                      fps: float,
                      blink_count: int,
                      yawn_count: int,
                      drowsy: bool,
                      yawning: bool,
                      alarm_on: bool,
                      face_detected: bool,
                      low_light: bool) -> None:
    """
    Render a semi-transparent statistics panel in the top-left corner.

    Parameters are self-explanatory metrics from the detection pipeline.
    """
    # Semi-transparent background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 220), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Title bar
    cv2.rectangle(frame, (0, 0), (280, 24), (50, 50, 50), -1)
    put_text(frame, "DROWSINESS MONITOR", (6, 17),
             color=(200, 200, 200), scale=0.48, thickness=1)

    # EAR row
    ear_color = (0, 0, 255) if ear < 0.25 else (0, 255, 0)
    put_text(frame, f"EAR : {ear:.3f}", (8, 45),
             color=ear_color, scale=0.55)

    # MAR row
    mar_color = (0, 255, 255) if mar > 0.75 else (0, 255, 0)
    put_text(frame, f"MAR : {mar:.3f}", (8, 70),
             color=mar_color, scale=0.55)

    # FPS
    put_text(frame, f"FPS : {fps:.1f}", (8, 95),
             color=(255, 255, 255), scale=0.55)

    # Blink / Yawn counts
    put_text(frame, f"Blinks : {blink_count}", (8, 120),
             color=(200, 200, 200), scale=0.50)
    put_text(frame, f"Yawns  : {yawn_count}", (8, 145),
             color=(200, 200, 200), scale=0.50)

    # Status indicators
    face_color = (0, 255, 0) if face_detected else (0, 0, 255)
    face_label = "Face : DETECTED" if face_detected else "Face : NOT FOUND"
    put_text(frame, face_label, (8, 170), color=face_color, scale=0.50)

    if low_light:
        put_text(frame, "LOW LIGHT WARNING", (8, 195),
                 color=(0, 165, 255), scale=0.50)

    # Alarm banner
    if alarm_on:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 200), -1)
        put_text(frame, "⚠  DROWSINESS ALERT — WAKE UP!  ⚠",
                 (int(w * 0.05), h - 15),
                 color=(255, 255, 255), scale=0.70, thickness=2)

    # Yawn label (top-right)
    if yawning:
        h, w = frame.shape[:2]
        put_text(frame, "YAWN DETECTED",
                 (w - 200, 30),
                 color=(0, 255, 255), scale=0.65, thickness=2)


def draw_ear_bar(frame: np.ndarray,
                 ear: float,
                 threshold: float,
                 x: int = 290,
                 y: int = 30,
                 width: int = 120,
                 height: int = 14) -> None:
    """
    Draw a horizontal EAR confidence bar.

    Parameters
    ----------
    frame     : np.ndarray  BGR image (in place).
    ear       : float       Current EAR.
    threshold : float       Alert threshold.
    x, y      : int         Top-left corner of the bar.
    width     : int         Total bar width in pixels.
    height    : int         Bar height in pixels.
    """
    put_text(frame, "EAR level", (x, y - 4),
             color=(200, 200, 200), scale=0.40, thickness=1)

    # Background track
    cv2.rectangle(frame, (x, y), (x + width, y + height),
                  (60, 60, 60), -1)

    # Fill (clamped 0–0.5 → 0–width)
    fill_ratio = min(ear / 0.5, 1.0)
    fill_w     = int(fill_ratio * width)
    bar_color  = (0, 255, 0) if ear >= threshold else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + fill_w, y + height), bar_color, -1)

    # Threshold marker
    marker_x = x + int((threshold / 0.5) * width)
    cv2.line(frame, (marker_x, y - 2), (marker_x, y + height + 2),
             (255, 255, 0), 2)


# ── Screenshot ─────────────────────────────────────────────────────────────

def save_screenshot(frame: np.ndarray, save_dir: str) -> str:
    """
    Save the current frame as a timestamped PNG.

    Parameters
    ----------
    frame    : np.ndarray  BGR image.
    save_dir : str         Directory to save into.

    Returns
    -------
    str  Full path of the saved file.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath  = os.path.join(save_dir, f"screenshot_{timestamp}.png")
    cv2.imwrite(filepath, frame)
    logger.info("Screenshot saved → %s", filepath)
    return filepath


# ── Low-light detection ────────────────────────────────────────────────────

def is_low_light(frame: np.ndarray, threshold: int = 50) -> bool:
    """
    Return True when the mean brightness of the frame is below *threshold*.

    Parameters
    ----------
    frame     : np.ndarray  BGR image.
    threshold : int         Mean pixel value (0–255) below which → low light.

    Returns
    -------
    bool
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < threshold


# ── Camera utilities ───────────────────────────────────────────────────────

def open_camera(index: int,
                width: int = 640,
                height: int = 480) -> cv2.VideoCapture:
    """
    Open the webcam at *index* with the requested resolution.

    Raises
    ------
    RuntimeError  If the camera cannot be opened.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {index}.  "
            "Check that the webcam is connected and not in use."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    logger.info("Camera %d opened (%dx%d).", index, width, height)
    return cap


def reconnect_camera(cap: cv2.VideoCapture,
                     index: int,
                     width: int = 640,
                     height: int = 480,
                     max_attempts: int = 5,
                     delay_s: float = 2.0) -> cv2.VideoCapture:
    """
    Release *cap* and try to re-open the camera up to *max_attempts* times.

    Parameters
    ----------
    cap          : cv2.VideoCapture  Existing (broken) capture object.
    index        : int               Camera index to reopen.
    width, height: int               Desired resolution.
    max_attempts : int               Maximum reconnect attempts.
    delay_s      : float             Seconds to wait between attempts.

    Returns
    -------
    cv2.VideoCapture  A valid capture object.

    Raises
    ------
    RuntimeError  If all reconnect attempts fail.
    """
    cap.release()
    logger.warning("Camera lost — attempting reconnect (max %d tries).", max_attempts)

    for attempt in range(1, max_attempts + 1):
        logger.info("Reconnect attempt %d/%d …", attempt, max_attempts)
        time.sleep(delay_s)
        new_cap = cv2.VideoCapture(index)
        if new_cap.isOpened():
            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info("Camera reconnected successfully.")
            return new_cap
        new_cap.release()

    raise RuntimeError("Camera reconnect failed after maximum attempts.")
