"""
utils/eye_detection.py
======================
Eye Aspect Ratio (EAR) calculation and eye-state analysis.

The EAR formula is taken from the paper:
  "Real-Time Eye Blink Detection using Facial Landmarks"
  by Tereza Soukupova and Jan Cech (2016).

Formula:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

where p1 … p6 are the six 2-D landmark coordinates of one eye
(ordered clockwise starting from the left corner).
"""

from scipy.spatial import distance as dist
import numpy as np
import cv2


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    Parameters
    ----------
    eye : np.ndarray
        Array of shape (6, 2) containing the (x, y) coordinates of
        the six eye landmarks in the standard Dlib ordering.

    Returns
    -------
    float
        EAR value. Typically 0.20–0.30 when open; drops below 0.20
        when closed.
    """
    # Vertical distances (||p2-p6|| and ||p3-p5||)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Horizontal distance (||p1-p4||)
    C = dist.euclidean(eye[0], eye[3])

    # Guard against division by zero (shouldn't happen, but safety first)
    if C == 0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return ear


def average_ear(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """
    Return the mean EAR across both eyes.

    Parameters
    ----------
    left_eye  : np.ndarray  Shape (6, 2)
    right_eye : np.ndarray  Shape (6, 2)

    Returns
    -------
    float
        Average EAR.
    """
    left_ear  = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0


def draw_eye_contours(frame: np.ndarray,
                      left_eye: np.ndarray,
                      right_eye: np.ndarray,
                      color: tuple = (0, 255, 0)) -> None:
    """
    Draw convex-hull contours around each eye on *frame* in place.

    Parameters
    ----------
    frame     : np.ndarray  BGR image to draw on.
    left_eye  : np.ndarray  Shape (6, 2)
    right_eye : np.ndarray  Shape (6, 2)
    color     : tuple       BGR colour for the contour lines.
    """
    left_hull  = cv2.convexHull(left_eye)
    right_hull = cv2.convexHull(right_eye)
    cv2.drawContours(frame, [left_hull],  -1, color, 1)
    cv2.drawContours(frame, [right_hull], -1, color, 1)


def is_eye_closed(ear: float, threshold: float) -> bool:
    """
    Return True when the EAR is below *threshold* (eyes considered closed).

    Parameters
    ----------
    ear       : float  Current average EAR.
    threshold : float  EAR_THRESHOLD from config.

    Returns
    -------
    bool
    """
    return ear < threshold


class EyeStateTracker:
    """
    Stateful tracker that counts consecutive frames where eyes are closed
    and manages the drowsiness flag.

    Usage
    -----
    tracker = EyeStateTracker(threshold=0.25, consec_frames=20)
    is_drowsy = tracker.update(current_ear)
    """

    def __init__(self, threshold: float, consec_frames: int):
        """
        Parameters
        ----------
        threshold     : float  EAR value below which eyes are "closed".
        consec_frames : int    How many consecutive closed frames trigger drowsiness.
        """
        self.threshold     = threshold
        self.consec_frames = consec_frames

        self._frame_counter = 0     # consecutive closed-eye frames
        self._drowsy        = False  # current drowsiness state
        self._blink_count   = 0      # total blinks detected
        self._was_open      = True   # tracks open→close transition for blink counting

    # ── public API ──────────────────────────────────────────────────────────

    def update(self, ear: float) -> bool:
        """
        Feed the latest EAR value; returns True when drowsiness is active.

        Parameters
        ----------
        ear : float  Average EAR for the current frame.

        Returns
        -------
        bool  True → driver is drowsy, False → driver is alert.
        """
        eyes_closed = ear < self.threshold

        if eyes_closed:
            self._frame_counter += 1
            # Count short closures as blinks (< consec_frames)
            if self._was_open:
                self._was_open = False
        else:
            # Transition: closed → open = one blink
            if not self._was_open and self._frame_counter < self.consec_frames:
                self._blink_count += 1
            self._frame_counter = 0
            self._drowsy        = False
            self._was_open      = True

        if self._frame_counter >= self.consec_frames:
            self._drowsy = True

        return self._drowsy

    @property
    def frame_counter(self) -> int:
        """Number of consecutive frames with closed eyes."""
        return self._frame_counter

    @property
    def drowsy(self) -> bool:
        """Current drowsiness flag."""
        return self._drowsy

    @property
    def blink_count(self) -> int:
        """Total blinks detected since tracker was created."""
        return self._blink_count

    def reset(self) -> None:
        """Reset all counters (e.g. when the face is lost)."""
        self._frame_counter = 0
        self._drowsy        = False
        self._was_open      = True
