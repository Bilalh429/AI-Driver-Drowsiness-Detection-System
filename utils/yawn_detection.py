"""
utils/yawn_detection.py
=======================
Mouth Aspect Ratio (MAR) computation and yawn detection logic.

The MAR is analogous to the EAR and measures how open the mouth is.
A high MAR value (mouth wide open) over several consecutive frames
is interpreted as a yawn.

Landmark indices used (from Dlib's 68-point model, 0-indexed):
  Outer mouth points: 48 – 67  (20 points)

For MAR we use a simplified 8-point representation:
  Top lip centre    : 51
  Bottom lip centre : 57
  Left corner       : 48
  Right corner      : 54
  (Plus two intermediate vertical pairs for robustness)
"""

from scipy.spatial import distance as dist
import numpy as np
import cv2


# ── Landmark index constants (relative to full shape array, 0-indexed) ──────
_MOUTH_LEFT   = 48   # left  corner
_MOUTH_RIGHT  = 54   # right corner
_MOUTH_TOP    = 51   # top   lip centre
_MOUTH_BOTTOM = 57   # bottom lip centre
_MOUTH_TL     = 50   # top-left  intermediate
_MOUTH_TR     = 52   # top-right intermediate
_MOUTH_BL     = 58   # bottom-left  intermediate
_MOUTH_BR     = 56   # bottom-right intermediate


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """
    Compute the Mouth Aspect Ratio (MAR).

    Uses four vertical measurement pairs for stability and one
    horizontal distance for normalisation, then averages.

    Parameters
    ----------
    mouth : np.ndarray
        Array of shape (20, 2) – the 20 outer-mouth landmark coordinates
        extracted from Dlib's 68-point shape (indices 48–67).

    Returns
    -------
    float
        MAR value.  ≈ 0.1–0.3 when closed; rises to 0.6–1.0+ when yawning.
    """
    # Remap to local indices (mouth[0] == landmark 48, etc.)
    left  = mouth[0]   # 48
    right = mouth[6]   # 54
    top1  = mouth[3]   # 51  (centre top)
    bot1  = mouth[9]   # 57  (centre bottom)
    top2  = mouth[2]   # 50
    bot2  = mouth[10]  # 58
    top3  = mouth[4]   # 52
    bot3  = mouth[8]   # 56

    # Vertical openings
    A = dist.euclidean(top1, bot1)
    B = dist.euclidean(top2, bot2)
    C = dist.euclidean(top3, bot3)

    # Horizontal width (normaliser)
    D = dist.euclidean(left, right)

    if D == 0:
        return 0.0

    mar = (A + B + C) / (3.0 * D)
    return mar


def draw_mouth_contour(frame: np.ndarray,
                       mouth: np.ndarray,
                       color: tuple = (0, 255, 255)) -> None:
    """
    Draw the convex hull around the mouth region on *frame* in place.

    Parameters
    ----------
    frame : np.ndarray   BGR image.
    mouth : np.ndarray   Shape (20, 2) mouth landmark coordinates.
    color : tuple        BGR colour for the contour.
    """
    hull = cv2.convexHull(mouth)
    cv2.drawContours(frame, [hull], -1, color, 1)


class YawnTracker:
    """
    Stateful tracker that counts consecutive yawn frames and
    maintains a total yawn count.

    Usage
    -----
    tracker = YawnTracker(threshold=0.75, consec_frames=15)
    is_yawning = tracker.update(current_mar)
    """

    def __init__(self, threshold: float, consec_frames: int):
        """
        Parameters
        ----------
        threshold     : float  MAR value above which mouth is "open".
        consec_frames : int    Consecutive open-mouth frames → confirmed yawn.
        """
        self.threshold     = threshold
        self.consec_frames = consec_frames

        self._frame_counter = 0
        self._yawning       = False
        self._yawn_count    = 0
        self._was_closed    = True  # tracks close→open transition

    # ── public API ──────────────────────────────────────────────────────────

    def update(self, mar: float) -> bool:
        """
        Feed the latest MAR value; returns True when a yawn is active.

        Parameters
        ----------
        mar : float  Mouth aspect ratio for the current frame.

        Returns
        -------
        bool  True → yawning detected.
        """
        mouth_open = mar > self.threshold

        if mouth_open:
            self._frame_counter += 1
            self._was_closed = False
        else:
            # Transition: open → closed = one yawn completed
            if not self._was_closed and self._yawning:
                self._yawn_count += 1
            self._frame_counter = 0
            self._yawning       = False
            self._was_closed    = True

        if self._frame_counter >= self.consec_frames:
            self._yawning = True

        return self._yawning

    @property
    def yawning(self) -> bool:
        """True while a yawn is ongoing."""
        return self._yawning

    @property
    def yawn_count(self) -> int:
        """Total completed yawns since tracker was created."""
        return self._yawn_count

    @property
    def frame_counter(self) -> int:
        """Consecutive frames with open mouth."""
        return self._frame_counter

    def reset(self) -> None:
        """Reset counters (e.g. when face is temporarily lost)."""
        self._frame_counter = 0
        self._yawning       = False
        self._was_closed    = True
