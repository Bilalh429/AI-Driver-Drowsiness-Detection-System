"""
utils/night_vision.py
=====================
Low-light image enhancement for improved drowsiness detection in dark conditions.

Two enhancement modes
---------------------
1. CLAHE  (Contrast Limited Adaptive Histogram Equalisation)
   Best for: structured indoor/night lighting.
   Works on the L channel of LAB colour space to preserve colour.

2. Gamma  (power-law correction)
   Best for: uniformly dim but clean images.
   Brightens midtones without blowing out highlights.

Auto-mode selects CLAHE or gamma based on mean frame brightness.

Usage
-----
    enhancer = NightVisionEnhancer()
    enhanced_frame, was_enhanced = enhancer.process(frame)
"""

import cv2
import numpy as np
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class EnhancementMode(Enum):
    OFF    = auto()   # passthrough
    CLAHE  = auto()   # adaptive histogram equalisation
    GAMMA  = auto()   # gamma correction
    AUTO   = auto()   # auto-select based on brightness


class NightVisionEnhancer:
    """
    Real-time low-light frame enhancer.

    Parameters
    ----------
    mode              : EnhancementMode  Processing mode (default AUTO).
    low_light_thresh  : int    Mean brightness below this triggers enhancement (default 60).
    clahe_clip        : float  CLAHE clip limit (default 3.0).
    clahe_tile        : tuple  CLAHE tile grid size (default (8,8)).
    gamma             : float  Gamma correction value (default 1.8).  >1 = brighter.
    overlay_label     : bool   Draw "NIGHT VISION ON" label on enhanced frames.
    """

    def __init__(
        self,
        mode: EnhancementMode    = EnhancementMode.AUTO,
        low_light_thresh: int    = 60,
        clahe_clip: float        = 3.0,
        clahe_tile: tuple        = (8, 8),
        gamma: float             = 1.8,
        overlay_label: bool      = True,
    ):
        self._mode        = mode
        self._thresh      = low_light_thresh
        self._gamma       = gamma
        self._overlay     = overlay_label

        # Build CLAHE object (works on single-channel images)
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=clahe_tile,
        )

        # Pre-compute gamma lookup table for speed (uint8 → uint8)
        self._gamma_lut = self._build_gamma_lut(gamma)

        # Statistics
        self._frames_enhanced = 0
        self._frames_total    = 0

        logger.info("NightVisionEnhancer ready (mode=%s, thresh=%d).",
                    mode.name, low_light_thresh)

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """
        Build a 256-entry lookup table for gamma correction.

        corrected = (pixel / 255) ^ (1/gamma) * 255
        """
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return table

    def _mean_brightness(self, frame: np.ndarray) -> float:
        """Return the mean pixel intensity of the grayscale version of *frame*."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on the L channel of LAB colour space.

        This brightens dark areas without distorting colours.
        """
        lab        = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        l_enhanced = self._clahe.apply(l)
        enhanced   = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _apply_gamma(self, frame: np.ndarray) -> np.ndarray:
        """Apply gamma correction via LUT (very fast — no per-pixel math)."""
        return cv2.LUT(frame, self._gamma_lut)

    def _apply_denoising(self, frame: np.ndarray) -> np.ndarray:
        """
        Light non-local means denoising.

        Called after enhancement because amplifying dark pixels also
        amplifies sensor noise.  h=5 is fast enough for real-time use.
        """
        return cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)

    def _draw_overlay(self, frame: np.ndarray,
                      brightness: float,
                      mode_used: EnhancementMode) -> None:
        """Draw a subtle night-vision badge on the top-right corner."""
        h, w = frame.shape[:2]
        label = f"NIGHT VISION ({mode_used.name})"
        font  = cv2.FONT_HERSHEY_SIMPLEX

        # Background pill
        (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
        x1, y1 = w - tw - 16, 6
        x2, y2 = w - 6,       th + 14
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 80, 20), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)

        cv2.putText(frame, label, (x1 + 4, y2 - 5),
                    font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Brightness readout
        bri_txt = f"Brightness: {brightness:.0f}"
        cv2.putText(frame, bri_txt, (x1 + 4, y2 + 14),
                    font, 0.38, (100, 255, 100), 1, cv2.LINE_AA)

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> tuple:
        """
        Enhance *frame* if low-light conditions are detected.

        Parameters
        ----------
        frame : np.ndarray  BGR image.

        Returns
        -------
        tuple (enhanced_frame, was_enhanced)
            enhanced_frame : np.ndarray  Enhanced or original BGR frame.
            was_enhanced   : bool        True if enhancement was applied.
        """
        self._frames_total += 1
        brightness = self._mean_brightness(frame)

        # Determine which mode to apply
        if self._mode == EnhancementMode.OFF:
            return frame, False

        if self._mode == EnhancementMode.AUTO:
            if brightness >= self._thresh:
                return frame, False           # bright enough — skip
            # Dark: pick CLAHE for very dark, gamma for moderately dark
            mode_used = (EnhancementMode.CLAHE
                         if brightness < self._thresh * 0.5
                         else EnhancementMode.GAMMA)
        else:
            mode_used = self._mode
            if self._mode != EnhancementMode.OFF and brightness >= self._thresh:
                return frame, False

        # Apply chosen enhancement
        if mode_used == EnhancementMode.CLAHE:
            enhanced = self._apply_clahe(frame)
        else:
            enhanced = self._apply_gamma(frame)

        self._frames_enhanced += 1

        if self._overlay:
            self._draw_overlay(enhanced, brightness, mode_used)

        return enhanced, True

    def set_mode(self, mode: EnhancementMode) -> None:
        """Change enhancement mode at runtime."""
        self._mode = mode
        logger.info("NightVisionEnhancer mode set to %s.", mode.name)

    def set_gamma(self, gamma: float) -> None:
        """Update gamma correction value and rebuild LUT."""
        self._gamma     = gamma
        self._gamma_lut = self._build_gamma_lut(gamma)

    @property
    def stats(self) -> dict:
        """Return enhancement statistics."""
        pct = (self._frames_enhanced / self._frames_total * 100
               if self._frames_total else 0)
        return {
            "frames_total":    self._frames_total,
            "frames_enhanced": self._frames_enhanced,
            "enhancement_pct": round(pct, 1),
        }

    @property
    def is_low_light_mode(self) -> bool:
        """True if the last processed frame triggered enhancement."""
        return self._frames_enhanced > 0 and (
            self._frames_enhanced == self._frames_total or
            self._frames_enhanced == self._frames_total - 1
        )
