"""
utils/voice_alert.py
====================
Voice-based alert system using pyttsx3 (offline, no internet needed).

Replaces / supplements the beep alarm with natural spoken warnings.
Runs speech synthesis on a dedicated daemon thread so it never
blocks the video pipeline.

Alerts are rate-limited (cooldown_s) to avoid rapid repetition.
"""

import threading
import time
import logging

logger = logging.getLogger(__name__)

# ── Alert messages ────────────────────────────────────────────────────────────
ALERTS = {
    "drowsy":     "Warning! You appear drowsy. Please stay alert.",
    "yawn":       "Yawn detected. Consider taking a break.",
    "eyes_closed": "Your eyes are closing. Wake up!",
    "no_face":    "Driver not detected. Please face the camera.",
    "low_light":  "Low light detected. Visibility may be reduced.",
}


class VoiceAlertSystem:
    """
    Thread-safe voice alert manager.

    Parameters
    ----------
    rate      : int    Speech rate in words per minute (default 160).
    volume    : float  Volume 0.0–1.0 (default 0.9).
    cooldown_s: float  Minimum seconds between same alert (default 8).
    enabled   : bool   Set False to silence all voice alerts.
    """

    def __init__(self,
                 rate: int = 160,
                 volume: float = 0.9,
                 cooldown_s: float = 8.0,
                 enabled: bool = True):
        self._rate       = rate
        self._volume     = volume
        self._cooldown   = cooldown_s
        self.enabled     = enabled

        # Track last time each alert was spoken
        self._last_spoken: dict[str, float] = {}
        self._lock  = threading.Lock()
        self._queue: list[str] = []
        self._speaking = False

        self._engine = None
        self._init_engine()

    # ── private ──────────────────────────────────────────────────────────────

    def _init_engine(self) -> None:
        """Initialise pyttsx3. Silently disables if unavailable."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   self._rate)
            engine.setProperty("volume", self._volume)
            # Prefer a female voice if available
            voices = engine.getProperty("voices")
            for v in voices:
                if "female" in v.name.lower() or "zira" in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break
            self._engine = engine
            logger.info("VoiceAlertSystem ready (rate=%d wpm).", self._rate)
        except Exception as exc:
            logger.warning("VoiceAlertSystem unavailable: %s", exc)
            self.enabled = False

    def _speak_worker(self, text: str) -> None:
        """Run in a daemon thread — synthesises and plays *text*."""
        try:
            if self._engine:
                self._engine.say(text)
                self._engine.runAndWait()
        except Exception as exc:
            logger.error("TTS error: %s", exc)
        finally:
            with self._lock:
                self._speaking = False

    # ── public API ────────────────────────────────────────────────────────────

    def speak(self, alert_key: str) -> None:
        """
        Speak the message for *alert_key* if cooldown has elapsed.

        Parameters
        ----------
        alert_key : str  Key from the ALERTS dict (e.g. "drowsy").
        """
        if not self.enabled or self._engine is None:
            return

        text = ALERTS.get(alert_key)
        if not text:
            return

        now = time.time()
        with self._lock:
            last = self._last_spoken.get(alert_key, 0.0)
            if now - last < self._cooldown:
                return           # still in cooldown
            if self._speaking:
                return           # another alert is playing
            self._last_spoken[alert_key] = now
            self._speaking = True

        thread = threading.Thread(
            target=self._speak_worker,
            args=(text,),
            daemon=True,
            name="VoiceAlert",
        )
        thread.start()
        logger.debug("Voice alert: '%s'", text)

    def speak_custom(self, text: str) -> None:
        """Speak an arbitrary *text* string (not rate-limited)."""
        if not self.enabled or self._engine is None:
            return
        with self._lock:
            if self._speaking:
                return
            self._speaking = True
        threading.Thread(
            target=self._speak_worker,
            args=(text,),
            daemon=True,
            name="VoiceAlert",
        ).start()

    def set_cooldown(self, seconds: float) -> None:
        """Update the per-alert cooldown at runtime."""
        self._cooldown = seconds

    def shutdown(self) -> None:
        """Stop the engine on application exit."""
        try:
            if self._engine:
                self._engine.stop()
        except Exception:
            pass
