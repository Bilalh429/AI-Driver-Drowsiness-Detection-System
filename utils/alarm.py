"""
utils/alarm.py
==============
Thread-safe alarm management using Pygame's mixer module.

The AlarmManager runs the sound playback on a dedicated daemon thread
so it never blocks the main video-processing loop.  It also generates
a synthetic beep WAV file if the configured alarm.wav is missing.
"""

import threading
import logging
import os
import wave
import struct
import math

logger = logging.getLogger(__name__)


# ── Synthetic beep generation ────────────────────────────────────────────────

def generate_beep_wav(filepath: str,
                      frequency: int = 880,
                      duration_s: float = 1.0,
                      volume: float = 0.8,
                      sample_rate: int = 44100) -> None:
    """
    Write a simple sine-wave beep to *filepath* as a 16-bit mono WAV.

    Called automatically when alarm.wav is absent so the system works
    out-of-the-box without an external sound file.

    Parameters
    ----------
    filepath    : str    Destination path for the WAV file.
    frequency   : int    Tone frequency in Hz (default 880 = A5).
    duration_s  : float  Duration in seconds.
    volume      : float  Amplitude 0.0–1.0.
    sample_rate : int    Samples per second.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    num_samples = int(sample_rate * duration_s)
    max_amp     = int(32767 * volume)

    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            # Sine wave with simple fade-out to avoid clicks
            t     = i / sample_rate
            fade  = 1.0 - (i / num_samples) * 0.3
            value = int(max_amp * fade * math.sin(2 * math.pi * frequency * t))
            wf.writeframes(struct.pack("<h", value))

    logger.info("Synthetic alarm WAV written → %s", filepath)


# ── Alarm Manager ─────────────────────────────────────────────────────────────

class AlarmManager:
    """
    Manages alarm state and playback on a background thread.

    Usage
    -----
    alarm = AlarmManager(sound_path="alarm/alarm.wav")
    alarm.start()          # when drowsiness detected
    alarm.stop()           # when driver is alert again
    alarm.shutdown()       # on application exit
    """

    def __init__(self, sound_path: str):
        """
        Parameters
        ----------
        sound_path : str  Path to the WAV alarm file.
        """
        self._sound_path = sound_path
        self._active     = False          # True while alarm should play
        self._lock       = threading.Lock()
        self._thread: threading.Thread | None = None

        self._init_pygame()

    # ── private ──────────────────────────────────────────────────────────────

    def _init_pygame(self) -> None:
        """Initialise Pygame mixer; generate beep if WAV is missing."""
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self._pygame = pygame

            # Generate fallback beep if file is absent
            if not os.path.isfile(self._sound_path):
                logger.warning(
                    "Alarm file not found: %s — generating synthetic beep.",
                    self._sound_path,
                )
                generate_beep_wav(self._sound_path)

            self._sound = pygame.mixer.Sound(self._sound_path)
            logger.info("AlarmManager ready (sound: %s)", self._sound_path)

        except Exception as exc:
            logger.error("AlarmManager init failed: %s", exc)
            self._pygame = None
            self._sound  = None

    def _playback_loop(self) -> None:
        """
        Internal loop that runs on a daemon thread.
        Plays the alarm on repeat while self._active is True.
        """
        if self._sound is None:
            return

        self._sound.play(loops=-1)      # loop indefinitely
        logger.debug("Alarm playback started.")

        while True:
            with self._lock:
                if not self._active:
                    break
            threading.Event().wait(0.1)   # yield; check every 100 ms

        self._sound.stop()
        logger.debug("Alarm playback stopped.")

    # ── public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Activate the alarm.  No-op if already playing.
        Safe to call from any thread.
        """
        with self._lock:
            if self._active:
                return
            self._active = True

        self._thread = threading.Thread(
            target=self._playback_loop,
            daemon=True,
            name="AlarmThread",
        )
        self._thread.start()
        logger.info("Alarm STARTED.")

    def stop(self) -> None:
        """
        Deactivate the alarm.  No-op if not playing.
        Safe to call from any thread.
        """
        with self._lock:
            if not self._active:
                return
            self._active = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)

        logger.info("Alarm STOPPED.")

    def shutdown(self) -> None:
        """
        Stop the alarm and uninitialise Pygame mixer.
        Call on application exit.
        """
        self.stop()
        if self._pygame:
            try:
                self._pygame.mixer.quit()
            except Exception:
                pass

    @property
    def is_active(self) -> bool:
        """True while the alarm is playing."""
        with self._lock:
            return self._active
