"""
utils/logger.py
===============
Detection-event logging to CSV and Python's standard logging module.

CSV columns
-----------
timestamp       : ISO-8601 datetime string
ear             : Eye Aspect Ratio (float, 4 decimal places)
mar             : Mouth Aspect Ratio (float, 4 decimal places)
eyes_closed     : Boolean – True when EAR < threshold
yawning         : Boolean – True when yawn detected
drowsy          : Boolean – True when drowsiness alarm active
alarm_triggered : Boolean – True when alarm sound is playing
fps             : Frames per second at time of log entry
"""

import csv
import logging
import os
from datetime import datetime
from typing import Optional

# Module-level Python logger (for console / file log output)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_py_logger = logging.getLogger("DrowsinessSystem")

# CSV column header
_CSV_HEADER = [
    "timestamp",
    "ear",
    "mar",
    "eyes_closed",
    "yawning",
    "drowsy",
    "alarm_triggered",
    "fps",
]


class DetectionLogger:
    """
    Appends detection events to a CSV file and writes to Python's
    standard logging system.

    Parameters
    ----------
    log_path : str   Full path to the CSV file.
    """

    def __init__(self, log_path: str):
        self._log_path = log_path
        self._ensure_file()

    # ── private ──────────────────────────────────────────────────────────────

    def _ensure_file(self) -> None:
        """Create the CSV with a header row if it does not exist."""
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

        if not os.path.isfile(self._log_path):
            with open(self._log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(_CSV_HEADER)
            _py_logger.info("Log file created → %s", self._log_path)

    # ── public API ───────────────────────────────────────────────────────────

    def log_event(
        self,
        ear: float,
        mar: float,
        eyes_closed: bool,
        yawning: bool,
        drowsy: bool,
        alarm_triggered: bool,
        fps: float,
    ) -> None:
        """
        Append one detection row to the CSV.

        Parameters
        ----------
        ear             : float   Eye Aspect Ratio.
        mar             : float   Mouth Aspect Ratio.
        eyes_closed     : bool    Whether eyes are below EAR threshold.
        yawning         : bool    Whether yawning is currently detected.
        drowsy          : bool    Whether the driver is drowsy.
        alarm_triggered : bool    Whether the alarm is active.
        fps             : float   Current frames-per-second.
        """
        row = [
            datetime.now().isoformat(timespec="milliseconds"),
            f"{ear:.4f}",
            f"{mar:.4f}",
            eyes_closed,
            yawning,
            drowsy,
            alarm_triggered,
            f"{fps:.1f}",
        ]

        try:
            with open(self._log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except OSError as exc:
            _py_logger.error("Failed to write log row: %s", exc)

        # Mirror important events to the Python logger
        if drowsy:
            _py_logger.warning(
                "DROWSY | EAR=%.4f  MAR=%.4f  alarm=%s  fps=%.1f",
                ear, mar, alarm_triggered, fps,
            )
        elif yawning:
            _py_logger.info(
                "YAWN   | EAR=%.4f  MAR=%.4f  fps=%.1f", ear, mar, fps
            )

    def get_summary(self) -> dict:
        """
        Read the CSV and return aggregate statistics.

        Returns
        -------
        dict with keys: total_events, drowsy_events, yawn_events,
                        alarm_events, avg_ear, avg_mar.
        """
        stats = {
            "total_events": 0,
            "drowsy_events": 0,
            "yawn_events": 0,
            "alarm_events": 0,
            "avg_ear": 0.0,
            "avg_mar": 0.0,
        }

        if not os.path.isfile(self._log_path):
            return stats

        ear_vals: list[float] = []
        mar_vals: list[float] = []

        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats["total_events"] += 1
                    if row.get("drowsy") == "True":
                        stats["drowsy_events"] += 1
                    if row.get("yawning") == "True":
                        stats["yawn_events"] += 1
                    if row.get("alarm_triggered") == "True":
                        stats["alarm_events"] += 1
                    try:
                        ear_vals.append(float(row["ear"]))
                        mar_vals.append(float(row["mar"]))
                    except (KeyError, ValueError):
                        pass

        except OSError as exc:
            _py_logger.error("Failed to read log file: %s", exc)

        if ear_vals:
            stats["avg_ear"] = sum(ear_vals) / len(ear_vals)
        if mar_vals:
            stats["avg_mar"] = sum(mar_vals) / len(mar_vals)

        return stats


# ── Module-level convenience accessor ────────────────────────────────────────

def get_logger(name: str = "DrowsinessSystem") -> logging.Logger:
    """Return a named Python logger (used by other modules)."""
    return logging.getLogger(name)
