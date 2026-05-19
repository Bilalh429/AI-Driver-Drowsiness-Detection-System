"""
main.py
=======
AI-Based Driver Drowsiness Detection System
============================================

Entry point.  Orchestrates:
  - Webcam capture (background thread)
  - Face/landmark detection (Dlib)
  - EAR / MAR computation
  - Drowsiness & yawn logic
  - Alarm management
  - CSV logging
  - Tkinter GUI

Keyboard shortcuts
------------------
  Q  — Quit
  S  — Save screenshot

Run
---
  python main.py

Requirements
------------
  See requirements.txt and README.md for setup instructions.
  Ensure models/shape_predictor_68_face_landmarks.dat exists.
"""

import sys
import os
import time
import queue
import logging
import threading
from datetime import datetime

import cv2
import numpy as np

# ── Local imports ────────────────────────────────────────────────────────────
import config
from utils.eye_detection  import average_ear, draw_eye_contours, EyeStateTracker
from utils.yawn_detection import mouth_aspect_ratio, draw_mouth_contour, YawnTracker
from utils.alarm          import AlarmManager
from utils.logger         import DetectionLogger, get_logger
from utils.helpers        import (
    FPSCounter, shape_to_np,
    draw_status_panel, draw_ear_bar,
    save_screenshot, is_low_light,
    open_camera, reconnect_camera,
)
from utils.gui import DrowsinessGUI
from utils.voice_alert  import VoiceAlertSystem
from utils.database     import DetectionDatabase, generate_pdf_report
from utils.head_pose    import HeadPoseEstimator, HeadState
from utils.night_vision import NightVisionEnhancer, EnhancementMode

logger = get_logger("main")


# ════════════════════════════════════════════════════════════════════════════
# Detection worker
# ════════════════════════════════════════════════════════════════════════════

class DetectionWorker:
    """
    Runs on a daemon thread: reads frames, runs all detection algorithms,
    and pushes (annotated_frame, stats_dict) into *frame_queue*.

    Parameters
    ----------
    frame_queue  : queue.Queue   Shared queue consumed by the GUI.
    stop_event   : threading.Event  Set externally to stop the loop.
    screenshot_event : threading.Event  Pulsed to trigger a screenshot.
    """

    def __init__(
        self,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        screenshot_event: threading.Event,
    ):
        self._queue            = frame_queue
        self._stop             = stop_event
        self._screenshot       = screenshot_event

        # ── Face detector (OpenCV Haar) + Dlib landmark predictor ─────
        import dlib
        self._dlib = dlib
        # OpenCV Haar cascade works on every dlib build (no dlib detector needed)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            raise RuntimeError("Haar cascade not found. Reinstall opencv-python.")
        logger.info("OpenCV Haar cascade face detector loaded.")

        if not os.path.isfile(config.MODEL_PATH):
            raise FileNotFoundError(
                f"Shape predictor model not found:\n  {config.MODEL_PATH}\n\n"
                "Download it from:\n"
                "  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                "Extract and place inside the  models/  folder."
            )
        self._predictor = dlib.shape_predictor(config.MODEL_PATH)
        logger.info("Dlib shape predictor loaded.")

        # ── Sub-systems ──────────────────────────────────────────────────
        self._alarm         = AlarmManager(config.ALARM_SOUND_PATH)
        self._det_logger    = DetectionLogger(config.LOG_FILE_PATH)
        self._voice         = VoiceAlertSystem(cooldown_s=8.0)
        db_path = os.path.join(os.path.dirname(config.LOG_FILE_PATH), "drowsiness.db")
        self._db            = DetectionDatabase(db_path)
        self._fps_counter   = FPSCounter(window=30)
        self._eye_tracker   = EyeStateTracker(config.EAR_THRESHOLD,
                                               config.EAR_CONSEC_FRAMES)
        self._yawn_tracker  = YawnTracker(config.MAR_THRESHOLD,
                                          config.YAWN_CONSEC_FRAMES)

        # ── Tier 2: Head pose + Night vision ────────────────────────────
        self._head_pose    = HeadPoseEstimator(
            pitch_nod_threshold=config.HEAD_PITCH_THRESHOLD,
            yaw_turn_threshold=config.HEAD_YAW_THRESHOLD,
            consec_frames=config.HEAD_CONSEC_FRAMES,
            draw_axes=True,
        )
        self._night_vision = NightVisionEnhancer(
            mode=EnhancementMode.AUTO,
            low_light_thresh=config.LOW_LIGHT_THRESHOLD,
        )

        # Session totals
        self._drowsy_total  = 0
        self._last_drowsy   = False   # detect transitions for counting

        # ── Camera ───────────────────────────────────────────────────────
        self._cap = open_camera(
            config.CAMERA_INDEX,
            config.FRAME_WIDTH,
            config.FRAME_HEIGHT,
        )

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main detection loop.  Runs until stop_event is set."""
        log_interval = 15  # log every N frames to keep CSV manageable
        frame_idx    = 0

        try:
            while not self._stop.is_set():
                ret, frame = self._cap.read()

                # Camera reconnect on failure
                if not ret or frame is None:
                    logger.warning("Frame grab failed — attempting camera reconnect.")
                    try:
                        self._cap = reconnect_camera(
                            self._cap,
                            config.CAMERA_INDEX,
                            config.FRAME_WIDTH,
                            config.FRAME_HEIGHT,
                        )
                        continue
                    except RuntimeError as exc:
                        logger.error("Camera reconnect failed: %s", exc)
                        break

                frame_idx += 1
                fps = self._fps_counter.tick()

                # ── Per-frame processing ─────────────────────────────────
                annotated, stats = self._process_frame(frame, fps)

                # ── Screenshot trigger ───────────────────────────────────
                if self._screenshot.is_set():
                    save_screenshot(annotated, config.SCREENSHOTS_DIR)
                    self._screenshot.clear()

                # ── Push to GUI (drop oldest if full) ────────────────────
                try:
                    self._queue.put_nowait((annotated, stats))
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._queue.put_nowait((annotated, stats))

                # ── CSV + SQLite logging (throttled) ─────────────────────
                if frame_idx % log_interval == 0:
                    self._det_logger.log_event(
                        ear=stats["ear"],
                        mar=stats["mar"],
                        eyes_closed=stats["eyes_closed"],
                        yawning=stats["yawning"],
                        drowsy=stats["drowsy"],
                        alarm_triggered=stats["alarm_on"],
                        fps=fps,
                    )
                    self._db.log_event(
                        ear=stats["ear"],
                        mar=stats["mar"],
                        eyes_closed=stats["eyes_closed"],
                        yawning=stats["yawning"],
                        drowsy=stats["drowsy"],
                        alarm_triggered=stats["alarm_on"],
                        fps=fps,
                    )

        finally:
            self._cleanup()

    # ── Frame processing ─────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray, fps: float) -> tuple:
        """
        Run the full detection pipeline on a single frame.

        Returns
        -------
        tuple  (annotated_frame, stats_dict)
        """
        frame        = np.ascontiguousarray(frame, dtype=np.uint8)

        # ── Night vision enhancement (Tier 2) ─────────────────────────────
        frame, night_active = self._night_vision.process(frame)
        frame        = np.ascontiguousarray(frame, dtype=np.uint8)
        gray         = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)

        low_light    = is_low_light(frame, config.LOW_LIGHT_THRESHOLD)

        # Detect faces with OpenCV Haar cascade (bypasses broken dlib detector)
        raw_faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Convert OpenCV (x,y,w,h) → dlib rectangles for the landmark predictor
        faces = [self._dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                 for (x, y, w, h) in (raw_faces if len(raw_faces) > 0 else [])]
        face_detected = len(faces) > 0

        ear = 0.0
        mar = 0.0

        if not face_detected:
            # Reset trackers when face lost
            self._eye_tracker.reset()
            self._yawn_tracker.reset()
            drowsy  = False
            yawning = False
            self._voice.speak("no_face")
        else:
            # Use the largest detected face
            face  = max(faces, key=lambda r: (r.right() - r.left()) *
                                              (r.bottom() - r.top()))
            shape = self._predictor(gray, face)
            pts   = shape_to_np(shape)

            # Extract landmark subsets
            left_eye  = pts[config.LEFT_EYE_START  : config.LEFT_EYE_END]
            right_eye = pts[config.RIGHT_EYE_START : config.RIGHT_EYE_END]
            mouth     = pts[config.MOUTH_START     : config.MOUTH_END]

            ear = average_ear(left_eye, right_eye)
            mar = mouth_aspect_ratio(mouth)

            # Tracker updates
            drowsy  = self._eye_tracker.update(ear)
            yawning = self._yawn_tracker.update(mar)

            # Alarm + voice alert control
            if drowsy:
                self._alarm.start()
                self._voice.speak("drowsy")
            elif yawning:
                self._alarm.start()
                self._voice.speak("yawn")
            else:
                self._alarm.stop()

            # Count new drowsiness episodes
            if drowsy and not self._last_drowsy:
                self._drowsy_total += 1
            self._last_drowsy = drowsy

            # ── Annotate eye and mouth contours ─────────────────────────
            eye_color = (config.COLOR_RED if self._eye_tracker.frame_counter > 0
                         else config.COLOR_GREEN)
            draw_eye_contours(frame, left_eye, right_eye, color=eye_color)

            mouth_color = (config.COLOR_YELLOW if yawning else config.COLOR_GREEN)
            draw_mouth_contour(frame, mouth, color=mouth_color)

            # Face bounding box
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            box_color = config.COLOR_RED if drowsy else config.COLOR_GREEN
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        eyes_closed = ear < config.EAR_THRESHOLD and face_detected

        # ── Head pose estimation (Tier 2) ────────────────────────────────
        head_state, head_angles = self._head_pose.process(frame)
        self._head_pose.draw_state(frame, head_state, head_angles)
        if head_state.is_alert() and face_detected:
            self._alarm.start()
            self._voice.speak("distracted")

        # ── Overlay status panel ─────────────────────────────────────────
        draw_status_panel(
            frame,
            ear=ear, mar=mar, fps=fps,
            blink_count=self._eye_tracker.blink_count,
            yawn_count=self._yawn_tracker.yawn_count,
            drowsy=drowsy,
            yawning=yawning,
            alarm_on=self._alarm.is_active,
            face_detected=face_detected,
            low_light=low_light,
        )

        draw_ear_bar(frame, ear, config.EAR_THRESHOLD)

        stats = {
            "ear":          ear,
            "mar":          mar,
            "fps":          fps,
            "face_detected": face_detected,
            "eyes_closed":  eyes_closed,
            "yawning":      yawning,
            "drowsy":       drowsy,
            "alarm_on":     self._alarm.is_active,
            "low_light":    low_light,
            "blink_count":  self._eye_tracker.blink_count,
            "yawn_count":   self._yawn_tracker.yawn_count,
            "drowsy_total": self._drowsy_total,
            "head_state":   head_state.label(),
            "head_alert":   head_state.is_alert(),
            "night_active": night_active,
        }

        return frame, stats

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        """Release all resources safely."""
        logger.info("Releasing resources …")
        self._alarm.shutdown()
        self._voice.shutdown()
        self._head_pose.shutdown()
        # Close DB session and generate PDF report
        try:
            self._db.close_session()
            from datetime import datetime
            report_dir  = os.path.join(os.path.dirname(config.LOG_FILE_PATH))
            report_path = os.path.join(
                report_dir,
                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            result = generate_pdf_report(self._db, report_path)
            if result:
                logger.info("Session PDF report → %s", result)
        except Exception as exc:
            logger.warning("PDF report generation failed: %s", exc)
        if self._cap.isOpened():
            self._cap.release()
        logger.info("Resources released.")


# ════════════════════════════════════════════════════════════════════════════
# Application entry point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Initialise and start the full drowsiness detection system.

    Flow
    ----
    1. Create shared synchronisation primitives.
    2. Start the DetectionWorker on a daemon thread.
    3. Start the Tkinter GUI on the main thread (required by Tk).
    4. On GUI close, signal the worker to stop and wait for it.
    """
    logger.info("=" * 60)
    logger.info("AI Driver Drowsiness Detection System — Starting")
    logger.info("=" * 60)

    # Shared primitives
    frame_queue      = queue.Queue(maxsize=2)
    stop_event       = threading.Event()
    screenshot_event = threading.Event()

    # ── Detection worker setup ───────────────────────────────────────────
    try:
        worker = DetectionWorker(frame_queue, stop_event, screenshot_event)
    except FileNotFoundError as exc:
        logger.critical(str(exc))
        sys.exit(1)
    except RuntimeError as exc:
        logger.critical("Startup failed: %s", exc)
        sys.exit(1)

    worker_thread = threading.Thread(
        target=worker.run,
        daemon=True,
        name="DetectionWorker",
    )

    # ── GUI callbacks ────────────────────────────────────────────────────
    def on_quit():
        logger.info("Quit requested by user.")
        stop_event.set()

    def on_screenshot():
        screenshot_event.set()
        logger.info("Screenshot requested.")

    # ── Build GUI ─────────────────────────────────────────────────────────
    gui = DrowsinessGUI(
        title="AI Driver Drowsiness Detection System",
        frame_queue=frame_queue,
        on_quit=on_quit,
        on_screenshot=on_screenshot,
    )

    # ── Start worker and GUI ──────────────────────────────────────────────
    worker_thread.start()
    logger.info("Detection thread started.")

    gui.start()   # ← blocks until window is closed

    # ── Teardown ──────────────────────────────────────────────────────────
    logger.info("GUI closed — waiting for detection thread to finish …")
    stop_event.set()
    worker_thread.join(timeout=5)
    logger.info("System shutdown complete.")


# ── Headless / OpenCV-only fallback ─────────────────────────────────────────

def main_headless() -> None:
    """
    Headless mode: runs the detection pipeline with OpenCV's own
    imshow() window instead of the Tkinter GUI.

    Useful on systems without a display server or for quick testing.

    Keyboard shortcuts
    ------------------
      Q / ESC  — Quit
      S        — Save screenshot
    """
    logger.info("Running in headless (OpenCV imshow) mode.")

    stop_event       = threading.Event()
    screenshot_event = threading.Event()
    frame_queue      = queue.Queue(maxsize=2)

    try:
        worker = DetectionWorker(frame_queue, stop_event, screenshot_event)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.critical(str(exc))
        sys.exit(1)

    worker_thread = threading.Thread(
        target=worker.run, daemon=True, name="DetectionWorker"
    )
    worker_thread.start()

    cv2.namedWindow("Drowsiness Monitor", cv2.WINDOW_NORMAL)

    try:
        while not stop_event.is_set():
            try:
                frame, _ = frame_queue.get(timeout=0.1)
                cv2.imshow("Drowsiness Monitor", frame)
            except queue.Empty:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                break
            if key in (ord("s"), ord("S")):
                screenshot_event.set()

    finally:
        stop_event.set()
        worker_thread.join(timeout=5)
        cv2.destroyAllWindows()
        logger.info("Headless mode shutdown complete.")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Driver Drowsiness Detection System"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run with OpenCV imshow window instead of the Tkinter GUI.",
    )
    args = parser.parse_args()

    if args.headless:
        main_headless()
    else:
        main()
