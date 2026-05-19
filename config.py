"""
config.py
=========
Central configuration file for the AI-Based Driver Drowsiness Detection System.

All tunable parameters are defined here so the system can be adjusted
without touching core logic files.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dlib 68-point landmark model (download separately — see README)
MODEL_PATH = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")

# Alarm audio file
ALARM_SOUND_PATH = os.path.join(BASE_DIR, "alarm", "alarm.wav")

# CSV detection log
LOG_FILE_PATH = os.path.join(BASE_DIR, "logs", "detection_logs.csv")

# Screenshots output folder
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

# ─────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────
CAMERA_INDEX = 0          # 0 = default webcam; change for external cameras
FRAME_WIDTH  = 640        # Capture width  (pixels)
FRAME_HEIGHT = 480        # Capture height (pixels)

# ─────────────────────────────────────────────
# EYE ASPECT RATIO (EAR) — drowsiness detection
# ─────────────────────────────────────────────
EAR_THRESHOLD    = 0.25   # EAR below this value → eyes considered closed
EAR_CONSEC_FRAMES = 20    # Number of consecutive frames below threshold → alarm

# ─────────────────────────────────────────────
# MOUTH ASPECT RATIO (MAR) — yawn detection
# ─────────────────────────────────────────────
MAR_THRESHOLD      = 0.45  # MAR above this value → yawn detected
YAWN_CONSEC_FRAMES = 15    # Consecutive frames above threshold → confirmed yawn

# ─────────────────────────────────────────────
# LOW-LIGHT WARNING
# ─────────────────────────────────────────────
LOW_LIGHT_THRESHOLD = 50   # Mean pixel brightness below this → warn user

# ─────────────────────────────────────────────
# DISPLAY / GUI
# ─────────────────────────────────────────────
FONT_SCALE    = 0.6
FONT_THICKNESS = 2

# Colours used in the OpenCV overlay (BGR)
COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_CYAN   = (255, 255, 0)

# ─────────────────────────────────────────────
# DLIB LANDMARK INDICES
# ─────────────────────────────────────────────
# Right eye: landmarks 36–41  (0-indexed)
RIGHT_EYE_START = 36
RIGHT_EYE_END   = 42

# Left eye: landmarks 42–47
LEFT_EYE_START  = 42
LEFT_EYE_END    = 48

# Mouth (outer): landmarks 48–67
MOUTH_START = 48
MOUTH_END   = 68

# -----------------------------------------
# HEAD POSE DETECTION  (Tier 2)
# -----------------------------------------
HEAD_PITCH_THRESHOLD  = 20   # Degrees downward before NODDING fires
HEAD_YAW_THRESHOLD    = 25   # Degrees left/right before LOOKING_AWAY fires
HEAD_CONSEC_FRAMES    = 10   # Consecutive frames to confirm a head state

# -----------------------------------------
# NIGHT VISION  (Tier 2)
# -----------------------------------------
LOW_LIGHT_THRESHOLD = 60     # Mean brightness below this triggers enhancement
