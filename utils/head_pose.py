"""
utils/head_pose.py
==================
Real-time head pose estimation using MediaPipe Face Mesh.

Detects three driver distraction states:
  - FORWARD      -- driver looking straight ahead (safe)
  - NODDING      -- head tilted downward (falling asleep)
  - LOOKING_AWAY -- head turned left/right (distracted)

Algorithm
---------
MediaPipe supplies 478 3-D face landmarks.  We pick 6 canonical points
that correspond to a standard 3-D face model, then use OpenCV's
solvePnP() to estimate the rotation vector.  Rodrigues formula converts
that to Euler angles (pitch / yaw / roll).

Euler angles
------------
  Pitch : positive = head down (nodding),  negative = head up
  Yaw   : positive = head right,           negative = head left
  Roll  : positive = tilting right,        negative = tilting left
"""

import cv2
import numpy as np
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class HeadState(Enum):
    FORWARD       = auto()
    NODDING       = auto()
    LOOKING_LEFT  = auto()
    LOOKING_RIGHT = auto()
    HEAD_UP       = auto()
    UNKNOWN       = auto()

    def label(self):
        return {
            HeadState.FORWARD:       "Forward",
            HeadState.NODDING:       "NODDING",
            HeadState.LOOKING_LEFT:  "Looking Left",
            HeadState.LOOKING_RIGHT: "Looking Right",
            HeadState.HEAD_UP:       "Head Up",
            HeadState.UNKNOWN:       "Unknown",
        }[self]

    def is_alert(self):
        return self in (HeadState.NODDING,
                        HeadState.LOOKING_LEFT,
                        HeadState.LOOKING_RIGHT)

    def color(self):
        if self == HeadState.FORWARD:
            return (0, 255, 0)
        if self == HeadState.UNKNOWN:
            return (150, 150, 150)
        return (0, 100, 255)


# 3-D reference face model (6 canonical points, mm, nose-tip origin)
_MODEL_3D = np.array([
    [ 0.0,    0.0,    0.0  ],   # Nose tip          MP idx 1
    [ 0.0,  -63.6,  -12.5 ],   # Chin              MP idx 152
    [-43.3,  32.7,  -26.0 ],   # Left eye corner   MP idx 263
    [ 43.3,  32.7,  -26.0 ],   # Right eye corner  MP idx 33
    [-28.9, -28.9,  -24.1 ],   # Left mouth corner MP idx 287
    [ 28.9, -28.9,  -24.1 ],   # Right mouth corner MP idx 57
], dtype=np.float64)

_MP_INDICES = [1, 152, 263, 33, 287, 57]


class HeadPoseEstimator:
    """
    Estimates driver head pose from a BGR frame using MediaPipe.

    Parameters
    ----------
    pitch_nod_threshold : float  Pitch degrees beyond which -> NODDING (default 20).
    yaw_turn_threshold  : float  Yaw   degrees beyond which -> LOOKING_AWAY (default 25).
    consec_frames       : int    Consecutive alert frames before state fires (default 10).
    draw_axes           : bool   Draw 3-D pose axes on frame (default True).
    """

    def __init__(self,
                 pitch_nod_threshold=20.0,
                 yaw_turn_threshold=25.0,
                 consec_frames=10,
                 draw_axes=True):
        self._pitch_thresh = pitch_nod_threshold
        self._yaw_thresh   = yaw_turn_threshold
        self._consec       = consec_frames
        self._draw_axes    = draw_axes
        self._frame_counter = {}
        self._current_state = HeadState.UNKNOWN
        self._face_mesh     = None
        self._available     = False
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._available = True
            logger.info("MediaPipe Face Mesh loaded for head pose.")
        except ImportError:
            logger.warning("mediapipe not installed -- head pose disabled. "
                           "Run: pip install mediapipe")
        except Exception as exc:
            logger.error("MediaPipe init error: %s", exc)

    def _get_2d_points(self, results, h, w):
        if not results.multi_face_landmarks:
            return None
        lms = results.multi_face_landmarks[0].landmark
        return np.array(
            [[lms[i].x * w, lms[i].y * h] for i in _MP_INDICES],
            dtype=np.float64,
        )

    def _camera_matrix(self, h, w):
        f = float(w)
        return np.array([[f, 0, w/2],
                         [0, f, h/2],
                         [0, 0,   1]], dtype=np.float64)

    def _euler_angles(self, rvec):
        mat, _ = cv2.Rodrigues(rvec)
        pitch = np.degrees(np.arctan2(mat[2][1], mat[2][2]))
        yaw   = np.degrees(np.arctan2(-mat[2][0],
                           np.sqrt(mat[2][1]**2 + mat[2][2]**2)))
        roll  = np.degrees(np.arctan2(mat[1][0], mat[0][0]))
        return pitch, yaw, roll

    def _classify(self, pitch, yaw):
        if pitch > self._pitch_thresh:
            return HeadState.NODDING
        if yaw < -self._yaw_thresh:
            return HeadState.LOOKING_LEFT
        if yaw > self._yaw_thresh:
            return HeadState.LOOKING_RIGHT
        if pitch < -15:
            return HeadState.HEAD_UP
        return HeadState.FORWARD

    def _smooth(self, raw):
        self._frame_counter[raw] = self._frame_counter.get(raw, 0) + 1
        for s in HeadState:
            if s != raw:
                self._frame_counter[s] = 0
        if self._frame_counter[raw] >= self._consec:
            self._current_state = raw
        return self._current_state

    def _draw_axes(self, frame, rvec, tvec, cam_mat):
        dist = np.zeros((4, 1), dtype=np.float64)
        axis = np.float64([[50,0,0],[0,50,0],[0,0,50],[0,0,0]])
        pts, _ = cv2.projectPoints(axis, rvec, tvec, cam_mat, dist)
        pts = pts.astype(int)
        o = tuple(pts[3].ravel())
        cv2.line(frame, o, tuple(pts[0].ravel()), (0,   0, 255), 2)  # X red
        cv2.line(frame, o, tuple(pts[1].ravel()), (0, 255,   0), 2)  # Y green
        cv2.line(frame, o, tuple(pts[2].ravel()), (255, 0,   0), 2)  # Z blue

    @property
    def available(self):
        return self._available

    def process(self, frame):
        """
        Estimate head pose from BGR frame (annotated in place).

        Returns
        -------
        tuple (HeadState, (pitch, yaw, roll) | None)
        """
        if not self._available:
            return HeadState.UNKNOWN, None

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._face_mesh.process(rgb)
        rgb.flags.writeable = True

        pts_2d = self._get_2d_points(results, h, w)
        if pts_2d is None:
            return HeadState.UNKNOWN, None

        cam   = self._camera_matrix(h, w)
        dist  = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_3D, pts_2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return HeadState.UNKNOWN, None

        pitch, yaw, roll = self._euler_angles(rvec)
        state = self._smooth(self._classify(pitch, yaw))

        if self._draw_axes:
            self._draw_axes(frame, rvec, tvec, cam)

        return state, (pitch, yaw, roll)

    def draw_state(self, frame, state, angles, x=290, y=60):
        """Render head-pose label and angles on frame."""
        font  = cv2.FONT_HERSHEY_SIMPLEX
        label = f"Head: {state.label()}"
        color = state.color()
        cv2.putText(frame, label, (x+1, y+1), font, 0.55, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x,   y  ), font, 0.55, color,   2, cv2.LINE_AA)
        if angles:
            p, y_, r = angles
            txt = f"P:{p:+.0f} Y:{y_:+.0f} R:{r:+.0f}"
            cv2.putText(frame, txt, (x+1, y+21), font, 0.40, (0,0,0),       1, cv2.LINE_AA)
            cv2.putText(frame, txt, (x,   y+20), font, 0.40, (180,180,180), 1, cv2.LINE_AA)

    def shutdown(self):
        if self._face_mesh:
            try:
                self._face_mesh.close()
            except Exception:
                pass
