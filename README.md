# 🚗 AI-Based Driver Drowsiness Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)](https://opencv.org)
[![Dlib](https://img.shields.io/badge/Dlib-19.24%2B-orange)](http://dlib.net)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

A **production-level**, real-time computer-vision system that monitors a
driver's face via webcam, detects drowsiness and yawning using facial
landmark geometry, and triggers an audio alarm before an accident can occur.

---

## 📌 Project Overview

Drowsy driving causes thousands of fatalities every year.  This system
continuously analyses a driver's:

| Signal | How | Threshold |
|--------|-----|-----------|
| Eye closure | Eye Aspect Ratio (EAR) | EAR < 0.25 for ≥ 20 frames |
| Yawning | Mouth Aspect Ratio (MAR) | MAR > 0.75 for ≥ 15 frames |
| Blink rate | EAR transitions | Counted per session |

When drowsiness is detected the system:
- Plays a loud audio alarm (Pygame)
- Displays a red banner on screen
- Logs the event to a CSV file

---

## ✨ Features

- **Real-time 30+ FPS** webcam processing
- **68-point facial landmark** detection (Dlib)
- **EAR** — eye-closure duration measurement
- **MAR** — yawn / mouth-opening measurement
- **Multi-threaded alarm** — never blocks the video pipeline
- **Tkinter GUI** with live stats panel (EAR, MAR, FPS, blink/yawn counts)
- **Headless mode** — `--headless` flag for servers / CI
- **CSV detection logs** with timestamps
- **Auto camera reconnect** on cable/USB disconnect
- **Low-light warning** overlay
- **Screenshot** on keypress (`S`)
- **Keyboard shortcuts**: `Q` = quit, `S` = screenshot
- **Synthetic alarm WAV** auto-generated if `alarm.wav` is absent
- Full **PEP 8** compliance, docstrings, type hints

---

## 🛠 Technologies Used

| Library | Purpose |
|---------|---------|
| **OpenCV** | Camera capture, frame processing, drawing |
| **Dlib** | HOG face detector + 68-point shape predictor |
| **NumPy** | Numerical operations on landmark arrays |
| **SciPy** | Euclidean distances in EAR / MAR formulas |
| **Pygame** | Audio alarm playback |
| **Imutils** | Frame-resize helpers |
| **Pillow** | BGR → RGB conversion for Tkinter canvas |
| **Tkinter** | Cross-platform GUI framework |

---

## 📁 Folder Structure

```
driver_drowsiness_system/
│
├── main.py                        ← Entry point
├── config.py                      ← All configurable parameters
├── requirements.txt
├── README.md
│
├── models/
│   └── shape_predictor_68_face_landmarks.dat   ← Download separately
│
├── alarm/
│   └── alarm.wav                  ← Auto-generated if missing
│
├── logs/
│   └── detection_logs.csv         ← Auto-created at runtime
│
├── utils/
│   ├── __init__.py
│   ├── eye_detection.py           ← EAR calculation & EyeStateTracker
│   ├── yawn_detection.py          ← MAR calculation & YawnTracker
│   ├── alarm.py                   ← Thread-safe AlarmManager
│   ├── logger.py                  ← CSV + Python logging
│   ├── gui.py                     ← Tkinter GUI (DrowsinessGUI)
│   └── helpers.py                 ← FPS, annotations, screenshot, camera
│
└── screenshots/
    └── *.png                      ← Saved via S key
```

---

## ⚙️ Installation

### 1. Clone / download the project

```bash
git clone https://github.com/yourname/driver-drowsiness-system.git
cd driver_drowsiness_system
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Dlib** requires a C++ compiler and CMake on some systems:
>
> **Windows**
> ```bash
> # Install Visual Studio Build Tools, then:
> pip install cmake
> pip install dlib
> ```
>
> **Ubuntu / Debian**
> ```bash
> sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev
> pip install dlib
> ```
>
> **macOS**
> ```bash
> brew install cmake
> pip install dlib
> ```

### 4. Download the landmark model

The 68-point shape predictor is **not bundled** due to its size (99 MB).

```bash
# Download from the official Dlib mirror:
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Extract:
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

# Move to the models/ folder:
mv shape_predictor_68_face_landmarks.dat models/
```

On Windows, use 7-Zip or WinRAR to extract the `.bz2` file, then copy
the `.dat` file into `models\`.

---

## ▶️ How to Run

### Default — Tkinter GUI mode

```bash
python main.py
```

### Headless — OpenCV imshow window (no Tkinter required)

```bash
python main.py --headless
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` | Quit the application |
| `S` | Save a screenshot to `screenshots/` |

---

## 🔧 Configuration

Edit `config.py` to tune the system:

```python
EAR_THRESHOLD    = 0.25   # lower → harder to trigger drowsiness
EAR_CONSEC_FRAMES = 20   # more frames → longer closed-eye period required
MAR_THRESHOLD    = 0.75   # higher → wider yawn required
YAWN_CONSEC_FRAMES = 15
CAMERA_INDEX     = 0      # 0 = built-in webcam; 1 = external USB
LOW_LIGHT_THRESHOLD = 50  # mean brightness below this → warning
```

---

## 📊 Detection Logs

All detection events are appended to `logs/detection_logs.csv`:

```
timestamp,ear,mar,eyes_closed,yawning,drowsy,alarm_triggered,fps
2024-03-15T08:42:01.123,0.2831,0.1204,False,False,False,False,29.8
2024-03-15T08:42:05.887,0.1742,0.1104,True,False,True,True,28.4
```

---

## 🖼 Screenshots

Place example output screenshots in `screenshots/` after capturing them
with `S` during a session.

---

## 🔬 Algorithm Details

### Eye Aspect Ratio (EAR)

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```

The six landmarks (p1–p6) are the standard Dlib eye points (36–41 for
the right eye, 42–47 for the left).  EAR ≈ 0.30 when fully open and
drops near 0.0 when closed.

### Mouth Aspect Ratio (MAR)

```
MAR = (||top-bot|| + ||tl-bl|| + ||tr-br||) / (3 × ||left-right||)
```

Uses eight outer-lip landmarks from the 68-point model.  MAR > 0.75
sustained for ≥ 15 frames indicates a yawn.

---

## 🚀 Future Improvements

- Head-pose estimation for nodding detection
- Pupil tracking for micro-sleep detection
- Mobile deployment (TFLite / ONNX export)
- SQLite back-end for richer log querying
- Dashboard web-app (Streamlit / Dash)
- Driver-profile calibration (per-person EAR baseline)
- Night-vision infrared camera support

---

## 🛟 Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: shape_predictor…` | Follow Step 4 in Installation |
| `Cannot open camera at index 0` | Check webcam connection; try `CAMERA_INDEX = 1` |
| `dlib not found` | Ensure CMake is installed; reinstall dlib |
| No sound / alarm | Ensure `pygame` is installed; check system audio |
| Tkinter ImportError | On Ubuntu: `sudo apt-get install python3-tk` |
| Low FPS | Reduce `FRAME_WIDTH` / `FRAME_HEIGHT` in `config.py` |
| False positives | Increase `EAR_CONSEC_FRAMES` or lower `EAR_THRESHOLD` |

---

## 📜 License

MIT License — see `LICENSE` for details.

---

## 👤 Author

Built with ❤️ as a production-level AI safety project.
