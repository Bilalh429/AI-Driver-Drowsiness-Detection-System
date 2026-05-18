"""
test_dlib.py — Diagnostic script
Run this to find the correct image format for your dlib build.
"""
import cv2
import numpy as np
import dlib

print("=" * 50)
print("dlib version:", dlib.__version__)

# Grab one frame from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("ERROR: Could not read from webcam!")
    exit()

print("Frame shape :", frame.shape)
print("Frame dtype :", frame.dtype)
print("Contiguous  :", frame.flags['C_CONTIGUOUS'])
print("=" * 50)

detector = dlib.get_frontal_face_detector()

tests = {
    "BGR (raw)"  : frame,
    "Gray"       : cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
    "RGB"        : cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
}

for name, img in tests.items():
    img_c = np.ascontiguousarray(img, dtype=np.uint8)
    print(f"\nTrying {name} — shape={img_c.shape} dtype={img_c.dtype}")
    try:
        faces = detector(img_c, 0)
        print(f"  ✅ WORKS!  faces detected: {len(faces)}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")

print("\n" + "=" * 50)
print("Done. Share the output above.")
