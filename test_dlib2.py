"""
test_dlib2.py — Extended diagnostic with dlib.cv_image()
"""
import cv2
import numpy as np
import dlib

print("dlib version:", dlib.__version__)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

detector = dlib.get_frontal_face_detector()

print("\nTrying dlib.cv_image(frame)  [BGR]...")
try:
    img = dlib.cv_image(frame)
    faces = detector(img, 0)
    print(f"  ✅ WORKS — faces: {len(faces)}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print("\nTrying dlib.cv_image(rgb)  [RGB]...")
try:
    img = dlib.cv_image(rgb)
    faces = detector(img, 0)
    print(f"  ✅ WORKS — faces: {len(faces)}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

print("\nTrying numpy array via np.array(frame, dtype=np.uint8)...")
try:
    img = np.array(frame, dtype=np.uint8, order='C')
    gray = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), dtype=np.uint8, order='C')
    faces = detector(gray, 0)
    print(f"  ✅ WORKS — faces: {len(faces)}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

print("\nDone.")
