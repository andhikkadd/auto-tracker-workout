import cv2
import time

from core.config import STATE
from core.tracker import detect_pose

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("tidak dpt membuka kamera")
    exit()

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("gagal menangkap frame")
        break

    now = time.time()
    dt = now - STATE["prev_t"]
    STATE["prev_t"] = now
    if dt > 0:
        inst_fps = 1.0 / dt
        STATE["fps"] = STATE["fps"] * 0.9 + inst_fps * 0.1

    frame = detect_pose(frame, None)
    cv2.imshow("Tracker Workout",frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        frame = detect_pose(frame, key)
        cv2.imshow("", frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()