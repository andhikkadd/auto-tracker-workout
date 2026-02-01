import cv2
import time

from state import STATE
from tracker import detect_pose

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

    frame = detect_pose(frame)

    cv2.imshow("Detect",frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
