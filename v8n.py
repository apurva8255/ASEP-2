import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import time

# === Load YOLOv8 Nano model (optimized for speed) ===
model = YOLO("yolov8s.pt")

# === Camera resolution ===
WIDTH, HEIGHT = 640, 480

# === Launch libcamera-vid process ===
command = [
    'libcamera-vid',
    '--inline',
    '-t', '0',
    '--width', str(WIDTH),
    '--height', str(HEIGHT),
    '--codec', 'yuv420',
    '--nopreview',
    '--output', '-',
    '--framerate', '30'
]

print("📷 Starting libcamera-vid stream in headless mode...")
proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

frame_size = int(WIDTH * HEIGHT * 1.5)  # YUV420 frame size
frame_count = 0
saved_count = 0

try:
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) != frame_size:
            print("⚠️ Incomplete frame received.")
            break

        # Convert YUV420 to BGR for OpenCV + YOLO
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((int(HEIGHT * 1.5), WIDTH))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        # Run YOLO detection
        results = model(bgr, verbose=False)[0]
        detections = 0

        for box in results.boxes.xywh.cpu().numpy():
            x, y, w, h = map(int, box)
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            roi = bgr[max(0, y1):min(bgr.shape[0], y2), max(0, x1):min(bgr.shape[1], x2)]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)

            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                param1=100, param2=15, minRadius=5, maxRadius=int(min(w, h) / 2)
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :1]:  # only strongest
                    cx, cy, r = circle
                    cv2.circle(bgr, (x1 + cx, y1 + cy), r, (0, 255, 255), 2)
                    cv2.putText(bgr, "Confirmed Sphere", (x1 + cx - 30, y1 + cy - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    # print("🎯 Hough-confirmed circle at:", (x1 + cx, y1 + cy))
                    detections += 1

        frame_count += 1
        # print(f"📸 Frame {frame_count} processed. Ball detected: {detections}")
        print(f"📸 Frame {frame_count} processed. Ball detected")

        # Save the frame if any detection occurred
        if detections > 0:
            filename = f"detection_{saved_count:03}.jpg"
            cv2.imwrite(filename, bgr)
            print(f"💾 Saved: {filename}")
            saved_count += 1

        time.sleep(0.01)  # Small pause to prevent CPU overload

except KeyboardInterrupt:
    print("🛑 Stopped by user.")

finally:
    proc.kill()
    print("🔚 Camera stream closed.")