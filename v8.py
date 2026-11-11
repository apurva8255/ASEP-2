import cv2
import numpy as np
from ultralytics import YOLO
import subprocess
import time
from collections import deque
import matplotlib.pyplot as plt
import threading

# === Load YOLOv8 model ===
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

# === For tracking and plotting trajectory ===
positions = deque(maxlen=100)  # (x, y, timestamp)

def plot_trajectory():
    """ Continuously plot the ball trajectory in a separate window. """
    plt.ion()
    fig, ax = plt.subplots()
    while True:
        if positions:
            xs, ys = zip(*[(pos[0], pos[1]) for pos in positions])
            ax.clear()
            ax.plot(xs, ys, 'go-', linewidth=2, markersize=5)
            ax.set_title("Golf Ball Trajectory")
            ax.set_xlim(0, WIDTH)
            ax.set_ylim(HEIGHT, 0)  # Invert Y axis (top-left is 0,0 in images)
            plt.pause(0.1)

# Start plotting in background
plot_thread = threading.Thread(target=plot_trajectory, daemon=True)
plot_thread.start()

try:
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) != frame_size:
            print("⚠️ Incomplete frame received.")
            break

        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((int(HEIGHT * 1.5), WIDTH))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        results = model(bgr, verbose=False)[0]
        detections = 0
        ball_center = None

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
                for circle in circles[0, :1]:  # strongest only
                    cx, cy, r = circle
                    abs_cx, abs_cy = x1 + cx, y1 + cy
                    ball_center = (abs_cx, abs_cy)
                    cv2.circle(bgr, (abs_cx, abs_cy), r, (0, 255, 255), 2)
                    cv2.putText(bgr, "Confirmed Sphere", (abs_cx - 30, abs_cy - r - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    detections += 1

        # Store timestamped position if ball detected
        if ball_center:
            timestamp = time.time()
            positions.append((ball_center[0], ball_center[1], timestamp))

        frame_count += 1
        print(f"📸 Frame {frame_count} processed. Ball detected")

        if detections > 0:
            filename = f"detection_{saved_count:03}.jpg"
            cv2.imwrite(filename, bgr)
            print(f"💾 Saved: {filename}")
            saved_count += 1

        time.sleep(0.01)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")

finally:
    proc.kill()
    print("🔚 Camera stream closed.")