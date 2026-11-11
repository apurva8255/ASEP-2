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
    '--framerate', '120'
]

print("📷 Starting libcamera-vid stream in headless mode...")
proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize = 10**8>

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