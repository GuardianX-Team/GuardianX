import cv2
import requests
import time
import tempfile

# -----------------------------
# Backend URL
backend_url = "http://127.0.0.1:8000/object-detect/"

# Camera URL (your phone IP webcam)
camera_url = "http://100.77.252.170:8080/video"  # live video stream
# -----------------------------

# Open video stream
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("Error: Cannot open camera stream. Check the IP and network.")
    exit()

print("Starting live video detection... Press Ctrl+C to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        time.sleep(1)
        continue

    # Save frame to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(temp_file.name, frame)

    # Send frame to backend for detection
    with open(temp_file.name, "rb") as f:
        files = {"file": ("frame.jpg", f, "image/jpeg")}
        try:
            resp = requests.post(backend_url, files=files)
            print(resp.json())  # prints detection results
        except Exception as e:
            print("Error sending frame:", e)

    time.sleep(0.5)  # 2 frames per second
