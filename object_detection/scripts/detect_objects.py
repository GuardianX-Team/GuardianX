import cv2
import numpy as np

RTSP_URL = "rtsp://192.168.1.5:8080/h264.sdp"
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ Error: Could not open RTSP stream.")
    exit()

# Sharpening kernel
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

print("✅ Stream started... press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        continue

    # Apply sharpening
    sharpened = cv2.filter2D(frame, -1, sharpen_kernel)

    cv2.imshow("RTSP Stream - Sharpened", sharpened)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
