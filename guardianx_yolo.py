import cv2
from ultralytics import YOLO
import threading
import pyttsx3
import numpy as np

# -------------------- CONFIG --------------------
RTSP_URL = "rtsp://192.168.1.5:8080/h264.sdp"
YOLO_MODEL = "yolov8n-seg.pt"  # segmentation model

# Distance threshold (in pixels) for alerts
ALERT_ZONE = 200  # adjust based on camera view

# Initialize YOLO model
model = YOLO(YOLO_MODEL)

# Initialize TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Thread-safe alert function
def speak_alert(text):
    threading.Thread(target=lambda: tts_engine.say(text) or tts_engine.runAndWait()).start()

# -------------------- VIDEO STREAM --------------------
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("❌ Unable to open RTSP stream.")
    exit()

print("✅ RTSP stream opened. Starting YOLO obstacle detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame. Retrying...")
        continue

    # Run YOLO segmentation
    results = model(frame)

    detected_classes = []

    for result in results:
        # Draw segmentation masks
        if hasattr(result, 'masks') and result.masks is not None:
            for mask in result.masks.xy:  # polygon points
                pts = np.array(mask, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cv2.fillPoly(frame, [pts], (0, 255, 0, 50))

        # Draw bounding boxes and labels
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            detected_classes.append((model.names[int(cls)], x1, y1, x2, y2))

    # Trigger voice alerts for objects **close to the camera**
    for cls, x1, y1, x2, y2 in detected_classes:
        # simple proximity check based on object size / position
        object_height = y2 - y1
        if object_height > ALERT_ZONE:  # adjust threshold based on camera
            speak_alert(f"{cls} ahead!")

    # Display smart video feed
    cv2.imshow("GuardianX - YOLO Obstacle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
