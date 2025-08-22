import cv2
import torch
from gtts import gTTS
import os
import playsound

# Load YOLOv5 model (first time will download from torch hub)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # small model for speed

# Function for voice alerts
def send_voice_alert(message="Alert! Object detected!"):
    try:
        tts = gTTS(message)
        filename = "alert.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)
    except Exception as e:
        print("Voice alert error:", e)

# Main object detection loop
def run_object_detection():
    cap = cv2.VideoCapture(0)  # 0 = webcam (change if needed)

    if not cap.isOpened():
        print("❌ Cannot access camera")
        return

    print("✅ Object detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model
        results = model(frame)

        # Extract labels
        labels = results.names
        detections = results.xyxy[0]  # (x1, y1, x2, y2, confidence, class)

        # Draw detections
        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            label = labels[int(cls)]
            confidence = float(conf)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

            # Voice alert for specific objects
            if label in ["person", "knife", "scissors"]:  
                send_voice_alert(f"Warning! {label} detected")

        # Show output
        cv2.imshow("GuardianX - Object Detection", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
