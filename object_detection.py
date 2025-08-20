import cv2
from ultralytics import YOLO
from voice_alert import send_voice_alert
from object_alerts import OBJECT_ALERTS

def start_object_detection():
    model = YOLO("yolov8n.pt")  # YOLOv8 nano model (fastest)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                if conf > 0.5:  # Only confident detections
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Trigger your gTTS voice alert
                    if label in OBJECT_ALERTS:
                        send_voice_alert(OBJECT_ALERTS[label])
                    else:
                        send_voice_alert("Obstacle detected!")

        cv2.imshow("GuardianX - Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
