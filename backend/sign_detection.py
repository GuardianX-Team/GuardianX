import cv2
import numpy as np
import tensorflow as tf
import os

# Load trained model
model = tf.keras.models.load_model("backend/models/sign_model.h5")

# Labels (must match your dataset folder names)
LABELS = sorted(os.listdir("dataset/train"))  

IMG_SIZE = 64  # must match training size

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def run_sign_detection(source=0):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("❌ Error: Could not open camera/video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[100:400, 100:400]  # Crop region for hand
        processed = preprocess_frame(roi)

        # Prediction
        preds = model.predict(processed, verbose=0)
        class_id = np.argmax(preds)
        confidence = np.max(preds)

        label = f"{LABELS[class_id]} ({confidence*100:.1f}%)"

        # Draw UI
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        cv2.putText(frame, label, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("🤟 GuardianX - Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
