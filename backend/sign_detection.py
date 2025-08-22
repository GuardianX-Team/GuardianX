# backend/sign_detection.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

IMG_SIZE = (64,64)
dataset_path = r"C:\Users\totek\Documents\GitHub\GuardianX\dataset\sign_language"
model_path = r"C:\Users\totek\Documents\GitHub\GuardianX\backend\sign_model.h5"

# --------- Load model ---------
classes = sorted(os.listdir(dataset_path))
model = load_model(model_path)
print("Model loaded successfully!")
print("Detected gesture classes:", classes)

# --------- Start camera ---------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, IMG_SIZE)
    img = img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # Predict gesture
    pred = model.predict(img)
    gesture = classes[np.argmax(pred)]

    # Display result
    cv2.putText(frame, f"Gesture: {gesture}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
