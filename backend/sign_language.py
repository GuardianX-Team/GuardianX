# type: ignore
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque, Counter
import time
import winsound  # for simple beep sound (Windows)

# Load trained model
model = load_model('sign_language_model.h5')

# Mini dataset gestures
GESTURES = ['1', '2', '3', 'A', 'B', 'C']

# Camera capture (default 0 for webcam)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# Prediction buffer and word tracking
buffer = deque(maxlen=10)
current_word = ""
last_letter = ""
no_detection_frames = 0
NO_DETECTION_THRESHOLD = 15

# Animation timing
last_update_time = time.time()
ANIMATION_DELAY = 0.1  # seconds between updates

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect

    # ROI
    x0, y0, width, height = 100, 100, 300, 300
    roi = frame[y0:y0+height, x0:x0+width]

    # Preprocess
    roi_resized = cv2.resize(roi, (64, 64))
    roi_array = img_to_array(roi_resized)
    roi_array = np.expand_dims(roi_array, axis=0) / 255.0

    # Predict
    preds = model.predict(roi_array, verbose=0)
    gesture_idx = np.argmax(preds[0])
    confidence = preds[0][gesture_idx]
    gesture_name = GESTURES[gesture_idx] if confidence > 0.3 else None  # lower threshold for small dataset

    # Update buffer
    if gesture_name:
        buffer.append(gesture_name)
        no_detection_frames = 0
    else:
        no_detection_frames += 1

    # Determine most common letter
    most_common = Counter(buffer).most_common(1)[0][0] if buffer else None

    # Add to current word if changed and delay passed
    if most_common and most_common != last_letter and time.time() - last_update_time > ANIMATION_DELAY:
        current_word += most_common
        last_letter = most_common
        last_update_time = time.time()
        winsound.Beep(1000, 100)  # optional sound feedback

    # Add space if no detection
    if no_detection_frames >= NO_DETECTION_THRESHOLD:
        current_word += " "
        last_letter = ""
        buffer.clear()
        no_detection_frames = 0

    # Draw ROI
    cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0, 255, 0), 2)

    # Overlays
    overlay = frame.copy()
    # Letter panel
    cv2.rectangle(overlay, (x0, y0-50), (x0+width, y0), (0, 0, 0), -1)
    # Word panel
    cv2.rectangle(overlay, (50, 20), (600, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Letter color based on confidence
    if confidence > 0.8:
        color = (0, 255, 0)
    elif confidence > 0.5:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    # Current letter
    if most_common:
        letter_text = f"{most_common} ({confidence*100:.1f}%)"
        cv2.putText(frame, letter_text, (x0 + 10, y0 - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3, cv2.LINE_AA)

    # Current word
    cv2.putText(frame, f"Word: {current_word}", (60, 65),
                cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 0), 4, cv2.LINE_AA)

    # Show frame
    cv2.imshow("Sign Language Detection", frame)

    # Key actions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        current_word = ""  # Reset word

cap.release()
cv2.destroyAllWindows()
