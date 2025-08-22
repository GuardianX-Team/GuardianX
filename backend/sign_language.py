# backend/sign_language.py
import os
import argparse
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ----------------------------
# Config
# ----------------------------
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")    # backend/dataset/
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")     # backend/models/
MODEL_PATH  = os.path.join(MODELS_DIR, "sign_model.h5")
LABELS_PATH = os.path.join(MODELS_DIR, "sign_labels.txt")
MAP_PATH    = os.path.join(MODELS_DIR, "label_map.json")            # optional

IMG_SIZE    = (64, 64)
BATCH_SIZE  = 32
EPOCHS      = 12  # bump if you have time / data

# ----------------------------
# Utils
# ----------------------------
def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)

def load_label_map_if_any(labels):
    """
    Optional: if backend/models/label_map.json exists (e.g. {"0":"A","1":"B", ...}),
    we map numeric folder names to human-friendly names.
    """
    if os.path.isfile(MAP_PATH):
        try:
            with open(MAP_PATH, "r") as f:
                m = json.load(f)
            return [m.get(lbl, lbl) for lbl in labels]
        except Exception:
            pass
    return labels

# ----------------------------
# Data
# ----------------------------
def make_datasets():
    """
    Expects backend/dataset/<class_folder> with images inside.
    Class folder names can be '0','1',... or words like 'Hello'.
    """
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset not found at {DATASET_DIR}")

    train_ds = image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="training",
        seed=1337,
        shuffle=True,
    )
    val_ds = image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        shuffle=True,
    )

    # cache + prefetch for speed
    AUTOTUNE = tf.data.AUTOTUNE
    norm = tf.keras.Sequential([layers.Rescaling(1./255)])
    train_ds = train_ds.map(lambda x, y: (norm(x), y)).cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (norm(x), y)).cache().prefetch(AUTOTUNE)

    # class names derived from folder names (strings)
    class_names = list(train_ds.class_names)  # e.g. ["0","1","2"] or ["Hello","Yes",...]
    class_names = load_label_map_if_any(class_names)

    return train_ds, val_ds, class_names

# ----------------------------
# Model
# ----------------------------
def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ----------------------------
# Train
# ----------------------------
def train():
    ensure_dirs()
    train_ds, val_ds, class_names = make_datasets()
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(num_classes=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        for c in class_names:
            f.write(c + "\n")

    print(f"✅ Saved model -> {MODEL_PATH}")
    print(f"✅ Saved labels -> {LABELS_PATH}")

# ----------------------------
# Predict helpers
# ----------------------------
def load_model_and_labels():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train first with: python backend/sign_language.py --train")

    model = tf.keras.models.load_model(MODEL_PATH)
    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError("Labels file not found. Train first to generate it.")
    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    return model, labels

def preprocess_bgr_frame(frame_bgr):
    img = cv2.resize(frame_bgr, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ----------------------------
# Realtime webcam
# ----------------------------
def realtime(cam_index=0):
    model, labels = load_model_and_labels()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("🎥 Realtime sign recognition running. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # (Optional) draw a guide box for consistent ROI
        # x1,y1,x2,y2 = 100, 60, 420, 380
        # roi = frame[y1:y2, x1:x2]
        roi = frame  # using full frame is simpler to start

        inp = preprocess_bgr_frame(roi)
        preds = model.predict(inp, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        label = labels[idx] if idx < len(labels) else f"#{idx}"

        cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("GuardianX — Sign Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# Single image test
# ----------------------------
def predict_image(path):
    if not os.path.isfile(path):
        print(f"❌ File not found: {path}")
        return
    model, labels = load_model_and_labels()
    bgr = cv2.imread(path)
    inp = preprocess_bgr_frame(bgr)
    preds = model.predict(inp, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    label = labels[idx] if idx < len(labels) else f"#{idx}"
    print(f"🖼️ {os.path.basename(path)} -> {label} ({conf*100:.1f}%)")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GuardianX Sign Language (TensorFlow)")
    parser.add_argument("--train", action="store_true", help="Train on backend/dataset/")
    parser.add_argument("--realtime", action="store_true", help="Run webcam recognition")
    parser.add_argument("--image", type=str, help="Predict a single image path")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default 0)")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.realtime:
        realtime(cam_index=args.cam)
    elif args.image:
        predict_image(args.image)
    else:
        print("Nothing to do. Use one of: --train | --realtime | --image path/to.jpg")
