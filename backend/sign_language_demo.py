import os
import shutil
import random

# ----- CONFIG -----
dataset_path = r"C:\Users\totek\Documents\GitHub\GuardianX\dataset\sign_language\test"
train_folder = os.path.join(dataset_path, "train")
test_folder = os.path.join(dataset_path, "test")
split_ratio = 0.7  # 70% train, 30% test

# Choose 3 classes for mini dataset
classes = ["A", "B", "1"]

for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        print(f"Class folder '{class_path}' does not exist. Skipping.")
        continue

    # List all images in the class folder
    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png"))]
    if len(images) == 0:
        print(f"No images found in '{class_path}'. Skipping.")
        continue

    # Shuffle images and split into train/test
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create train/test folders for the class
    os.makedirs(os.path.join(train_folder, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_folder, class_name), exist_ok=True)

    # Copy images to train folder
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_folder, class_name, img))

    # Copy images to test folder
    for img in test_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_folder, class_name, img))

    print(f"Class '{class_name}': {len(train_images)} train, {len(test_images)} test images.")

print("Mini dataset ready ✅")
