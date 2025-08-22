import os

# Base dataset path
base_path = os.path.join("..", "dataset", "sign_language")

# Train and test folders
folders = ["train", "test"]
classes = ["1", "2", "3", "A", "B", "C"]

# Create folder structure
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    os.makedirs(folder_path, exist_ok=True)
    for cls in classes:
        class_path = os.path.join(folder_path, cls)
        os.makedirs(class_path, exist_ok=True)

print("Mini dataset folder structure is ready ✅")
