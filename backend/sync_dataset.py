import os
import shutil
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Paths
source_folder = r"C:\Users\totek\Desktop\Indian"
destination_folder = r"C:\Users\totek\Documents\GitHub\GuardianX\dataset"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

class DatasetHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            src_file = event.src_path
            dest_file = os.path.join(destination_folder, os.path.basename(src_file))
            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file)
                print(f"New file copied: {os.path.basename(src_file)}")
            else:
                print(f"File already exists: {os.path.basename(src_file)}")

# Set up observer
observer = Observer()
observer.schedule(DatasetHandler(), path=source_folder, recursive=False)
observer.start()

print(f"Watching {source_folder} for new files... Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
