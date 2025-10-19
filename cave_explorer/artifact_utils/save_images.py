import os
import cv2
import time
import math
from datetime import datetime

# Base dataset directory (in your home folder)
DATASET_ROOT = os.path.expanduser('~/cave_dataset')

# --- Simple save parameters ---
SAVE_COOLDOWN = 2.0        # wait at least 3 sec between saves
MIN_SAVE_DISTANCE = 0.5     # only save if moved 0.4 m since last image

last_save_time = 0
last_saved_pose = None

def make_dirs():
    """Create dataset folders if missing."""
    os.makedirs(os.path.join(DATASET_ROOT, 'images'), exist_ok=True)

make_dirs()

def save_image(image, current_pose=None, label='dataset'):
    """Save image at controlled intervals."""
    global last_save_time, last_saved_pose

    now = time.time()
    # --- Time filter ---
    if now - last_save_time < SAVE_COOLDOWN:
        return

    # --- Distance filter ---
    if current_pose and last_saved_pose:
        dist = math.hypot(current_pose.x - last_saved_pose.x,
                          current_pose.y - last_saved_pose.y)
        if dist < MIN_SAVE_DISTANCE:
            return

    # --- Save image ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = os.path.join(DATASET_ROOT, 'images', f'{label}_{timestamp}.jpg')
    cv2.imwrite(path, image)
    print(f"[dataset] saved: {path}")

    last_save_time = now
    last_saved_pose = current_pose

