import time
import os
import sys
import cv2
import numpy as np
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json

# Load config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

INPUT_DIR = config["INPUT_DIR"]
OUTPUT_DIR = config["OUTPUT_DIR"]
MASK_FILENAME = config["MASK_FILENAME"]

# Image Processing Settings (From your script)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
MASK_SIZE = 45
PADDING_RIGHT = 30
PADDING_BOTTOM = 30

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get absolute path of the mask to avoid "File not found" errors
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_PATH = os.path.join(SCRIPT_DIR, MASK_FILENAME)

# --- IMAGE PROCESSING LOGIC (Your Code) ---

def resize_image(input_path):
    # Load and resize the input image to 1280x720
    print(f"[Processing] Loading and resizing {input_path}...")
    try:
        with Image.open(input_path) as img:
            resized_img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
            if resized_img.mode != 'RGB':
                resized_img = resized_img.convert('RGB')
            img_cv = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)
        return img_cv
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def create_mask(mask_path):
    # Load mask, resize to 64x64, and position at bottom-right corner
    # Note: We load this every time to keep logic simple, 
    # but for optimization, you could load this once at startup.
    
    with Image.open(mask_path) as mask_img:
        mask_resized = mask_img.resize((MASK_SIZE, MASK_SIZE), Image.Resampling.LANCZOS)
        if mask_resized.mode != 'L':
            mask_resized = mask_resized.convert('L')
    
    offset_x = TARGET_WIDTH - MASK_SIZE - PADDING_RIGHT
    offset_y = TARGET_HEIGHT - MASK_SIZE - PADDING_BOTTOM
    
    mask_canvas = Image.new('L', (TARGET_WIDTH, TARGET_HEIGHT), 0)
    mask_canvas.paste(mask_resized, (offset_x, offset_y))
    mask_cv = np.array(mask_canvas)
    
    _, mask_binary = cv2.threshold(mask_cv, 127, 255, cv2.THRESH_BINARY)
    return mask_binary

def inpaint_image(img_cv, mask_binary):
    # Apply content-aware inpainting
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask_binary, kernel, iterations=2)
    
    # Using 'ns' method as default per your script preference
    result_cv = cv2.inpaint(img_cv, mask_dilated, inpaintRadius=15, flags=cv2.INPAINT_NS)
    return result_cv

def process_file_logic(input_path, output_path):
    if not os.path.exists(MASK_PATH):
        print(f"CRITICAL ERROR: Mask file not found at {MASK_PATH}")
        return

    # 1. Resize
    img_cv = resize_image(input_path)
    if img_cv is None: return

    # 2. Create Mask
    mask_binary = create_mask(MASK_PATH)

    # 3. Inpaint
    result_cv = inpaint_image(img_cv, mask_binary)

    # 4. Save
    print(f"Saving result to {output_path}...")
    result_img = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
    result_img.save(output_path, quality=95)
    print(f"✓ DONE: {os.path.basename(output_path)}")

# --- WATCHDOG HANDLER (The Trigger) ---

class WatermarkHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        filename = os.path.basename(event.src_path)
        
        # Check for valid image extensions
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            return

        # Slight delay to ensure file write is complete (prevents reading half-written files)
        time.sleep(1)

        print(f"\n--- New Image Detected: {filename} ---")
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            process_file_logic(event.src_path, output_path)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # check for mask file before starting
    if not os.path.exists(MASK_PATH):
        print(f"Error: '{MASK_FILENAME}' not found in {SCRIPT_DIR}")
        print("Please place your mask file in the same folder as this script.")
        sys.exit(1)

    observer = Observer()
    event_handler = WatermarkHandler()
    
    observer.schedule(event_handler, INPUT_DIR, recursive=False)
    observer.start()
    
    print(f"Started Watching: {INPUT_DIR}")
    print(f"Saving To:      {OUTPUT_DIR}")
    print(f"Using Mask:     {MASK_PATH}")
    print("Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
