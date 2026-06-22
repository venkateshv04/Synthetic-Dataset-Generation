import os
import sys
import bz2
import shutil
import dlib
import numpy as np
import PIL.Image
import scipy.ndimage
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_DATA_DIR = r"Final_dataset" 
ALIGNED_DIR = r"Final_dataset_aligned"
LANDMARK_PATH = r"shape_predictor_68_face_landmarks.dat"
# ---------------------

def get_landmark(filepath, predictor, detector):
    """Detects face landmarks."""
    try:
        img = dlib.load_rgb_image(filepath)
    except Exception:
        return None
        
    # Upsample 0 times for speed on CPU
    dets = detector(img, 0)
    if len(dets) == 0:
        return None 
    
    shape = predictor(img, dets[0])
    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    return np.array(a)

def process_single_image(filepath, predictor, detector):
    """Combines Alignment AND Grayscale conversion."""
    lm = get_landmark(filepath, predictor, detector)
    if lm is None:
        return None

    # FFHQ Standard Alignment Math
    lm_eye_left = lm[36:42]
    lm_eye_right = lm[42:48]
    lm_mouth_outer = lm[48:60]

    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    
    # Transform
    img = PIL.Image.open(filepath).convert('RGB')
    transform_size = 4096
    output_size = 1024
    
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    img = img.resize((output_size, output_size), PIL.Image.LANCZOS)
    
    # Grayscale Conversion (Purify) -> RGB
    gray_img = img.convert('L')
    final_img = gray_img.convert('RGB')
        
    return final_img

def main():
    print("Starting Processing (Safe CPU Mode)...")

    # 1. Setup Models (No GPU checks)
    if not os.path.exists(LANDMARK_PATH):
        print(f"❌ ERROR: Landmarks file not found at: {LANDMARK_PATH}")
        return

    print("Loading Dlib Predictor...")
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(LANDMARK_PATH)
    except Exception as e:
        print(f"❌ CRITICAL DLIB ERROR: {e}")
        print("Your Dlib installation might be corrupted.")
        return
    
    # 2. Gather Files
    subfolders = [f for f in os.listdir(RAW_DATA_DIR) 
                  if os.path.isdir(os.path.join(RAW_DATA_DIR, f)) 
                  and "aligned" not in f 
                  and "vectors" not in f]
    
    all_files = []
    for folder in subfolders:
        path = os.path.join(RAW_DATA_DIR, folder)
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            all_files.append((folder, f))

    print(f"Found {len(all_files)} images.")
    
    # 3. Processing Loop
    success_count = 0
    skip_count = 0
    
    with tqdm(total=len(all_files), unit="img") as pbar:
        for folder_name, filename in all_files:
            input_path = os.path.join(RAW_DATA_DIR, folder_name, filename)
            
            output_folder = os.path.join(ALIGNED_DIR, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, filename)
            
            try:
                final_img = process_single_image(input_path, predictor, detector)
                
                if final_img:
                    final_img.save(output_path)
                    success_count += 1
                else:
                    skip_count += 1
                    
            except Exception as e:
                pbar.write(f"Error on {filename}: {e}")
                skip_count += 1
            
            pbar.update(1)
            pbar.set_postfix({"Saved": success_count, "Skipped": skip_count})

    print("-" * 40)
    print(f"COMPLETE. Output saved to: {ALIGNED_DIR}")

if __name__ == "__main__":
    main()