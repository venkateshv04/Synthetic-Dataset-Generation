import os
import numpy as np

# --- CONFIGURATION ---
# The folder where your individual .npy files are (from the previous step)
INPUT_ROOT = r"vectors"
# Where to save the big merged files
OUTPUT_DIR = r"merged_vectors"
# ---------------------

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Scanning '{INPUT_ROOT}' for emotion folders...")
    
    # Get all subfolders (Attentiveness, Confusion, etc.)
    folders = [f for f in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, f))]
    
    if not folders:
        print("❌ Error: No folders found. Did you run the inversion script?")
        return

    for emotion in folders:
        print(f"\nProcessing Emotion: {emotion}...")
        emotion_path = os.path.join(INPUT_ROOT, emotion)
        
        # Find all .npy files
        files = [f for f in os.listdir(emotion_path) if f.endswith('.npy')]
        
        if len(files) == 0:
            print(f"  ⚠️  Skipping {emotion} (No .npy files found)")
            continue

        vector_list = []
        for i, f in enumerate(files):
            # Load individual vector
            file_path = os.path.join(emotion_path, f)
            w = np.load(file_path)
            vector_list.append(w)
            
            # Show progress every 100 images
            if i % 100 == 0:
                print(f"  Loading {i}/{len(files)}...", end="\r")

        # STACK THEM: Convert list of arrays into one big array
        # Shape changes from (1, 16, 512) -> (Count, 16, 512)
        merged_data = np.stack(vector_list)
        
        # Remove the extra dimension (1) if it exists
        # If your vectors are shape (1, 16, 512), we want (Count, 16, 512)
        if merged_data.ndim == 4 and merged_data.shape[1] == 1:
             merged_data = merged_data.squeeze(1)

        # Save
        save_name = os.path.join(OUTPUT_DIR, f"{emotion}.npy")
        np.save(save_name, merged_data)
        
        print(f"  ✅ Done! Saved {len(files)} vectors to: {save_name}")
        print(f"     Final Shape: {merged_data.shape}")

    print("\n" + "="*50)
    print("ALL MERGED. You are ready for the SVM Step.")
    print("="*50)

if __name__ == "__main__":
    main()