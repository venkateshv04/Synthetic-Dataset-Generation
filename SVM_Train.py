import os
import numpy as np

# --- CONFIGURATION ---
MERGED_DIR = "merged_vectors"
OUTPUT_DIR = "boundaries"
NEUTRAL_NAME = "neutral" 
# ---------------------

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Neutral (Baseline)
    neutral_path = os.path.join(MERGED_DIR, f"{NEUTRAL_NAME}.npy")
    if not os.path.exists(neutral_path):
        print("❌ CRITICAL: Neutral file not found.")
        return
    
    # Calculate the AVERAGE Neutral Vector (The "Centroid")
    # Shape: (1, 16, 512)
    neutral_vectors = np.load(neutral_path)
    neutral_mean = np.mean(neutral_vectors, axis=0, keepdims=True)
    print(f"✅ Calculated Average {NEUTRAL_NAME} Vector.")

    # 2. Process Targets
    files = [f for f in os.listdir(MERGED_DIR) if f.endswith('.npy') and f != f"{NEUTRAL_NAME}.npy"]

    for filename in files:
        target_name = filename.replace(".npy", "")
        target_path = os.path.join(MERGED_DIR, filename)
        
        # Calculate AVERAGE Target Vector
        target_vectors = np.load(target_path)
        target_mean = np.mean(target_vectors, axis=0, keepdims=True)
        
        # --- THE MAGIC CALCULATION (Difference of Means) ---
        # Direction = Destination - Origin
        direction = target_mean - neutral_mean
        
        # Normalize (make unit length)
        direction = direction / np.linalg.norm(direction)
        
        # Save
        save_name = os.path.join(OUTPUT_DIR, f"direction_{target_name}.npy")
        np.save(save_name, direction)
        
        print(f"   ➡️  Saved Vector: {NEUTRAL_NAME} -> {target_name}")

    print("\n✅ NEW SIMPLE BOUNDARIES CALCULATED.")

if __name__ == "__main__":
    main()