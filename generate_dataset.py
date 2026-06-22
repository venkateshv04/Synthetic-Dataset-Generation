import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
import dnnlib
import legacy

# --- CONFIGURATION ---
NETWORK_PKL = "stylegan3-t-ffhq-1024x1024.pkl"
BOUNDARIES_DIR = r"boundaries"
OUTPUT_DIR = "Final_Synthetic_Dataset"
IMAGES_PER_EMOTION = 1000
TRUNCATION_PSI = 0.7  

# --- EMOTION SETTINGS ---
# Format: "Emotion": (Strength, Sign, Anti_Smile_Amount)
# Anti_Smile_Amount: 0.0 = Keep smile, 1.0 = Remove smile fully, 0.5 = Mix
EMOTION_SETTINGS = {
    # 1. FRUSTRATION (Heavy Duty)
    # Strength 10 + Full Anti-Smile (1.0) to look annoyed
    "Frustration":   (26.0, 0.7, 1.0),  
    
    # 2. CONFUSION (Natural)
    # Strength 5.0 + Zero Anti-Smile (0.0) so mouth can open slightly
    "Confusion":     (27.8,  0.7, 1.0),  
    
    # 3. ATTENTIVENESS (Focus)
    # Strength 3.0 + Full Anti-Smile (1.0) to look serious/focused
    "Attentiveness": (3.0,   1.0, 0.4),  
    
    # 4. SURPRISE (Mouth Open)
    # Strength 10.0 + Zero Anti-Smile (0.0)
    "Surprise":      (-10.0,  -1.2, 0.8), 
    
    # 5. HAPPY (Smiling)
    # Strength 1.5 + Zero Anti-Smile (0.0) obviously
    "Happy":         (1.5,   1.0, 0.0), 
    
    # Default
    "default":       (10.0,  1.0, 1.0)   
}
# ---------------------

def main():
    # 1. ARGUMENT PARSER
    parser = argparse.ArgumentParser(description="Generate synthetic dataset.")
    parser.add_argument("--emotion", type=str, default=None, 
                        help="Specific emotion to generate (e.g., Frustration). If empty, generates all.")
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. IDENTIFY TASKS
    all_files = [f for f in os.listdir(BOUNDARIES_DIR) if f.startswith('direction_') and f.endswith('.npy')]
    
    if args.emotion:
        target_filename = f"direction_{args.emotion}.npy"
        if target_filename in all_files:
            direction_files = [target_filename]
            print(f"🎯 TARGET MODE: Generating only '{args.emotion}'")
        else:
            print(f"❌ Error: Emotion file '{target_filename}' not found.")
            return
    else:
        direction_files = all_files
        print(f"🏭 FACTORY MODE: Generating ALL emotions.")

    # 3. LOAD NETWORK
    print(f"   Loading Network...")
    device = torch.device('cuda')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        print(f"✅ USING GPU: {gpu_name}")
        print(f"   CUDA Version: {cuda_ver}")
    with dnnlib.util.open_url(NETWORK_PKL) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    # 4. LOAD HAPPY VECTOR (For Anti-Smile)
    happy_path = os.path.join(BOUNDARIES_DIR, "direction_Happy.npy")
    if os.path.exists(happy_path):
        direction_happy = np.load(happy_path)
        direction_happy = torch.from_numpy(direction_happy).to(device)
    else:
        direction_happy = None

    # 5. GENERATION LOOP
    for d_file in direction_files:
        emotion_name = d_file.replace("direction_", "").replace(".npy", "")
        
        # Get settings (Now reading the 3rd number as float)
        strength, sign, anti_smile_amount = EMOTION_SETTINGS.get(emotion_name, EMOTION_SETTINGS["default"])
        
        print(f"\n🎨 Task: {emotion_name} | Strength: {strength} | Anti-Smile: {anti_smile_amount}")

        direction = np.load(os.path.join(BOUNDARIES_DIR, d_file))
        direction = torch.from_numpy(direction).to(device)

        emotion_out = os.path.join(OUTPUT_DIR, emotion_name)
        if not os.path.exists(emotion_out): os.makedirs(emotion_out)

        for i in range(IMAGES_PER_EMOTION):
            z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
            w = G.mapping(z, None, truncation_psi=TRUNCATION_PSI)
            
            # A. Base Emotion (Direction * Strength * Sign)
            w_emotion = w + (direction * strength * sign)
            
            # B. Apply Variable Anti-Smile
            # Formula: Vector - (HappyVector * 10.0 * Amount)
            if anti_smile_amount > 0.0 and direction_happy is not None:
                 w_emotion = w_emotion - (direction_happy * 10.0 * anti_smile_amount)
            
            # Synthesize
            img = G.synthesis(w_emotion, noise_mode='const', force_fp32=True)
            
            # Save
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            Image.fromarray(img[0].cpu().numpy(), 'RGB').save(os.path.join(emotion_out, f"gen_{i:04d}.jpg"))
            print(f"   [{i+1}/{IMAGES_PER_EMOTION}] Saved...", end="\r")

    print("\n\n✅ DONE.")

if __name__ == "__main__":
    main()