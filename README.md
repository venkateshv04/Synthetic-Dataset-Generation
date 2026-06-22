# Synthetic Data Generation (StyleGAN3 + Hybrid Emotion Pipeline)

A production-ready pipeline for **synthetic facial emotion dataset generation** and **emotion recognition**, combining **StyleGAN3 latent space manipulation** with a clean downstream CNN training setup.

This repository is designed to be **clonable and directly usable**. You **do not need to repeat the research or boundary-creation steps**. All required artifacts (latent vectors, merged vectors, and emotion boundaries) are already included.

Your only responsibility after cloning is:

1. Set up the environment  
2. Run `generate_dataset.py`

---

## What This Repository Does

- Generates **high-quality synthetic facial emotion images** using **StyleGAN3**
- Supports **fine-grained emotions** like *Confusion*, *Frustration*, and *Attentiveness*
- Uses **latent direction vectors** with tunable intensity and emotion mixing
- Enables **one-command dataset generation** for research, training, or benchmarking
- Produces data in a clean, class-wise folder structure ready for CNN training

---

## Repository Structure

```
Synthetic-Data-Generation/
│
├── requirements.txt
├── preprocess.py              # Face alignment + normalization (optional)
├── generate_dataset.py        # MAIN ENTRY POINT (you will use this)
├── merge_vectors.py           # Latent vector consolidation (pre-done)
├── train.py                   # CNN training on hybrid dataset
│
├── latent_vectors/            # Precomputed latent vectors (per emotion)
│   ├── Happy/
│   ├── Surprise/
│   ├── Frustration/
│   ├── Confusion/
│   ├── Attentiveness/
│   └── neutral/
│
├── merged_latent_vectors/     # Final merged latent representations
│   ├── Happy.npy
│   ├── Surprise.npy
│   ├── Frustration.npy
│   ├── Confusion.npy
│   ├── Attentiveness.npy
│   └── neutral.npy
│
├── Boundaries/                # Emotion direction vectors (READY TO USE)
│   ├── direction_Attentiveness.npy
│   ├── direction_Confusion.npy
│   ├── direction_Frustration.npy
│   ├── direction_Happy.npy
│   └── direction_Surprise.npy
│
├── SVM_Train/                 
│
└── Generated_Dataset_Sample/ 
    ├── Happy/
    ├── Surprise/
    ├── Frustration/
    ├── Confusion/
    └── Attentiveness/
```

---

## Environment Setup

### 0. Repository Layout (Required)

This project **requires the official StyleGAN3 repository** to be cloned inside the root directory.

Final expected structure:

```
Synthetic-Data-Generation/
│
├── stylegan3/                  # Official NVlabs StyleGAN3 repo
││   ├── dnnlib/
││   ├── torch_utils/
││   ├── legacy.py
││   └── ...
│
├── requirements.txt
├── preprocess.py
├── generate_dataset.py
├── merge_vectors.py
├── train.py
│
├── latent_vectors/
├── merged_latent_vectors/
├── Boundaries/
├── SVM_Train/
└── Generated_Dataset_Sample/
```

`generate_dataset.py` directly imports StyleGAN3 internals. This structure **must not be changed**.

---

### 1. Clone the Official StyleGAN3 Repository

```bash
git clone https://github.com/NVlabs/stylegan3.git
```

---

### 2. Download Pretrained StyleGAN3 Model (Required)

Place the model **inside the `stylegan3/` directory**:

```bash
curl -L 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan3/1/files?redirect=true&path=stylegan3-t-ffhq-1024x1024.pkl' \
     -o stylegan3/stylegan3-t-ffhq-1024x1024.pkl
```

---

### 3. (Optional) Download dlib Landmark Model (Only for preprocess.py)

```bash
curl -L https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2 \
     -o shape_predictor_68_face_landmarks.dat.bz2

bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

---

### 4. Create a Virtual Environment

#### 1. Create an Environment

```bash
conda create -n emotion_gan python=3.9 -y
conda activate emotion_gan
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

GPU Required:

- NVIDIA GPU  
- CUDA compatible PyTorch  
- Visual Studio Build Tools (Windows) or GCC (Linux)

---

## Dataset Generation (Main Usage)

### Generate ALL Emotions (Default)

```bash
python generate_dataset.py
```

This will:

- Load the pretrained StyleGAN3 model
- Apply emotion-specific latent directions
- Generate images into:

```
Generated dataset sample/
```

Each emotion will have its own folder.

---

### Generate a Single Emotion Only

```bash
python generate_dataset.py --emotion Frustration
```

Supported emotion names (case-sensitive):

- `Happy`
- `Surprise`
- `Frustration`
- `Confusion`
- `Attentiveness`

---

## Emotion Control Logic (Already Tuned)

Each emotion is generated using a **latent arithmetic recipe**:

```
w_emotion = w + (direction × strength × sign) − (happy_vector × mix_amount)
```

Where:

- **strength** → emotion intensity  
- **sign** → direction inversion  
- **mix_amount** → smile suppression (0.0 → 1.0+)  

These values are **final and stable**. You are not expected to tune them again unless doing research.

---

## Preprocessing (Optional, Already done)

If you want to preprocess **real images** before training:

```bash
python preprocess.py
```

Includes:

- Face alignment  
- CLAHE  
- Normalization  

Synthetic images do **not** require preprocessing.

---

## Training the Emotion Classifier (Optional, Already done)

Once synthetic data is generated:

```bash
python train.py
```

This trains a **CNN-based emotion classifier** using:

- Real FER data (optional)
- GAN-generated synthetic data
- Balanced emotion classes

Outputs:

- Accuracy, Precision, Recall, F1
- Training & validation plots

---

## Happiness Bias & Why It Matters

### The "Happy Face" Bias Problem

Both **real-world FER datasets** and **StyleGAN-based generators** suffer from a well-known issue:

- Real datasets are **heavily skewed toward smiling / neutral-positive faces**
- StyleGAN (trained on FFHQ-like data) learns a **strong prior toward pleasant expressions**
- Subtle or negative emotions collapse into neutral or slight smiles

This phenomenon is known as **happiness bias**.

---

### How This Repository Fixes It

This pipeline **counteracts happiness bias in latent space**, not post-hoc.

Key strategies:

- **Explicit Happy Direction Extraction**  
  `direction_Happy.npy` is treated as a separable semantic component.

- **Anti-Smile Latent Mixing**

```
w_emotion = w + (emotion_direction × strength × sign)
            − (happy_direction × anti_smile_amount)
```

- **Continuous Control (0.0 → 1.0+)**
- **Emotion-Specific Policies**
  - Confusion → partial mouth opening
  - Attentiveness → full smile suppression
  - Frustration → aggressive happiness removal

---

### Result

- Confusion ≠ Attentiveness  
- Frustration ≠ Neutral  
- Happy does not leak into other classes  

This correction is essential for training reliable emotion classifiers.

---

## Why This Works

- Avoids data scarcity for rare emotions  
- Enables controllable expression synthesis  
- Preserves identity realism  
- Eliminates manual annotation  
- Fully privacy-preserving  

This repository behaves like a **tool**, not a demo.

---

## Troubleshooting

### `ModuleNotFoundError: numpy`

Wrong environment active.

```bash
conda activate emotion_gan
pip install -r requirements.txt
```

---

### CUDA / bias_act_plugin Errors

Ensure:

- CUDA matches PyTorch  
- Build tools installed  
- `nvcc --version` works  

If needed:

```bash
set TORCH_CUDA_ARCH_LIST=8.6
```

---

## License

Released for **research and development use**.

---

## Final Note

If you want to:

- Generate emotion datasets  
- Balance rare classes  
- Train FER models  

This repository is ready as-is.

Clone. Install. Generate.