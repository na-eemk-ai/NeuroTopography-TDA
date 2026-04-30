# NeuroTopography — MRI Dataset Sources Guide
## Open-Access Brain MRI Datasets for TDA Research

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install numpy Pillow scipy gudhi

# Option A — No credentials needed (generates realistic synthetic images):
python download_mri_assets.py --source synthetic

# Option B — Figshare (real data, no login, needs scipy):
python download_mri_assets.py --source figshare

# Option C — Kaggle (best quality, needs free account):
pip install kaggle
python download_mri_assets.py --source kaggle

# Auto mode (tries Kaggle → Figshare → Mendeley → synthetic):
python download_mri_assets.py
```

---

## 📚 Dataset 1 — Figshare / Cheng et al. (2017)
**RECOMMENDED: No login required — direct HTTP download**

| Field | Details |
|---|---|
| **URL** | https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 |
| **DOI** | 10.6084/m9.figshare.1512427 |
| **License** | CC BY 4.0 (free for academic use) |
| **Images** | 3,064 T1-weighted contrast-enhanced MRI slices |
| **Classes** | Glioma (1,426) · Meningioma (708) · Pituitary (930) |
| **Format** | .mat (MATLAB) files — script converts automatically |
| **Resolution** | 512 × 512 pixels |
| **Patients** | 233 patients |

**Direct ZIP Download URLs** (no authentication needed):
```
Part 1: https://figshare.com/ndownloader/files/3381290
Part 2: https://figshare.com/ndownloader/files/3381293
Part 3: https://figshare.com/ndownloader/files/3381296
Part 4: https://figshare.com/ndownloader/files/3381299
```
> ⚠️  Note: Normal/healthy tissue is NOT in this dataset.
> The script supplements the `normal` class with synthetic images automatically.

**Citation:**
```
Cheng, J. (2017). Brain tumor dataset [Data set].
figshare. https://doi.org/10.6084/m9.figshare.1512427.v5
```

---

## 📚 Dataset 2 — Kaggle / Masoud Nickparvar (2021)
**Best quality — requires free Kaggle account**

| Field | Details |
|---|---|
| **URL** | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| **License** | CC BY-SA 4.0 |
| **Images** | 7,023 MRI images |
| **Classes** | Glioma (1,621) · Meningioma (1,645) · Pituitary (1,757) · No Tumor (2,000) |
| **Format** | JPG — directly usable |
| **Split** | Training (80%) + Testing (20%) |

**Setup Kaggle API:**
```bash
# 1. Create free account at kaggle.com
# 2. Go to: kaggle.com/settings → API → Create New Token
# 3. This downloads kaggle.json
# 4. Move it:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 5. Install and test:
pip install kaggle
kaggle datasets list --search "brain tumor"
```

**Citation:**
```
Nickparvar, M. (2021). Brain Tumor MRI Dataset [Data set].
Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
```

---

## 📚 Dataset 3 — Mendeley Data (2021)
**Alternative — CC BY 4.0, no login for API access**

| Field | Details |
|---|---|
| **URL** | https://data.mendeley.com/datasets/zwr4ntf94j/6 |
| **DOI** | 10.17632/zwr4ntf94j.6 |
| **License** | CC BY 4.0 |
| **Images** | 11,148 T1-weighted contrast-enhanced MRIs |
| **Classes** | Glioma · Meningioma · Pituitary · No_Tumor |
| **Format** | JPEG / PNG |

**Citation:**
```
Brain Tumor MRI Dataset (2021). Mendeley Data, V6.
https://doi.org/10.17632/zwr4ntf94j.6
```

---

## 📚 Dataset 4 — BRISC (2025)
**Most recent — radiologist-annotated, includes segmentation masks**

| Field | Details |
|---|---|
| **Figshare** | https://figshare.com/articles/dataset/30533120 |
| **Kaggle** | https://www.kaggle.com/datasets/brisc2025 |
| **License** | CC BY 4.0 |
| **Images** | 6,000 MRI scans with pixel-level annotations |
| **Classes** | Glioma · Meningioma · Pituitary · No Tumor |
| **Planes** | Axial ✓ · Coronal · Sagittal |
| **Extras** | Radiologist-reviewed segmentation masks |

**Citation:**
```
BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification (2025).
figshare. https://figshare.com/articles/dataset/30533120
arXiv: https://arxiv.org/abs/2506.14318
```

---

## 🔬 Pre-processing Pipeline Details

Each downloaded image goes through this pipeline before saving:

```
Input (any size, any format)
        ↓
1. Convert to Grayscale (L mode)
        ↓
2. Gaussian Denoising (σ = 0.8)
   — reduces MRI acquisition noise
   — smooths salt-and-pepper artifacts
        ↓
3. Contrast Enhancement (percentile stretch)
   — clips to [2nd, 98th] percentile
   — improves TDA threshold sensitivity
        ↓
4. Resize to 256 × 256 px (Lanczos filter)
   — Lanczos = highest quality downsampling
        ↓
5. Normalise to uint8 [0, 255]
        ↓
Output: PNG, 256×256, Grayscale, uint8
```

---

## 🧬 Expected TDA Results

After loading these images into NeuroTopography:

| Class | Pixel Threshold | Expected β₀ | Expected β₁ | Interpretation |
|---|---|---|---|---|
| Normal | 100–130 | 1 | 1 | Single smooth brain boundary |
| Meningioma | 110–140 | 1 | 1–2 | Well-defined mass, clean boundary |
| Pituitary | 110–140 | 1 | 2–3 | Small mass + micro-cavity |
| Glioma | 100–130 | ≥2 | ≥3 | Irregular + multiple necrotic voids |

**Recommended NeuroTopography settings:**
```
Pixel Threshold  : 110 – 130
Gaussian σ       : 1.0 – 1.3
Max Points       : 200 – 350
ε (Epsilon)      : 0.35 – 0.55
Max Edge Length  : 1.6 – 2.0
```

---

## 📁 Output Folder Structure

```
assets/
├── glioma/
│   ├── glioma_001.png
│   ├── glioma_002.png
│   └── ...
├── meningioma/
│   ├── meningioma_001.png
│   └── ...
├── pituitary/
│   ├── pituitary_001.png
│   └── ...
├── normal/
│   ├── normal_001.png
│   └── ...
└── dataset_info.json       ← metadata, citations, validation report
```

---

## ⚖️ License & Academic Use

All datasets listed here are free for academic, research, and educational use:

| Dataset | License | Commercial Use |
|---|---|---|
| Figshare (Cheng 2017) | CC BY 4.0 | ✓ with attribution |
| Kaggle (Nickparvar) | CC BY-SA 4.0 | ✓ with attribution + share-alike |
| Mendeley | CC BY 4.0 | ✓ with attribution |
| BRISC | CC BY 4.0 | ✓ with attribution |

**⚠️ Medical Disclaimer:** All datasets are de-identified / anonymized.
Results from TDA analysis are for research purposes only and must not
be used for clinical diagnosis without expert medical review.

---

## 🛠️ Troubleshooting

**"scipy.io not available" when using Figshare:**
```bash
pip install scipy
```

**"kaggle package not installed":**
```bash
pip install kaggle
```

**"kaggle.json not found":**
- Go to https://www.kaggle.com/settings
- Scroll to "API" section
- Click "Create New Token"
- Move downloaded file: `mv kaggle.json ~/.kaggle/kaggle.json`
- Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

**Figshare download fails (slow connection):**
```bash
# Download manually from browser:
# https://figshare.com/ndownloader/files/3381290
# Save to current directory, then re-run script
```

**GUDHI not installed:**
```bash
pip install gudhi
# or with conda:
conda install -c conda-forge gudhi
```
