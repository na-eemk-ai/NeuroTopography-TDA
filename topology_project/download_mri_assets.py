#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  NeuroTopography — MRI Asset Downloader & Pre-Processor                 ║
║  Version 1.0  |  Academic Research Use Only                             ║
╚══════════════════════════════════════════════════════════════════════════╝

PURPOSE
───────
Downloads real, open-access MRI images from three academic sources and
pre-processes them into 256×256 grayscale PNGs ready for GUDHI TDA analysis.

DATASET SOURCES (all CC BY 4.0 — free for academic use)
────────────────────────────────────────────────────────
SOURCE A │ Figshare — Cheng et al. (2017)
          │ "Brain tumor dataset" — 3,064 T1-weighted contrast-enhanced MRIs
          │ DOI: 10.6084/m9.figshare.1512427
          │ Labels: Glioma (1426), Meningioma (708), Pituitary (930)
          │ Format: .mat (MATLAB) files → extracted to JPG

SOURCE B │ Kaggle — Masoud Nickparvar (2021)
          │ "Brain Tumor MRI Dataset" — 7,023 classified MRI images
          │ URL: kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
          │ Labels: Glioma, Meningioma, Pituitary, No Tumor (Normal)
          │ Format: JPG (directly usable)
          │ Requires: Kaggle API credentials (~/.kaggle/kaggle.json)

SOURCE C │ Mendeley Data — Msoud et al. (2021)
          │ "Brain Tumor MRI Dataset" — 11,148 images, CC BY 4.0
          │ DOI: 10.17632/zwr4ntf94j.6
          │ Labels: Glioma, Meningioma, Pituitary, No_Tumor
          │ Format: JPEG

OUTPUT
──────
assets/
├── glioma/          ← glioma_001.png … glioma_010.png (10 samples)
├── meningioma/      ← meningioma_001.png … (10 samples)
├── pituitary/       ← pituitary_001.png … (10 samples)
├── normal/          ← normal_001.png … (10 samples)
└── dataset_info.json

PRE-PROCESSING PIPELINE (each image)
──────────────────────────────────────
1. Load  →  2. Grayscale  →  3. CLAHE contrast enhance
4. Gaussian denoise (σ=0.8)  →  5. Resize 256×256 (Lanczos)
6. Normalise [0, 255] uint8  →  7. Save as PNG

USAGE
─────
  python download_mri_assets.py

  # Kaggle API method only:
  python download_mri_assets.py --source kaggle

  # Figshare method only (no credentials needed):
  python download_mri_assets.py --source figshare

  # Synthetic fallback (no internet needed):
  python download_mri_assets.py --source synthetic
"""

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from typing import Optional

# ── Third-party (install if missing) ─────────────────────────────
try:
    import numpy as np
    from PIL import Image, ImageFilter
    from scipy.ndimage import gaussian_filter
    print("✓  numpy, Pillow, scipy found")
except ImportError as e:
    print(f"[ERROR] Missing library: {e}")
    print("       Run:  pip install numpy Pillow scipy")
    sys.exit(1)

# Optional for Kaggle source
try:
    import kaggle  # noqa: F401
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# Optional for .mat files (Figshare source)
try:
    import scipy.io as sio
    SCIPY_IO_AVAILABLE = True
except ImportError:
    SCIPY_IO_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
ASSETS_DIR     = Path("assets")
TARGET_SIZE    = (256, 256)
SAMPLES_PER_CLASS = 10    # how many images to keep per class

CLASSES = ["glioma", "meningioma", "pituitary", "normal"]

# Figshare direct download URLs for the 4 .zip parts
# Cheng et al. (2017)  DOI: 10.6084/m9.figshare.1512427
FIGSHARE_ZIPS = [
    "https://figshare.com/ndownloader/files/3381290",   # cvs1.zip (766 images)
    "https://figshare.com/ndownloader/files/3381293",   # cvs2.zip
    "https://figshare.com/ndownloader/files/3381296",   # cvs3.zip
    "https://figshare.com/ndownloader/files/3381299",   # cvs4.zip
]
# Label mapping in .mat files: 1=meningioma, 2=glioma, 3=pituitary
FIGSHARE_LABEL_MAP = {1: "meningioma", 2: "glioma", 3: "pituitary"}

# Kaggle dataset identifier
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"


# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════

def print_header(text: str):
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(bar)


def print_step(step: str, status: str = "", ok: bool = True):
    icon = "✓" if ok else "✗"
    line = f"  [{icon}] {step}"
    if status:
        line += f"  →  {status}"
    print(line)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, label: str = "") -> bool:
    """
    Download a file from URL to dest path.
    Returns True on success.
    Shows progress during download.
    """
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "NeuroTopography-Research/1.0 (academic)"
        })
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 64  # 64 KB
            with open(dest, "wb") as f:
                while True:
                    block = response.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total > 0:
                        pct = downloaded / total * 100
                        bar_len = 30
                        filled = int(bar_len * downloaded / total)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        print(f"\r  {label}  [{bar}]  {pct:5.1f}%", end="", flush=True)
        print()  # newline after progress
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"\n  [!] Download failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# PRE-PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════

def preprocess_image(
    pil_img: Image.Image,
    target_size: tuple = TARGET_SIZE,
    sigma: float = 0.8,
    clahe: bool = True,
) -> Image.Image:
    """
    Pre-processing pipeline for TDA:

    Steps:
      1. Convert to grayscale (L mode)
      2. Gaussian denoise   — reduces MRI acquisition noise
      3. CLAHE-style contrast enhance via histogram equalisation
         (improves structural visibility for thresholding)
      4. Resize to target_size using Lanczos (best quality)
      5. Normalise to [0, 255] uint8

    Parameters
    ----------
    pil_img    : Input PIL image (any mode)
    target_size: Output (width, height) in pixels
    sigma      : Gaussian blur sigma for denoising
    clahe      : Apply contrast enhancement

    Returns
    -------
    PIL Image in mode 'L', size target_size, dtype uint8
    """
    # 1. Grayscale
    img = pil_img.convert("L")
    arr = np.array(img, dtype=np.float64)

    # 2. Gaussian denoise
    if sigma > 0:
        arr = gaussian_filter(arr, sigma=sigma)

    # 3. Contrast enhancement (global histogram stretch)
    if clahe:
        p_low  = np.percentile(arr, 2)
        p_high = np.percentile(arr, 98)
        if p_high > p_low:
            arr = np.clip((arr - p_low) / (p_high - p_low) * 255, 0, 255)

    # 4. Normalise to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # 5. Resize
    img_out = Image.fromarray(arr, mode="L")
    img_out = img_out.resize(target_size, Image.LANCZOS)

    return img_out


def save_sample(img: Image.Image, class_name: str, index: int) -> Path:
    """Save a pre-processed image to assets/<class_name>/<class>_NNN.png"""
    out_dir = ASSETS_DIR / class_name
    ensure_dir(out_dir)
    fname   = out_dir / f"{class_name}_{index:03d}.png"
    img.save(fname, format="PNG", optimize=False)
    return fname


# ══════════════════════════════════════════════════════════════════
# SOURCE A — FIGSHARE (.mat format)
# ══════════════════════════════════════════════════════════════════

def download_figshare(n_per_class: int = SAMPLES_PER_CLASS) -> dict:
    """
    Download and extract the Cheng et al. (2017) Figshare dataset.

    The dataset stores images as MATLAB .mat structs:
      cjdata.image  → pixel array (512×512)
      cjdata.label  → 1=meningioma, 2=glioma, 3=pituitary
      cjdata.tumorBorder → mask coordinates

    Returns dict mapping class_name → list of saved paths.
    """
    if not SCIPY_IO_AVAILABLE:
        print_step("scipy.io not available — cannot parse .mat files", ok=False)
        print("       Run:  pip install scipy")
        return {}

    print_header("SOURCE A — Figshare (Cheng et al., 2017)")
    print("  DOI: 10.6084/m9.figshare.1512427  |  License: CC BY 4.0")
    print("  Note: Glioma (1426) + Meningioma (708) + Pituitary (930)")
    print("  [Normal tissue NOT included — use Source B for normal class]")

    tmp_dir = Path("_figshare_tmp")
    ensure_dir(tmp_dir)

    collected = {cls: [] for cls in CLASSES}
    needed    = {cls: n_per_class for cls in CLASSES if cls != "normal"}

    for zip_idx, zip_url in enumerate(FIGSHARE_ZIPS, 1):
        if all(len(v) >= n_per_class for k, v in collected.items() if k != "normal"):
            print_step(f"All classes satisfied — skipping remaining zips")
            break

        zip_path = tmp_dir / f"figshare_part{zip_idx}.zip"
        print(f"\n  Downloading part {zip_idx}/4 ...")

        if not zip_path.exists():
            ok = download_file(zip_url, zip_path, label=f"Part {zip_idx}")
            if not ok:
                print_step(f"Part {zip_idx} download failed — skipping", ok=False)
                continue
        else:
            print_step(f"Part {zip_idx} already cached")

        # Extract and process .mat files
        print(f"  Extracting part {zip_idx} ...")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                mat_files = [n for n in zf.namelist() if n.endswith(".mat")]
                random.shuffle(mat_files)  # randomise selection

                for mat_name in mat_files:
                    try:
                        mat_bytes = zf.read(mat_name)
                        mat_buf   = io.BytesIO(mat_bytes)
                        mat_data  = sio.loadmat(mat_buf)

                        # Navigate the struct
                        if "cjdata" not in mat_data:
                            continue
                        cj    = mat_data["cjdata"]
                        label = int(cj["label"][0, 0].flat[0])
                        image = cj["image"][0, 0]

                        cls_name = FIGSHARE_LABEL_MAP.get(label)
                        if cls_name is None:
                            continue
                        if len(collected[cls_name]) >= n_per_class:
                            continue

                        # Convert numpy array to PIL
                        img_arr  = np.array(image, dtype=np.float64)
                        img_pil  = Image.fromarray(
                            np.clip(
                                (img_arr / img_arr.max() * 255)
                                if img_arr.max() > 0
                                else img_arr,
                                0, 255
                            ).astype(np.uint8), mode="L"
                        )

                        processed = preprocess_image(img_pil)
                        idx       = len(collected[cls_name]) + 1
                        saved     = save_sample(processed, cls_name, idx)
                        collected[cls_name].append(str(saved))
                        print_step(f"{cls_name} {idx:02d}/{n_per_class}  ← {mat_name}")

                    except Exception as err:
                        continue  # silently skip corrupt files

        except zipfile.BadZipFile:
            print_step(f"Part {zip_idx} is corrupt — skipping", ok=False)

    # Clean up temp
    shutil.rmtree(tmp_dir, ignore_errors=True)

    totals = {k: len(v) for k, v in collected.items()}
    print(f"\n  Figshare result: {totals}")
    return collected


# ══════════════════════════════════════════════════════════════════
# SOURCE B — KAGGLE API
# ══════════════════════════════════════════════════════════════════

def download_kaggle(n_per_class: int = SAMPLES_PER_CLASS) -> dict:
    """
    Download via official Kaggle Python API.

    Prerequisites:
      1. pip install kaggle
      2. Create API token at kaggle.com → Account → API → Create New Token
      3. Place kaggle.json in  ~/.kaggle/kaggle.json  (chmod 600)

    Dataset: masoudnickparvar/brain-tumor-mri-dataset
    7,023 JPG images across 4 classes.
    """
    print_header("SOURCE B — Kaggle API (Masoud Nickparvar, 2021)")
    print("  URL: kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
    print("  License: CC BY-SA 4.0  |  7,023 images (Glioma/Meningioma/Pituitary/No_Tumor)")

    if not KAGGLE_AVAILABLE:
        print_step("kaggle package not installed", ok=False)
        print("       Run:  pip install kaggle")
        return {}

    # Check credentials
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        print_step("kaggle.json not found", ok=False)
        print(f"       Expected: {cred_path}")
        print("       Create at: kaggle.com → Account → API → Create New Token")
        return {}

    import kaggle
    tmp_dir = Path("_kaggle_tmp")
    ensure_dir(tmp_dir)

    print("  Downloading dataset (may take a few minutes for 155 MB) ...")
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(tmp_dir),
            unzip=True,
            quiet=False,
        )
    except Exception as e:
        print_step(f"Kaggle download failed: {e}", ok=False)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {}

    # The unzipped structure:
    # Training/  Testing/  → each has  glioma/  meningioma/  pituitary/  notumor/
    kaggle_class_map = {
        "glioma":      "glioma",
        "meningioma":  "meningioma",
        "pituitary":   "pituitary",
        "notumor":     "normal",
        "no_tumor":    "normal",    # alternate folder name
        "normal":      "normal",
    }

    collected = {cls: [] for cls in CLASSES}

    for split in ["Training", "Testing", "train", "test"]:
        split_dir = tmp_dir / split
        if not split_dir.exists():
            continue
        for folder in split_dir.iterdir():
            if not folder.is_dir():
                continue
            cls_name = kaggle_class_map.get(folder.name.lower())
            if cls_name is None:
                continue
            if len(collected[cls_name]) >= n_per_class:
                continue

            img_files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg")) + \
                        sorted(folder.glob("*.png"))
            random.shuffle(img_files)

            for img_path in img_files:
                if len(collected[cls_name]) >= n_per_class:
                    break
                try:
                    img_pil   = Image.open(img_path)
                    processed = preprocess_image(img_pil)
                    idx       = len(collected[cls_name]) + 1
                    saved     = save_sample(processed, cls_name, idx)
                    collected[cls_name].append(str(saved))
                    print_step(f"{cls_name} {idx:02d}/{n_per_class}  ← {img_path.name}")
                except Exception:
                    continue

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n  Kaggle result: {{k: len(v) for k, v in collected.items()}}")
    return collected


# ══════════════════════════════════════════════════════════════════
# SOURCE C — MENDELEY DATA (Direct HTTP)
# ══════════════════════════════════════════════════════════════════

def download_mendeley(n_per_class: int = SAMPLES_PER_CLASS) -> dict:
    """
    Download a subset from Mendeley Data (Msoud, 2021).

    DOI: 10.17632/zwr4ntf94j.6
    License: CC BY 4.0
    11,148 T1-weighted contrast-enhanced brain MRIs.

    Mendeley provides a direct download endpoint per file.
    We download one zip shard and extract the required classes.
    """
    print_header("SOURCE C — Mendeley Data (DOI: 10.17632/zwr4ntf94j.6)")
    print("  License: CC BY 4.0  |  11,148 images (Glioma/Meningioma/Pituitary/No_Tumor)")

    # Mendeley v2 API endpoint
    api_base = "https://data.mendeley.com/public-files/datasets/zwr4ntf94j/6"

    # First, try to get the file listing via Mendeley API
    files_url = "https://data.mendeley.com/api/datasets/zwr4ntf94j/6/files"

    print("  Fetching dataset file manifest ...")
    try:
        req = urllib.request.Request(
            files_url,
            headers={"Accept": "application/json",
                     "User-Agent": "NeuroTopography-Research/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            manifest = json.loads(resp.read().decode())
    except Exception as e:
        print_step(f"Could not fetch Mendeley manifest: {e}", ok=False)
        return {}

    # Find the zip file(s)
    zip_entries = [
        f for f in manifest
        if isinstance(f, dict) and f.get("filename", "").endswith(".zip")
    ]

    if not zip_entries:
        print_step("No ZIP files found in Mendeley manifest", ok=False)
        return {}

    tmp_dir = Path("_mendeley_tmp")
    ensure_dir(tmp_dir)
    collected = {cls: [] for cls in CLASSES}

    mendeley_class_map = {
        "glioma":    "glioma",
        "meningioma":"meningioma",
        "pituitary": "pituitary",
        "no_tumor":  "normal",
        "notumor":   "normal",
        "normal":    "normal",
    }

    for entry in zip_entries[:2]:  # try first two zips max
        zip_url  = entry.get("content_details", {}).get("download_url") or \
                   entry.get("download_url", "")
        zip_name = entry.get("filename", "mendeley.zip")
        zip_path = tmp_dir / zip_name

        if not zip_url:
            continue

        print(f"  Downloading {zip_name} ...")
        ok = download_file(zip_url, zip_path, label=zip_name)
        if not ok:
            continue

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                all_imgs = [
                    n for n in zf.namelist()
                    if n.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                random.shuffle(all_imgs)

                for img_name in all_imgs:
                    parts    = Path(img_name).parts
                    cls_name = None
                    for part in parts:
                        cls_name = mendeley_class_map.get(part.lower())
                        if cls_name:
                            break
                    if cls_name is None:
                        continue
                    if len(collected[cls_name]) >= n_per_class:
                        continue

                    try:
                        img_bytes = zf.read(img_name)
                        img_pil   = Image.open(io.BytesIO(img_bytes))
                        processed = preprocess_image(img_pil)
                        idx       = len(collected[cls_name]) + 1
                        saved     = save_sample(processed, cls_name, idx)
                        collected[cls_name].append(str(saved))
                        print_step(f"{cls_name} {idx:02d}/{n_per_class}  ← {Path(img_name).name}")
                    except Exception:
                        continue

        except zipfile.BadZipFile:
            print_step(f"{zip_name} corrupt", ok=False)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return collected


# ══════════════════════════════════════════════════════════════════
# FALLBACK — HIGH-QUALITY SYNTHETIC MRI GENERATOR
# Used when no internet or credentials are available.
# Generates realistic-looking MRI-style grayscale images
# that still have genuine topological structure for TDA.
# ══════════════════════════════════════════════════════════════════

def _radial_mask(size, cx, cy, rx, ry, smooth=8):
    """Create an elliptical soft mask."""
    Y, X = np.mgrid[0:size, 0:size]
    d    = np.sqrt(((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2)
    mask = np.clip(1.0 - d, 0, 1)
    return gaussian_filter(mask, sigma=smooth)


def generate_synthetic_mri(class_name: str, idx: int, size: int = 256,
                             seed: int = None) -> Image.Image:
    """
    Generate a synthetic MRI-style image with anatomically-inspired topology
    for a given tumor class.

    Each class has a distinct topological signature:
      normal     → β₀=1, β₁=1  (single smooth ring — skull boundary)
      meningioma → β₀=1, β₁=1  (round, homogeneous enhancing mass)
      pituitary  → β₀=1, β₁=2  (small mass + one cavity)
      glioma     → β₀≥2, β₁≥3  (irregular, multi-cavity, satellite masses)
    """
    rng = np.random.default_rng(seed if seed is not None else hash(f"{class_name}_{idx}") % 2**31)
    img = np.zeros((size, size), dtype=np.float64)
    cx, cy = size // 2, size // 2

    # ── Background: skull + brain parenchyma ──────────────────────
    skull_r = size * 0.44
    brain_r = skull_r * 0.88

    Y, X = np.mgrid[0:size, 0:size]
    skull_d = np.sqrt((X - cx)**2 + (Y - cy)**2)
    brain_mask = skull_d < brain_r
    skull_ring = (skull_d >= brain_r) & (skull_d < skull_r)

    img[brain_mask]  = 0.30 + rng.normal(0, 0.04, brain_mask.sum())
    img[skull_ring]  = 0.18 + rng.normal(0, 0.02, skull_ring.sum())

    # White matter regions
    wm_cx, wm_cy = cx + rng.integers(-20, 20), cy + rng.integers(-15, 15)
    wm_mask      = _radial_mask(size, wm_cx, wm_cy, brain_r * 0.55, brain_r * 0.45, smooth=12)
    img          += wm_mask * (0.18 + rng.random() * 0.06)

    # ── Class-specific tumor topology ─────────────────────────────
    if class_name == "normal":
        # No mass — just add ventricle-like structures (dark CSF)
        v_cx = cx + rng.integers(-18, 18)
        v_cy = cy + rng.integers(-12, 12)
        v_mask = _radial_mask(size, v_cx, v_cy,
                              size * 0.08 + rng.random() * 5,
                              size * 0.05 + rng.random() * 3, smooth=4)
        img -= v_mask * 0.20

    elif class_name == "meningioma":
        # Well-defined, homogeneous, bright, round mass — often peripheral
        angle = rng.random() * 2 * np.pi
        dist  = brain_r * (0.55 + rng.random() * 0.25)
        mx    = int(cx + dist * np.cos(angle))
        my    = int(cy + dist * np.sin(angle))
        mr    = size * (0.09 + rng.random() * 0.05)
        mass  = _radial_mask(size, mx, my, mr, mr * (0.8 + rng.random() * 0.3), smooth=6)
        img  += mass * (0.55 + rng.random() * 0.15)

    elif class_name == "pituitary":
        # Small, well-defined mass at sella turcica (inferior midline)
        mx   = cx + rng.integers(-10, 10)
        my   = cy + int(size * (0.18 + rng.random() * 0.08))
        mr   = size * (0.06 + rng.random() * 0.04)
        mass = _radial_mask(size, mx, my, mr, mr * 0.85, smooth=5)
        img += mass * (0.50 + rng.random() * 0.20)
        # Micro-cavity inside
        ci_r  = mr * 0.35
        cav   = _radial_mask(size, mx, my, ci_r, ci_r, smooth=3)
        img  -= cav * 0.30

    elif class_name == "glioma":
        # Infiltrating, heterogeneous, ring-enhancing + necrotic core(s) + satellite
        mx, my = cx + rng.integers(-30, 30), cy + rng.integers(-25, 25)
        mr     = size * (0.18 + rng.random() * 0.08)

        # Outer enhancing ring
        outer  = _radial_mask(size, mx, my, mr, mr * (0.85 + rng.random() * 0.2), smooth=6)
        img   += outer * (0.48 + rng.random() * 0.18)

        # Irregular boundary perturbations
        for _ in range(6):
            bx  = mx + rng.integers(-int(mr*0.6), int(mr*0.6))
            by  = my + rng.integers(-int(mr*0.6), int(mr*0.6))
            br  = mr * (0.25 + rng.random() * 0.25)
            bump = _radial_mask(size, bx, by, br, br, smooth=4)
            img += bump * rng.random() * 0.22

        # Necrotic core (dark)
        core_r = mr * (0.40 + rng.random() * 0.15)
        core   = _radial_mask(size, mx, my, core_r, core_r, smooth=4)
        img   -= core * (0.28 + rng.random() * 0.12)

        # Second smaller necrotic pocket
        nx, ny = mx + rng.integers(-int(mr*0.4), int(mr*0.4)), \
                 my + rng.integers(-int(mr*0.4), int(mr*0.4))
        nc2    = _radial_mask(size, nx, ny, core_r * 0.45, core_r * 0.45, smooth=3)
        img   -= nc2 * 0.22

        # Satellite mass
        s_angle = rng.random() * 2 * np.pi
        s_dist  = mr * (1.6 + rng.random() * 0.5)
        sx      = int(mx + s_dist * np.cos(s_angle))
        sy      = int(my + s_dist * np.sin(s_angle))
        # Clamp to image bounds
        sx      = max(int(mr), min(size - int(mr) - 1, sx))
        sy      = max(int(mr), min(size - int(mr) - 1, sy))
        sat_r   = mr * (0.22 + rng.random() * 0.12)
        sat     = _radial_mask(size, sx, sy, sat_r, sat_r, smooth=4)
        img    += sat * (0.42 + rng.random() * 0.16)

    # ── Final processing ──────────────────────────────────────────
    img += rng.normal(0, 0.012, img.shape)            # acquisition noise
    img  = gaussian_filter(img, sigma=0.9)             # smooth
    img  = np.clip(img, 0, 1)

    # Contrast stretch
    p2, p98 = np.percentile(img, 2), np.percentile(img, 98)
    if p98 > p2:
        img = np.clip((img - p2) / (p98 - p2), 0, 1)

    arr = (img * 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="L")
    return pil.resize(TARGET_SIZE, Image.LANCZOS)


def generate_synthetic_assets(n_per_class: int = SAMPLES_PER_CLASS) -> dict:
    """
    Generate synthetic MRI images for all 4 classes.
    Used as fallback when real data is unavailable.
    """
    print_header("SYNTHETIC FALLBACK — Anatomically-inspired MRI Generator")
    print("  Generating synthetic images with realistic topological structure.")
    print("  These are NOT real patient data — for demonstration only.")

    collected = {cls: [] for cls in CLASSES}

    for cls in CLASSES:
        print(f"\n  Generating {n_per_class} × {cls} ...")
        for i in range(1, n_per_class + 1):
            img   = generate_synthetic_mri(cls, i)
            saved = save_sample(img, cls, i)
            collected[cls].append(str(saved))
            print_step(f"{cls} {i:02d}/{n_per_class}  →  {saved.name}")

    return collected


# ══════════════════════════════════════════════════════════════════
# VALIDATION & REPORT
# ══════════════════════════════════════════════════════════════════

def validate_assets() -> dict:
    """
    Validate every saved image:
      - File exists and is non-empty
      - Correct dimensions (256×256)
      - Correct mode (L = grayscale)
      - Valid pixel range [0, 255]

    Returns validation report dict.
    """
    print_header("VALIDATION — Checking saved assets")
    report = {
        "total": 0, "valid": 0, "errors": [],
        "classes": {}
    }

    for cls in CLASSES:
        cls_dir = ASSETS_DIR / cls
        if not cls_dir.exists():
            report["classes"][cls] = {"count": 0, "errors": [f"Directory missing"]}
            continue

        pngs    = sorted(cls_dir.glob("*.png"))
        errors  = []
        valid_n = 0

        for p in pngs:
            report["total"] += 1
            try:
                img = Image.open(p)
                assert img.size == TARGET_SIZE, f"Size {img.size} ≠ {TARGET_SIZE}"
                assert img.mode == "L",         f"Mode {img.mode} ≠ L (grayscale)"
                arr = np.array(img)
                assert arr.dtype == np.uint8,   f"dtype {arr.dtype} ≠ uint8"
                assert arr.min() >= 0,          f"Pixel min < 0"
                assert arr.max() <= 255,        f"Pixel max > 255"
                valid_n        += 1
                report["valid"] += 1
                print_step(f"{p.name}  {img.size}  {img.mode}")
            except (AssertionError, Exception) as e:
                errors.append(f"{p.name}: {e}")
                report["errors"].append(str(e))
                print_step(f"{p.name}  FAILED: {e}", ok=False)

        report["classes"][cls] = {"count": valid_n, "errors": errors}

    pct = 100 * report["valid"] / max(report["total"], 1)
    print(f"\n  Result: {report['valid']}/{report['total']} valid ({pct:.0f}%)")
    return report


def write_dataset_info(source: str, collected: dict, validation: dict):
    """Write dataset_info.json to assets/ folder."""
    info = {
        "project":        "NeuroTopography",
        "version":        "1.0",
        "generated_at":   time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "source":         source,
        "image_size":     list(TARGET_SIZE),
        "image_mode":     "L (grayscale)",
        "image_format":   "PNG",
        "preprocessing": {
            "gaussian_sigma":         0.8,
            "contrast_enhancement":   True,
            "normalisation":          "uint8 [0, 255]",
            "resize_filter":          "Lanczos",
        },
        "classes": {
            cls: {
                "count":  len(paths),
                "files":  [Path(p).name for p in paths],
            }
            for cls, paths in collected.items()
        },
        "validation": validation,
        "citations": {
            "figshare": (
                "Cheng, J. (2017). Brain tumor dataset [Data set]. "
                "figshare. https://doi.org/10.6084/m9.figshare.1512427.v5  "
                "License: CC BY 4.0"
            ),
            "kaggle": (
                "Nickparvar, M. (2021). Brain Tumor MRI Dataset [Data set]. "
                "Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/"
                "brain-tumor-mri-dataset  License: CC BY-SA 4.0"
            ),
            "mendeley": (
                "Msoud et al. (2021). Brain Tumor MRI Dataset [Data set]. "
                "Mendeley Data. https://doi.org/10.17632/zwr4ntf94j.6  "
                "License: CC BY 4.0"
            ),
        },
        "tda_notes": (
            "Images pre-processed for GUDHI Vietoris-Rips persistent homology. "
            "Recommended threshold range: 80-140. "
            "Recommended Gaussian sigma for point cloud: 1.0-1.5. "
            "Expected β₁ values: normal≈1, meningioma≈1-2, pituitary≈2-3, glioma≥3."
        ),
    }

    out_path = ASSETS_DIR / "dataset_info.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print_step(f"Metadata saved  →  {out_path}")


# ══════════════════════════════════════════════════════════════════
# QUICK TDA SMOKE TEST
# Verifies that saved images actually work with GUDHI
# ══════════════════════════════════════════════════════════════════

def tda_smoke_test():
    """
    Load one image per class and run a quick persistent homology
    computation to verify the full pipeline works end-to-end.
    """
    try:
        import gudhi
    except ImportError:
        print_step("gudhi not installed — skipping TDA smoke test", ok=False)
        print("       Run:  pip install gudhi")
        return

    print_header("TDA SMOKE TEST — Verifying GUDHI pipeline")

    from scipy.ndimage import gaussian_filter as gf

    for cls in CLASSES:
        cls_dir = ASSETS_DIR / cls
        pngs    = sorted(cls_dir.glob("*.png"))
        if not pngs:
            print_step(f"{cls}  No images found", ok=False)
            continue

        img_path = pngs[0]
        img_arr  = np.array(Image.open(img_path).convert("L"), dtype=np.float64)
        smooth   = gf(img_arr, sigma=1.2)
        threshold = 120.0
        rows, cols = np.where(smooth > threshold)

        if len(rows) < 5:
            print_step(f"{cls}  Too few points at threshold=120 — try lower", ok=False)
            continue

        # Build point cloud
        pts = np.column_stack([cols.astype(float), -rows.astype(float)])
        for d in range(2):
            lo, hi = pts[:, d].min(), pts[:, d].max()
            if hi > lo:
                pts[:, d] = 2 * (pts[:, d] - lo) / (hi - lo) - 1

        # Downsample
        if len(pts) > 200:
            idx = np.random.default_rng(0).choice(len(pts), 200, replace=False)
            pts = pts[idx]

        # Run GUDHI
        rips   = gudhi.RipsComplex(points=pts, max_edge_length=1.8)
        st     = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()
        pers   = st.persistence()

        b1 = sum(1 for dim, (b, d) in pers if dim == 1 and d != float("inf"))
        b0 = sum(1 for dim, (b, d) in pers if dim == 0 and d != float("inf"))

        print_step(
            f"{cls:<12}  {img_path.name}  "
            f"pts={len(pts):3d}  β₀={b0}  β₁={b1}  ✓ GUDHI OK"
        )


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def print_final_summary(collected: dict, source: str):
    print_header("DOWNLOAD COMPLETE")
    total = sum(len(v) for v in collected.values())
    for cls in CLASSES:
        n = len(collected.get(cls, []))
        bar = "█" * n + "░" * (SAMPLES_PER_CLASS - n)
        print(f"  {cls:<12}  [{bar}]  {n}/{SAMPLES_PER_CLASS}")
    print(f"\n  Total: {total} images  →  {ASSETS_DIR}/")
    print(f"  Source: {source}")
    print()
    print("  Next steps:")
    print("  1. Run your NeuroTopography app:")
    print("       streamlit run app.py")
    print("  2. Switch sidebar to '🖼️  อัพโหลดภาพ MRI'")
    print("  3. Upload any image from  assets/<class>/")
    print()
    print("  Recommended TDA settings for these images:")
    print("    Pixel Threshold : 100 – 140")
    print("    Gaussian σ      : 1.0 – 1.5")
    print("    Max Points      : 200 – 350")
    print("    ε (Epsilon)     : 0.3 – 0.6")


def main():
    # Declare globals FIRST — before any reference to these names anywhere
    # in this function.  Python's scoping rule: if 'global X' appears
    # anywhere in a function, every reference to X in that function is
    # treated as the global, including default-argument expressions.
    # Placing the declaration here eliminates the SyntaxError.
    global ASSETS_DIR, SAMPLES_PER_CLASS

    # Snapshot the module-level defaults BEFORE argparse overwrites them,
    # so we can use them safely as help-string values.
    _default_n   = SAMPLES_PER_CLASS
    _default_dir = str(ASSETS_DIR)

    parser = argparse.ArgumentParser(
        description="NeuroTopography — MRI Asset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_mri_assets.py                  # auto-detect best source
  python download_mri_assets.py --source kaggle  # Kaggle API only
  python download_mri_assets.py --source figshare
  python download_mri_assets.py --source synthetic   # no internet needed
  python download_mri_assets.py --n 5            # 5 images per class
        """,
    )
    parser.add_argument(
        "--source",
        choices=["auto", "kaggle", "figshare", "mendeley", "synthetic"],
        default="auto",
        help="Data source to use (default: auto)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=_default_n,
        help=f"Images per class to download (default: {_default_n})",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip TDA smoke test after download",
    )
    parser.add_argument(
        "--output-dir",
        default=_default_dir,
        help=f"Output directory (default: {_default_dir})",
    )
    args = parser.parse_args()

    # Now update the globals with user-supplied values
    ASSETS_DIR        = Path(args.output_dir)
    SAMPLES_PER_CLASS = args.n

    # Create directories
    for cls in CLASSES:
        ensure_dir(ASSETS_DIR / cls)

    print("\n  NeuroTopography — MRI Asset Downloader v1.0")
    print(f"  Output directory : {ASSETS_DIR.resolve()}")
    print(f"  Images per class : {SAMPLES_PER_CLASS}")
    print(f"  Target size      : {TARGET_SIZE[0]}×{TARGET_SIZE[1]} px  (grayscale)")

    collected = {cls: [] for cls in CLASSES}
    used_source = args.source

    # ── Strategy: fill missing classes ────────────────────────────
    def all_done():
        return all(len(collected[c]) >= SAMPLES_PER_CLASS for c in CLASSES)

    if args.source in ("auto", "kaggle") and not all_done():
        result = download_kaggle(SAMPLES_PER_CLASS)
        for cls in CLASSES:
            collected[cls].extend(result.get(cls, []))
        if result:
            used_source = "kaggle"

    if args.source in ("auto", "figshare") and not all_done():
        result = download_figshare(SAMPLES_PER_CLASS)
        for cls in CLASSES:
            needed = SAMPLES_PER_CLASS - len(collected[cls])
            collected[cls].extend(result.get(cls, [])[:needed])
        if result and not all_done():
            # Figshare has no normal class — fill it from synthetic
            if len(collected["normal"]) < SAMPLES_PER_CLASS:
                print_header("Supplementing 'normal' class with synthetic images")
                missing = SAMPLES_PER_CLASS - len(collected["normal"])
                for i in range(1, missing + 1):
                    img   = generate_synthetic_mri("normal", i)
                    saved = save_sample(img, "normal", i)
                    collected["normal"].append(str(saved))

    if args.source in ("auto", "mendeley") and not all_done():
        result = download_mendeley(SAMPLES_PER_CLASS)
        for cls in CLASSES:
            needed = SAMPLES_PER_CLASS - len(collected[cls])
            collected[cls].extend(result.get(cls, [])[:needed])

    # ── Final fallback: synthetic ──────────────────────────────────
    if args.source == "synthetic" or not all_done():
        if args.source == "synthetic":
            for cls in CLASSES:
                # Clear existing to regenerate cleanly
                for cls_dir in (ASSETS_DIR / cls).glob("*.png"):
                    cls_dir.unlink(missing_ok=True)
            collected = {cls: [] for cls in CLASSES}
        result = generate_synthetic_assets(SAMPLES_PER_CLASS)
        for cls in CLASSES:
            needed = SAMPLES_PER_CLASS - len(collected[cls])
            collected[cls].extend(result.get(cls, [])[:needed])
        if args.source == "synthetic":
            used_source = "synthetic"
        else:
            used_source = "mixed (real + synthetic fallback)"

    # ── Validate & report ──────────────────────────────────────────
    validation = validate_assets()
    write_dataset_info(used_source, collected, validation)

    # ── TDA smoke test ─────────────────────────────────────────────
    if not args.skip_test:
        tda_smoke_test()

    print_final_summary(collected, used_source)


if __name__ == "__main__":
    random.seed(42)
    main()
