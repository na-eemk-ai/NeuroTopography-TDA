"""
NeuroTopography v5.0
Advanced TDA for Precision Neuro-Oncology
─────────────────────────────────────────
Design direction: Clean Hospital White — clinical minimalism.
Pure white surfaces, charcoal typography, one surgical accent colour
(deep navy-blue #1a3a5c). No decoration for decoration's sake.
Every pixel earns its place.

Aesthetic references: Radiology PACS workstations, Nature Medicine figures,
high-end clinical software (Brainlab, Philips IntelliSpace).
"""

# ══════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════
import io
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import streamlit as st

warnings.filterwarnings("ignore")

# ── TDA backend detection ─────────────────────────────────────────
TDA_BACKEND = None
try:
    from gtda.homology import VietorisRipsPersistence
    TDA_BACKEND = "giotto"
except ImportError:
    pass

if not TDA_BACKEND:
    try:
        import gudhi  # noqa: F401
        TDA_BACKEND = "gudhi"
    except ImportError:
        TDA_BACKEND = None

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroTopography | Neuro-Oncology TDA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Clean Hospital White
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&family=Noto+Sans+Thai:wght@300;400;500;600&display=swap');

/* ── Design tokens ── */
:root {
  /* Surfaces */
  --white:      #ffffff;
  --offwhite:   #f7f8fa;
  --surface:    #f0f2f5;
  --panel:      #e8ecf0;

  /* Borders */
  --border:     #d8dde5;
  --border-hi:  #b8c2ce;

  /* Typography */
  --ink:        #0d1117;       /* near-black — headings */
  --body:       #2c3e50;       /* dark navy — body text */
  --secondary:  #546e7a;       /* medium slate — secondary */
  --muted:      #8fa5b4;       /* light slate — hints */
  --disabled:   #b8c9d4;

  /* Accent — surgical navy */
  --navy:       #1a3a5c;
  --navy-mid:   #2c5f8a;
  --navy-light: #4a8ab5;
  --navy-pale:  #e8f1f8;
  --navy-glow:  rgba(26,58,92,0.08);

  /* Data colours */
  --h0-color:   #0d6e8a;       /* teal — H₀ connected components */
  --h1-color:   #b45309;       /* amber — H₁ holes/loops */
  --normal-c:   #15803d;       /* green — normal tissue */
  --anomaly-c:  #b91c1c;       /* red — anomaly */
  --grade4-c:   #9b1d20;
  --grade2-c:   #92400e;
  --mening-c:   #1e40af;
  --meta-c:     #5b21b6;

  /* Typography */
  --font-main:  'DM Sans', 'Noto Sans Thai', sans-serif;
  --font-thai:  'Noto Sans Thai', 'DM Sans', sans-serif;
  --font-mono:  'DM Mono', monospace;
}

/* ── Reset & base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--white) !important;
  color: var(--body) !important;
  font-family: var(--font-main) !important;
  font-size: 14px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--offwhite) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  font-family: var(--font-main) !important;
  color: var(--body) !important;
}
[data-testid="stSidebar"] .stMarkdown p {
  color: var(--secondary) !important;
  font-size: 0.8rem !important;
}
[data-testid="stSidebar"] h3 {
  color: var(--navy) !important;
  font-size: 0.68rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  margin: 16px 0 6px !important;
}

/* ── Selectbox labels — high contrast black ── */
[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
[data-testid="stSidebar"] [data-testid="stSelectbox"] span,
[data-testid="stSidebar"] .stSelectbox label {
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: 0.85rem !important;
}
/* Selectbox selected value text */
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
  color: var(--ink) !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
}

/* ── Radio labels — force dark ── */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] span {
  color: var(--ink) !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
}

/* ── Slider labels ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] label span {
  color: var(--ink) !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
}
[data-testid="stSlider"] > div > div > div {
  background: var(--navy) !important;
}
/* Slider value display */
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
  color: var(--navy) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.80rem !important;
  font-weight: 500 !important;
}

/* ── Typography ── */
h1 {
  font-family: var(--font-main) !important;
  font-weight: 700 !important;
  font-size: 1.55rem !important;
  color: var(--ink) !important;
  letter-spacing: -0.02em !important;
  line-height: 1.2 !important;
}
h2 {
  font-family: var(--font-main) !important;
  font-weight: 600 !important;
  font-size: 1.05rem !important;
  color: var(--ink) !important;
}
h3 {
  font-family: var(--font-main) !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  color: var(--secondary) !important;
  letter-spacing: 0.02em !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--white) !important;
  border: 1px solid var(--border) !important;
  border-top: 3px solid var(--navy) !important;
  border-radius: 6px !important;
  padding: 14px 16px !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--font-mono) !important;
  color: var(--navy) !important;
  font-size: 2.1rem !important;
  font-weight: 500 !important;
}
[data-testid="stMetricLabel"] {
  color: var(--secondary) !important;
  font-size: 0.70rem !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--font-mono) !important;
  font-size: 0.76rem !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] {
  border-bottom: 2px solid var(--border) !important;
}
[data-testid="stTabs"] button {
  font-family: var(--font-main) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  padding: 10px 18px !important;
}
[data-testid="stTabs"] button:hover {
  color: var(--secondary) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--navy) !important;
  border-bottom: 2px solid var(--navy) !important;
  font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--navy) !important;
  border: none !important;
  color: var(--white) !important;
  font-family: var(--font-main) !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  border-radius: 6px !important;
  padding: 10px 22px !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  background: var(--navy-mid) !important;
  box-shadow: 0 4px 12px rgba(26,58,92,0.25) !important;
  transform: translateY(-1px) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--offwhite) !important;
  border: 1.5px dashed var(--border-hi) !important;
  border-radius: 8px !important;
}
[data-testid="stFileUploader"] * {
  color: var(--secondary) !important;
  font-family: var(--font-main) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
  background: var(--white) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--font-main) !important;
  font-size: 0.9rem !important;
  font-weight: 600 !important;
  color: var(--ink) !important;
}
[data-testid="stExpanderDetails"] {
  background: var(--white) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
  background: var(--offwhite) !important;
  border-radius: 6px !important;
  font-family: var(--font-main) !important;
  color: var(--body) !important;
}

/* ── Clinical card components ── */
.nt-card {
  background: var(--white);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
  margin: 10px 0;
  font-family: var(--font-thai);
  font-size: 0.87rem;
  line-height: 1.8;
  color: var(--body);
}
.nt-card.navy   { border-left: 3px solid var(--navy);     }
.nt-card.amber  { border-left: 3px solid var(--h1-color); }
.nt-card.teal   { border-left: 3px solid var(--h0-color); }
.nt-card.green  { border-left: 3px solid var(--normal-c); }
.nt-card.red    { border-left: 3px solid var(--anomaly-c);}
.nt-card.lav    { border-left: 3px solid #7c3aed;         }
.nt-card.grey   {
  background: var(--offwhite);
  border: 1px solid var(--border);
  border-left: 3px solid var(--border-hi);
}

/* ── Status chips ── */
.nt-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-family: var(--font-mono);
  font-size: 0.62rem;
  font-weight: 500;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  padding: 3px 9px;
  border-radius: 20px;
  border: 1px solid;
}
.chip-navy   { color:var(--navy);      border-color:var(--navy-mid);   background:var(--navy-pale);      }
.chip-ok     { color:#15803d;          border-color:#bbf7d0;            background:#f0fdf4;               }
.chip-warn   { color:#b45309;          border-color:#fde68a;            background:#fffbeb;               }
.chip-alert  { color:#b91c1c;          border-color:#fecaca;            background:#fef2f2;               }
.chip-info   { color:#1e40af;          border-color:#bfdbfe;            background:#eff6ff;               }
.chip-purple { color:#5b21b6;          border-color:#ddd6fe;            background:#f5f3ff;               }

/* ── Section rule ── */
.nt-rule {
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 24px 0 14px;
}
.nt-rule-line { flex:1; height:1px; background:var(--border); }
.nt-rule-text {
  font-family: var(--font-mono);
  font-size: 0.60rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  white-space: nowrap;
}

/* ── Formula blocks ── */
.nt-formula {
  background: var(--offwhite);
  border: 1px solid var(--border);
  border-left: 3px solid #7c3aed;
  border-radius: 5px;
  padding: 12px 18px;
  font-family: var(--font-mono);
  font-size: 0.76rem;
  color: var(--navy);
  margin: 8px 0;
  line-height: 2.0;
}

/* ── MRI image frame ── */
.mri-frame {
  background: #000;
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
  position: relative;
}
.mri-label {
  background: rgba(0,0,0,0.72);
  color: #fff;
  font-family: var(--font-mono);
  font-size: 0.62rem;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  padding: 4px 10px;
  position: absolute;
  top: 8px;
  left: 8px;
  border-radius: 3px;
}

/* ── Sidebar tooltip text ── */
.sb-tooltip {
  display: block;
  font-family: var(--font-thai);
  font-size: 0.73rem;
  color: var(--muted) !important;
  line-height: 1.5;
  margin-top: -4px;
  margin-bottom: 8px;
  padding-left: 2px;
}

/* ── Case info card (dashboard) ── */
.case-badge {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 0.58rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 2px 8px;
  border-radius: 3px;
  margin-bottom: 6px;
}
.badge-a { background:#e8f1f8; color:var(--navy); }
.badge-b { background:#fff7ed; color:#92400e; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--offwhite); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }

/* ── dividers ── */
hr { border-color: var(--border) !important; }

/* ── Selectbox dropdown items ── */
[data-baseweb="menu"] * {
  color: var(--ink) !important;
  font-family: var(--font-main) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib white clinical theme ─────────────────────────────
_BG   = "#ffffff"
_SURF = "#f7f8fa"
_GRID = "#e0e4ea"
_H0   = "#0d6e8a"
_H1   = "#b45309"
_NAV  = "#1a3a5c"
_GRN  = "#15803d"
_RED  = "#b91c1c"
_INK  = "#0d1117"
_MUT  = "#8fa5b4"

plt.rcParams.update({
    "figure.facecolor":  _BG,
    "axes.facecolor":    _SURF,
    "axes.edgecolor":    _GRID,
    "axes.labelcolor":   "#546e7a",
    "xtick.color":       "#8fa5b4",
    "ytick.color":       "#8fa5b4",
    "text.color":        _INK,
    "grid.color":        _GRID,
    "grid.linestyle":    ":",
    "grid.alpha":        0.7,
    "font.family":       "sans-serif",
    "font.size":         8.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlepad":     10,
    "axes.titlesize":    9.5,
    "axes.titlecolor":   _INK,
    "figure.dpi":        100,
})


# ══════════════════════════════════════════════════════════════════
# CASE CATALOGUE
# ══════════════════════════════════════════════════════════════════
CASES = {
    "NormalTissue": {
        "label":    "Normal Cerebral Tissue",
        "thai":     "เนื้อเยื่อสมองปกติ",
        "who":      "—",
        "chip":     "chip-ok",
        "badge_a":  "badge-a",
        "pl_color": _GRN,
        "desc":     "เนื้อเยื่อปกติ — วงกลมสมบูรณ์ β₀=1, β₁=1 ไม่มีโพรงภายใน",
        "mri_fn":   "normal",
    },
    "Meningioma": {
        "label":    "Meningioma  (WHO Grade I)",
        "thai":     "เนื้องอกเยื่อหุ้มสมอง  WHO Grade I",
        "who":      "Grade I",
        "chip":     "chip-info",
        "badge_a":  "badge-a",
        "pl_color": "#1e40af",
        "desc":     "เนื้องอกขอบเขตชัดเจน มีขอบเรียบ β₁ ต่ำ-ปานกลาง",
        "mri_fn":   "meningioma",
    },
    "LGG_Grade2": {
        "label":    "Low-Grade Glioma  (WHO Grade II)",
        "thai":     "กลิโอมาระดับต่ำ  WHO Grade II",
        "who":      "Grade II",
        "chip":     "chip-warn",
        "badge_a":  "badge-a",
        "pl_color": _H1,
        "desc":     "เนื้องอกเติบโตช้า มีโพรงภายในน้อย β₁ ปานกลาง",
        "mri_fn":   "lgg",
    },
    "GBM_Grade4": {
        "label":    "Glioblastoma Multiforme  (WHO Grade IV)",
        "thai":     "กลิโอบลาสโตมา มัลติฟอร์มี  WHO Grade IV",
        "who":      "Grade IV",
        "chip":     "chip-alert",
        "badge_a":  "badge-b",
        "pl_color": _RED,
        "desc":     "เนื้องอกระดับสูง โครงสร้างซับซ้อน มีโพรงเนื้อตายหลายแห่ง β₁ สูงมาก",
        "mri_fn":   "gbm",
    },
    "Metastasis": {
        "label":    "Brain Metastasis  (Multiple)",
        "thai":     "เนื้องอกแพร่กระจายสมอง (หลายก้อน)",
        "who":      "—",
        "chip":     "chip-purple",
        "badge_a":  "badge-b",
        "pl_color": "#5b21b6",
        "desc":     "ก้อนเนื้องอกหลายตำแหน่งแยกกัน β₀ สูงมาก บ่งชี้การแพร่กระจาย",
        "mri_fn":   "meta",
    },
}


# ══════════════════════════════════════════════════════════════════
# SYNTHETIC MRI IMAGE GENERATOR
# Generates realistic-looking MRI thumbnails programmatically
# so we don't depend on external files.
# ══════════════════════════════════════════════════════════════════

def _make_base_mri(size=256, rng=None):
    """Create a dark, grainy brain MRI background."""
    if rng is None:
        rng = np.random.default_rng(0)
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    r = size * 0.42
    Y, X = np.ogrid[:size, :size]
    skull_mask = ((X - cx)**2 + (Y - cy)**2) <= r**2
    inner_r = r * 0.88
    brain_mask = ((X - cx)**2 + (Y - cy)**2) <= inner_r**2
    img[skull_mask]  = 0.18 + rng.normal(0, 0.02, skull_mask.sum())
    img[brain_mask]  = 0.28 + rng.normal(0, 0.04, brain_mask.sum())
    img = gaussian_filter(img, sigma=2.0)
    img = np.clip(img, 0, 1)
    return img, cx, cy, size


def _as_pil_mri(arr):
    """Convert float [0,1] array to 8-bit grayscale PIL."""
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB")


def generate_mri_thumbnail(case_id: str, size: int = 256) -> Image.Image:
    """
    Generate a synthetic MRI-style PNG for a given case.
    Returns a PIL RGB image with PACS-style overlay text.
    """
    rng = np.random.default_rng(hash(case_id) % 2**31)
    img, cx, cy, s = _make_base_mri(size, rng)
    Y, X = np.ogrid[:s, :s]

    if case_id == "NormalTissue":
        # Uniform, symmetric — no tumor
        img = gaussian_filter(img, sigma=1.5)

    elif case_id == "Meningioma":
        # Well-defined, bright, homogeneous mass on the periphery
        mx, my = cx + int(s * 0.25), cy - int(s * 0.18)
        mr = s * 0.11
        m = ((X - mx)**2 + (Y - my)**2) <= mr**2
        img[m] = 0.85 + rng.normal(0, 0.03, m.sum())
        img = gaussian_filter(img, sigma=1.2)

    elif case_id == "LGG_Grade2":
        # Subtle infiltrating mass, moderate brightness
        mx, my = cx - int(s * 0.08), cy + int(s * 0.05)
        mr = s * 0.18
        m = ((X - mx)**2 + (Y - my)**2) <= mr**2
        img[m] += 0.28 + rng.normal(0, 0.04, m.sum())
        # Small cavity
        cr = s * 0.06
        cm = ((X - mx)**2 + (Y - my)**2) <= cr**2
        img[cm] = 0.05 + rng.normal(0, 0.01, cm.sum())
        img = gaussian_filter(img, sigma=1.5)

    elif case_id == "GBM_Grade4":
        # Heterogeneous, ring-enhancing, central necrosis
        mx, my = cx + int(s * 0.05), cy - int(s * 0.04)
        # Outer ring (enhancing)
        ro = s * 0.22
        mo = ((X - mx)**2 + (Y - my)**2) <= ro**2
        img[mo] += 0.40 + rng.normal(0, 0.06, mo.sum())
        # Necrotic core (dark)
        ri = s * 0.12
        mi = ((X - mx)**2 + (Y - my)**2) <= ri**2
        img[mi]  = 0.04 + rng.normal(0, 0.02, mi.sum())
        # Second smaller satellite
        sx2, sy2 = cx - int(s * 0.22), cy + int(s * 0.15)
        rs2 = s * 0.08
        ms2 = ((X - sx2)**2 + (Y - sy2)**2) <= rs2**2
        img[ms2] += 0.32 + rng.normal(0, 0.04, ms2.sum())
        img = gaussian_filter(img, sigma=0.9)

    elif case_id == "Metastasis":
        # Multiple small bright lesions
        centres = [
            (cx + int(s*0.22), cy - int(s*0.10), s*0.065),
            (cx - int(s*0.25), cy + int(s*0.18), s*0.055),
            (cx + int(s*0.08), cy + int(s*0.28), s*0.050),
            (cx - int(s*0.12), cy - int(s*0.24), s*0.045),
        ]
        for mx, my, mr in centres:
            m = ((X - mx)**2 + (Y - my)**2) <= mr**2
            img[m] = 0.78 + rng.normal(0, 0.04, m.sum())
        img = gaussian_filter(img, sigma=1.0)

    img = np.clip(img, 0, 1)
    pil = _as_pil_mri(img)

    # PACS-style overlay
    draw = ImageDraw.Draw(pil)
    cinfo = CASES.get(case_id, {})
    overlay_lines = [
        f"NeuroTopography v5.0",
        f"{cinfo.get('label', case_id)}",
        f"WHO {cinfo.get('who', '—')}",
        "AXIAL  T1+C",
        f"{size}×{size} px",
    ]
    y_pos = 6
    for line in overlay_lines:
        draw.text((8, y_pos), line, fill=(180, 220, 255))
        y_pos += 14

    # Corner cross-hair
    cx2, cy2 = size // 2, size // 2
    draw.line([(cx2-14, cy2), (cx2-4, cy2)], fill=(80, 160, 80), width=1)
    draw.line([(cx2+4, cy2), (cx2+14, cy2)], fill=(80, 160, 80), width=1)
    draw.line([(cx2, cy2-14), (cx2, cy2-4)], fill=(80, 160, 80), width=1)
    draw.line([(cx2, cy2+4), (cx2, cy2+14)], fill=(80, 160, 80), width=1)

    return pil


# ══════════════════════════════════════════════════════════════════
# POINT CLOUD PIPELINE
# ══════════════════════════════════════════════════════════════════

def mri_to_cloud(img: np.ndarray, threshold: int,
                  max_pts: int = 350, sigma: float = 1.3) -> np.ndarray:
    smooth     = gaussian_filter(img.astype(np.float64), sigma=sigma)
    rows, cols = np.where(smooth > threshold)
    if len(rows) < 3:
        return np.random.default_rng(0).standard_normal((10, 2)) * 0.1
    pts = np.column_stack([cols.astype(np.float64), -rows.astype(np.float64)])
    for d in range(2):
        lo, hi = pts[:, d].min(), pts[:, d].max()
        if hi > lo:
            pts[:, d] = 2.0 * (pts[:, d] - lo) / (hi - lo) - 1.0
    if len(pts) > max_pts:
        pts = pts[np.random.default_rng(42).choice(len(pts), max_pts, replace=False)]
    return pts


def pipeline_views(img: np.ndarray, threshold: int, sigma: float = 1.3):
    smooth  = gaussian_filter(img.astype(np.float64), sigma=sigma)
    binary  = (smooth > threshold).astype(np.float64)
    heatmap = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9)
    return img, smooth, binary, heatmap


# ══════════════════════════════════════════════════════════════════
# BATCH CASE POINT CLOUD GENERATORS
# ══════════════════════════════════════════════════════════════════

def generate_batch_case(case_id: str, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    def ring(cx, cy, r, n, noise=0.03, t0=0, t1=2*np.pi):
        t  = np.linspace(t0, t1, n, endpoint=False)
        rx = r + rng.normal(0, noise, n)
        return np.column_stack([cx + rx*np.cos(t), cy + rx*np.sin(t)])

    if case_id == "NormalTissue":
        return ring(0, 0, 1.0, 180, noise=0.025)

    elif case_id == "Meningioma":
        outer = ring(0, 0, 1.0, 160, noise=0.06)
        bumps = np.vstack([ring(0, 0, 1.08+0.04*np.sin(k), 18, noise=0.02)
                           for k in range(0, 12, 2)])
        return np.vstack([outer, bumps])

    elif case_id == "LGG_Grade2":
        return np.vstack([ring(0, 0, 1.0, 180, noise=0.07),
                          ring(0, 0, 0.38, 55, noise=0.04)])

    elif case_id == "GBM_Grade4":
        outer  = ring(0, 0, 1.0, 200, noise=0.13)
        outer += rng.normal(0, 0.025, outer.shape)
        cav1   = ring(0.1, 0.1, 0.40, 60, noise=0.05)
        cav2   = ring(-0.15, -0.2, 0.22, 35, noise=0.04)
        sat1   = ring(1.82, 0.55, 0.20, 42, noise=0.03)
        sat2   = ring(-1.65, -0.60, 0.18, 38, noise=0.03)
        finger = ring(0, 0, 0.70, 30, noise=0.04, t0=np.pi/4, t1=3*np.pi/4)
        return np.vstack([outer, cav1, cav2, sat1, sat2, finger])

    elif case_id == "Metastasis":
        centres = [(0,0,0.30), (1.8,0.5,0.22), (-1.7,-0.4,0.20),
                   (0.3,-1.6,0.18), (-0.5,1.7,0.16)]
        return np.vstack([ring(cx, cy, r, max(25, int(r*150)), noise=0.03)
                           for cx, cy, r in centres])

    return ring(0, 0, 1.0, 150, noise=0.03)


# ══════════════════════════════════════════════════════════════════
# TDA ENGINE
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, max_entries=16)
def cached_persistence(pts_bytes: bytes, max_edge: float):
    pts = np.frombuffer(pts_bytes, dtype=np.float64).reshape(-1, 2)
    return _run_persistence(pts, max_edge)


def _run_persistence(pts, max_edge=1.8):
    if TDA_BACKEND == "giotto":
        return _p_giotto(pts, max_edge)
    elif TDA_BACKEND == "gudhi":
        return _p_gudhi(pts, max_edge)
    return None


def _p_giotto(pts, max_edge):
    try:
        X   = pts[np.newaxis, :, :]
        vr  = VietorisRipsPersistence(
            homology_dimensions=[0, 1],
            max_edge_length=max_edge,
            collapse_edges=True, n_jobs=-1)
        dgm = vr.fit_transform(X)[0]
        return {"backend": "giotto", "pairs": dgm, "max_edge": max_edge}
    except Exception as e:
        st.warning(f"Giotto error: {e}")
        return _p_gudhi(pts, max_edge)


def _p_gudhi(pts, max_edge):
    import gudhi
    rips   = gudhi.RipsComplex(points=pts, max_edge_length=max_edge)
    st_obj = rips.create_simplex_tree(max_dimension=2)
    st_obj.compute_persistence()
    cap    = max_edge * 1.15
    raw    = st_obj.persistence()
    finite = [(b, min(d, cap), dim) for dim, (b, d) in raw if d != float("inf")]
    inf_p  = [(b, cap, dim)         for dim, (b, d) in raw if d == float("inf")]
    pairs  = np.array(finite + inf_p, dtype=float) if (finite or inf_p) else np.zeros((0, 3))
    return {"backend": "gudhi", "pairs": pairs, "max_edge": max_edge}


def _dim_pairs(result, dim):
    if result is None or len(result["pairs"]) == 0:
        return []
    p = result["pairs"]
    return [(float(r[0]), float(r[1])) for r in p[p[:, 2] == dim]]


def betti_at(result, eps):
    if not result:
        return 0, 0
    b0 = sum(1 for b, d in _dim_pairs(result, 0) if b <= eps < d)
    b1 = sum(1 for b, d in _dim_pairs(result, 1) if b <= eps < d)
    return b0, b1


def topo_stats(result, dim=1):
    pairs = _dim_pairs(result, dim)
    if not pairs:
        return dict(n=0, max_p=0., mean_p=0., entropy=0., lifetimes=np.array([]))
    lt = np.array([d - b for b, d in pairs if d > b])
    if not len(lt):
        return dict(n=0, max_p=0., mean_p=0., entropy=0., lifetimes=np.array([]))
    p = lt / (lt.sum() + 1e-12)
    return dict(n=len(lt), max_p=float(lt.max()), mean_p=float(lt.mean()),
                entropy=float(-np.sum(p * np.log(p + 1e-12))), lifetimes=lt)


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION FACTORY
# ══════════════════════════════════════════════════════════════════

def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def fig_simplicial(pts, eps, title="", color=_NAV, wh=(5.2, 5.2)):
    fig, ax = plt.subplots(figsize=wh)
    ax.set_facecolor(_SURF)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35)
    D = cdist(pts, pts)
    n = len(pts)
    for pt in pts:
        ax.add_patch(Circle(pt, eps/2, color=color, alpha=0.07, zorder=1))
    segs = [[pts[i], pts[j]] for i in range(n) for j in range(i+1,n) if D[i,j]<=eps]
    if segs:
        ax.add_collection(LineCollection(segs, color=color, alpha=0.30, lw=0.8, zorder=2))
    for i in range(n):
        for j in range(i+1, n):
            if D[i,j] > eps: continue
            for k in range(j+1, n):
                if D[i,k] <= eps and D[j,k] <= eps:
                    ax.add_patch(plt.Polygon([pts[i],pts[j],pts[k]],
                                              color=color, alpha=0.09, zorder=2))
    ax.scatter(pts[:,0], pts[:,1], s=20, color=color, zorder=6,
               edgecolors=_BG, linewidths=0.35, alpha=0.9)
    pad = 0.35
    ax.set_xlim(pts[:,0].min()-pad, pts[:,0].max()+pad)
    ax.set_ylim(pts[:,1].min()-pad, pts[:,1].max()+pad)
    ax.set_title(f"{title}  |  ε = {eps:.3f}", color=_INK, fontsize=9.5)
    fig.tight_layout()
    return _png(fig)


def fig_filtration_strip(pts, color=_NAV, eps_list=(0.12,0.42,0.82,1.38)):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    D = cdist(pts, pts); n = len(pts)
    for ax, eps in zip(axes, eps_list):
        ax.set_facecolor(_SURF); ax.set_aspect("equal")
        for pt in pts:
            ax.add_patch(Circle(pt, eps/2, color=color, alpha=0.07))
        segs = [[pts[i],pts[j]] for i in range(n) for j in range(i+1,n) if D[i,j]<=eps]
        if segs:
            ax.add_collection(LineCollection(segs, color=color, alpha=0.30, lw=0.85))
        for i in range(n):
            for j in range(i+1,n):
                if D[i,j]>eps: continue
                for k in range(j+1,n):
                    if D[i,k]<=eps and D[j,k]<=eps:
                        ax.add_patch(plt.Polygon([pts[i],pts[j],pts[k]],
                                                  color=color, alpha=0.08))
        ax.scatter(pts[:,0], pts[:,1], s=14, color=color, zorder=5,
                   edgecolors=_BG, linewidths=0.28)
        ax.set_title(f"ε = {eps:.2f}", color=_INK, fontsize=10)
        ax.set_xlim(pts[:,0].min()-0.28, pts[:,0].max()+0.28)
        ax.set_ylim(pts[:,1].min()-0.28, pts[:,1].max()+0.28)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Filtration Process — Simplicial Complex Growth",
                 color=_INK, fontsize=10, y=1.02)
    fig.tight_layout()
    return _png(fig)


def fig_barcode(result, title="Persistence Barcode"):
    if result is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "ไม่พบข้อมูล", ha="center", va="center",
                color=_RED, fontsize=11)
        return _png(fig)
    d0 = _dim_pairs(result, 0); d1 = _dim_pairs(result, 1)
    total = len(d0) + len(d1)
    if total == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "ไม่พบ persistence features",
                ha="center", va="center", color=_H1)
        return _png(fig)
    h = max(3.5, total*0.30+1.8)
    fig, ax = plt.subplots(figsize=(9, h))
    ax.set_facecolor(_SURF); ax.grid(axis="x", alpha=0.35)
    y = 0
    for i,(b,d) in enumerate(sorted(d0,key=lambda x:x[1]-x[0],reverse=True)):
        ax.barh(y, d-b, left=b, height=0.60, color=_H0, alpha=0.80, zorder=3,
                label=f"H₀  Components (n={len(d0)})" if i==0 else "_nl")
        y += 1
    for i,(b,d) in enumerate(sorted(d1,key=lambda x:x[1]-x[0],reverse=True)):
        ax.barh(y, d-b, left=b, height=0.60, color=_H1, alpha=0.82, zorder=3,
                label=f"H₁  Holes / Loops (n={len(d1)})" if i==0 else "_nl")
        y += 1
    ax.axvline(0, color=_GRID, lw=1)
    ax.set_xlabel("Filtration parameter  ε", labelpad=7)
    ax.set_yticks([]); ax.set_title(title, color=_INK, fontsize=9.5)
    ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    return _png(fig)


def fig_persistence_diagram(result, title="Persistence Diagram"):
    if result is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.text(0.5, 0.5, "ไม่พบข้อมูล", ha="center", va="center", color=_RED)
        return _png(fig)
    d0 = [(b,d) for b,d in _dim_pairs(result,0) if d<float("inf")]
    d1 = [(b,d) for b,d in _dim_pairs(result,1) if d<float("inf")]
    all_v = [v for p in d0+d1 for v in p]
    mv = (max(all_v) if all_v else 1.5)*1.12
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.set_facecolor(_SURF)
    ax.plot([0,mv],[0,mv], color=_GRID, lw=1.4, zorder=1)
    ax.fill_between([0,mv],[0,mv], alpha=0.04, color=_NAV)
    if d0:
        bv,dv = zip(*d0)
        ax.scatter(bv,dv,s=50,color=_H0,alpha=0.88,zorder=5,
                   edgecolors=_BG,lw=0.4,label=f"H₀  n={len(d0)}")
    if d1:
        bv,dv = zip(*d1)
        ax.scatter(bv,dv,s=65,color=_H1,alpha=0.88,zorder=5,
                   edgecolors=_BG,lw=0.4,marker="D",label=f"H₁  n={len(d1)}")
    ax.set_xlabel("Birth  ε",labelpad=7); ax.set_ylabel("Death  ε",labelpad=7)
    ax.set_xlim(0,mv); ax.set_ylim(0,mv); ax.set_aspect("equal")
    ax.set_title(title, color=_INK, fontsize=9.5)
    ax.legend(facecolor=_BG,edgecolor=_GRID,fontsize=8.5)
    ax.grid(True,alpha=0.3)
    fig.tight_layout()
    return _png(fig)


def fig_betti_curves(result, title="Betti Curves"):
    if result is None:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5,0.5,"ไม่พบข้อมูล",ha="center",va="center",color=_RED)
        return _png(fig)
    eps_r = np.linspace(0, result["max_edge"], 300)
    b0s,b1s = [],[]
    for e in eps_r:
        b0,b1 = betti_at(result, e)
        b0s.append(b0); b1s.append(b1)
    fig, ax = plt.subplots(figsize=(8.5,4.2))
    ax.set_facecolor(_SURF)
    ax.step(eps_r,b0s,where="post",color=_H0,lw=2.0,label="β₀  Components",zorder=3)
    ax.step(eps_r,b1s,where="post",color=_H1,lw=2.0,label="β₁  Holes",zorder=3)
    ax.fill_between(eps_r,b0s,step="post",alpha=0.07,color=_H0)
    ax.fill_between(eps_r,b1s,step="post",alpha=0.07,color=_H1)
    ax.set_xlabel("Filtration parameter  ε",labelpad=7)
    ax.set_ylabel("Betti Number",labelpad=7)
    ax.set_title(title,color=_INK,fontsize=9.5)
    ax.legend(facecolor=_BG,edgecolor=_GRID,fontsize=9)
    ax.grid(True,alpha=0.3)
    fig.tight_layout()
    return _png(fig)


def fig_pipeline(img, threshold, sigma):
    orig,smooth,binary,hmap = pipeline_views(img, threshold, sigma)
    panels = [(orig,"gray",f"(1) Grayscale Input"),
              (smooth,"gray",f"(2) Gaussian Smoothing  σ={sigma:.1f}"),
              (binary,"gray",f"(3) Binary Threshold  t={threshold}"),
              (hmap,"inferno","(4) Intensity Heatmap")]
    fig, axes = plt.subplots(1,4,figsize=(15,4.0))
    for ax,(data,cmap,lbl) in zip(axes,panels):
        im = ax.imshow(data,cmap=cmap,aspect="auto")
        ax.set_title(lbl,color=_INK,fontsize=8.5,pad=8)
        ax.axis("off")
        if cmap=="inferno":
            plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    fig.suptitle("ท่อประมวลผลภาพ MRI  |  Pre-processing Pipeline",
                 color=_INK,fontsize=10,y=1.02)
    fig.tight_layout()
    return _png(fig)


def fig_dual_diagnostic(r_a, r_b, la, lb, ca=_H0, cb=_H1):
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.28)

    def _bc(ax, res, color, lbl):
        ax.set_facecolor(_SURF); ax.grid(axis="x",alpha=0.30)
        d0=_dim_pairs(res,0) if res else []; d1=_dim_pairs(res,1) if res else []
        y=0
        for i,(b,d) in enumerate(sorted(d0,key=lambda x:x[1]-x[0],reverse=True)):
            ax.barh(y,d-b,left=b,height=0.60,color=_H0,alpha=0.80,zorder=3,
                    label=f"H₀  n={len(d0)}" if i==0 else "_nl"); y+=1
        for i,(b,d) in enumerate(sorted(d1,key=lambda x:x[1]-x[0],reverse=True)):
            ax.barh(y,d-b,left=b,height=0.60,color=color,alpha=0.82,zorder=3,
                    label=f"H₁  n={len(d1)}" if i==0 else "_nl"); y+=1
        ax.set_yticks([]); ax.set_xlabel("ε",labelpad=5)
        ax.set_title(f"Barcode — {lbl}",color=_INK,fontsize=9.5)
        if d0 or d1:
            ax.legend(facecolor=_BG,edgecolor=_GRID,fontsize=7.5)

    def _pd(ax, res, color, lbl):
        ax.set_facecolor(_SURF)
        d0=[(b,d) for b,d in (_dim_pairs(res,0) if res else []) if d<float("inf")]
        d1=[(b,d) for b,d in (_dim_pairs(res,1) if res else []) if d<float("inf")]
        all_v=[v for p in d0+d1 for v in p]; mv=(max(all_v) if all_v else 1.5)*1.12
        ax.plot([0,mv],[0,mv],color=_GRID,lw=1.2,zorder=1)
        if d0:
            bv,dv=zip(*d0); ax.scatter(bv,dv,s=50,color=_H0,alpha=0.88,zorder=5,
                                        edgecolors=_BG,lw=0.35,label=f"H₀ n={len(d0)}")
        if d1:
            bv,dv=zip(*d1); ax.scatter(bv,dv,s=62,color=color,alpha=0.88,zorder=5,
                                        edgecolors=_BG,lw=0.35,marker="D",label=f"H₁ n={len(d1)}")
        ax.set_xlim(0,mv); ax.set_ylim(0,mv); ax.set_aspect("equal")
        ax.set_xlabel("Birth ε",labelpad=5); ax.set_ylabel("Death ε",labelpad=5)
        ax.set_title(f"Diagram — {lbl}",color=_INK,fontsize=9.5)
        ax.grid(True,alpha=0.25)
        if d0 or d1: ax.legend(facecolor=_BG,edgecolor=_GRID,fontsize=7.5)

    _bc(fig.add_subplot(gs[0,0]), r_a, ca, la)
    _bc(fig.add_subplot(gs[0,1]), r_b, cb, lb)
    _pd(fig.add_subplot(gs[1,0]), r_a, ca, la)
    _pd(fig.add_subplot(gs[1,1]), r_b, cb, lb)
    fig.suptitle("Primary Diagnostic Outputs — Persistence Barcode & Diagram",
                 color=_INK,fontsize=11,y=1.01)
    return _png(fig)


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
_DEF = dict(result_a=None, result_b=None, result_u=None,
            pts_a=None, pts_b=None, pts_u=None,
            img_u=None, case_a="NormalTissue", case_b="GBM_Grade4",
            mode="batch", ready=False,
            mri_a=None, mri_b=None)
for k, v in _DEF.items():
    if k not in st.session_state: st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
_be_chip = {
    "giotto": '<span class="nt-chip chip-ok">giotto-tda · online</span>',
    "gudhi":  '<span class="nt-chip chip-navy">GUDHI · online</span>',
    None:     '<span class="nt-chip chip-alert">TDA lib · offline</span>',
}[TDA_BACKEND]

st.markdown(f"""
<div style="
  display:flex; align-items:flex-end;
  justify-content:space-between; flex-wrap:wrap; gap:12px;
  padding-bottom:20px; margin-bottom:26px;
  border-bottom:2px solid #1a3a5c;
">
  <div>
    <div style="
      font-family:'DM Mono',monospace;
      font-size:0.58rem;
      letter-spacing:0.20em;
      text-transform:uppercase;
      color:#8fa5b4;
      margin-bottom:8px;
    ">Medical AI Research Platform  ·  Persistent Homology Engine  ·  v5.0</div>
    <h1 style="margin:0 0 4px 0; color:#0d1117;">
      NeuroTopography
    </h1>
    <div style="
      font-family:'DM Sans',sans-serif;
      font-size:0.82rem;
      color:#546e7a;
      letter-spacing:0.06em;
    ">Advanced TDA for Precision Neuro-Oncology</div>
  </div>
  <div style="display:flex;gap:8px;align-items:center;padding-bottom:6px;">
    {_be_chip}
    <span class="nt-chip chip-navy">🧬 TDA · Persistent Homology</span>
  </div>
</div>
""", unsafe_allow_html=True)

if TDA_BACKEND is None:
    st.error("⚠️  ไม่พบ TDA library — กรุณาติดตั้ง: `pip install gudhi`")


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### โหมดการสแกน")

    mode_options = ["🔬  Batch Case Scanning", "🖼️  อัพโหลดภาพ MRI"]
    mode_sel = st.radio("เมนู", mode_options, label_visibility="collapsed", key="mode_radio")
    st.session_state.mode = "batch" if "Batch" in mode_sel else "upload"

    # Tooltip descriptions
    if "Batch" in mode_sel:
        st.markdown('<span class="sb-tooltip">โหมดเปรียบเทียบเคสตัวอย่างจากฐานข้อมูล</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="sb-tooltip">วิเคราะห์ภาพถ่าย MRI ของคุณเอง</span>',
                    unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.mode == "batch":
        st.markdown("### เลือก Case")
        opts   = list(CASES.keys())
        labels = [CASES[k]["thai"] for k in opts]

        idx_a  = st.selectbox(
            "Case A",
            range(len(opts)),
            format_func=lambda i: labels[i],
            index=opts.index("NormalTissue"),
            key="sel_a",
        )
        idx_b  = st.selectbox(
            "Case B",
            range(len(opts)),
            format_func=lambda i: labels[i],
            index=opts.index("GBM_Grade4"),
            key="sel_b",
        )
        st.session_state.case_a = opts[idx_a]
        st.session_state.case_b = opts[idx_b]

    else:
        st.markdown("### อัพโหลดภาพ")
        upload_file = st.file_uploader(
            "MRI slice (.jpg / .png)", type=["jpg","jpeg","png"],
            label_visibility="visible")

    st.markdown("---")
    st.markdown("### พารามิเตอร์การประมวลผล")

    threshold = st.slider("Pixel Threshold", 30, 230, 115, 5, key="thr")
    st.markdown('<span class="sb-tooltip">ปรับความเข้มแสงเพื่อแยกเนื้องอกออกจากเนื้อเยื่อปกติ</span>',
                unsafe_allow_html=True)

    sigma = st.slider("Gaussian  σ", 0.5, 3.0, 1.3, 0.1, key="sig")
    st.markdown('<span class="sb-tooltip">ปรับความสมูทเพื่อลดจุดรบกวนในภาพสแกน</span>',
                unsafe_allow_html=True)

    max_pts = st.slider("Max Points", 50, 700, 300, 25, key="mpts")
    st.markdown('<span class="sb-tooltip">จำนวนจุดสูงสุดใน Point Cloud — ลดเพื่อเพิ่มความเร็ว</span>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### พารามิเตอร์ Filtration")

    epsilon  = st.slider("ε  (Epsilon)",    0.05, 2.0, 0.45, 0.05, key="eps")
    st.markdown('<span class="sb-tooltip">รัศมีของ ball รอบแต่ละจุด — เพิ่มเพื่อดู complex เติบโต</span>',
                unsafe_allow_html=True)

    max_edge = st.slider("Max Edge Length", 0.5,  3.0, 1.8,  0.1,  key="me")
    st.markdown('<span class="sb-tooltip">ระยะ edge สูงสุดที่พิจารณาใน Rips filtration</span>',
                unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("▶  SCAN & ANALYSE", use_container_width=True)

    st.markdown("---")
    st.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:0.62rem;
            color:#8fa5b4;line-height:2.0;padding:4px 0;">
MRI → Smooth → Threshold<br>
→ Point Cloud → Rips<br>
→ Persistence Pairs<br>
→ Betti Numbers<br>
→ Clinical Report
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ANALYSIS EXECUTION
# ══════════════════════════════════════════════════════════════════
if run_btn:
    if st.session_state.mode == "batch":
        with st.spinner("กำลังประมวลผล Persistent Homology..."):
            ca, cb = st.session_state.case_a, st.session_state.case_b
            pts_a  = generate_batch_case(ca)
            pts_b  = generate_batch_case(cb)
            st.session_state.pts_a    = pts_a
            st.session_state.pts_b    = pts_b
            st.session_state.result_a = _run_persistence(pts_a, max_edge)
            st.session_state.result_b = _run_persistence(pts_b, max_edge)
            st.session_state.mri_a    = generate_mri_thumbnail(ca)
            st.session_state.mri_b    = generate_mri_thumbnail(cb)
            st.session_state.ready    = True
        st.sidebar.success("✅  Scan complete")

    elif st.session_state.mode == "upload":
        uf = upload_file if "upload_file" in dir() else None
        if uf is not None:
            with st.spinner("กำลังประมวลผลภาพ..."):
                img_arr = np.array(Image.open(uf).convert("L"))
                pts_u   = mri_to_cloud(img_arr, threshold, max_pts, sigma)
                st.session_state.img_u    = img_arr
                st.session_state.pts_u    = pts_u
                st.session_state.result_u = _run_persistence(pts_u, max_edge)
                st.session_state.ready    = True
            st.sidebar.success("✅  Analysis complete")
        else:
            st.sidebar.warning("⚠️  กรุณาอัพโหลดภาพก่อน")

# Shortcuts
r_a  = st.session_state.result_a
r_b  = st.session_state.result_b
r_u  = st.session_state.result_u
p_a  = st.session_state.pts_a
p_b  = st.session_state.pts_b
p_u  = st.session_state.pts_u
ca   = st.session_state.case_a
cb   = st.session_state.case_b
ready = st.session_state.ready
mri_a = st.session_state.mri_a
mri_b = st.session_state.mri_b


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
(tab_scan, tab_diag, tab_filt,
 tab_pipe, tab_theory, tab_report) = st.tabs([
    "📊  แดชบอร์ด",
    "🔬  Persistence Diagnostics",
    "🌀  Filtration Explorer",
    "🌡️  Image Pipeline",
    "📐  Theory & Diagnostics",
    "📋  Case Report",
])


# ════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard
# ════════════════════════════════════════════════════════════════
with tab_scan:

    if not ready:
        st.markdown("""
<div class="nt-card navy" style="text-align:center;padding:52px 28px;">
  <div style="font-size:2.8rem;margin-bottom:16px;">🧬</div>
  <div style="font-size:1.1rem;font-weight:700;color:#0d1117;margin-bottom:10px;">
    NeuroTopography — Advanced TDA for Precision Neuro-Oncology
  </div>
  <div style="font-size:0.88rem;color:#546e7a;line-height:1.9;">
    เลือก Case จาก Sidebar แล้วกด
    <strong style="color:#1a3a5c;">▶ SCAN &amp; ANALYSE</strong>
    เพื่อเริ่มการวิเคราะห์
  </div>
</div>""", unsafe_allow_html=True)

    else:
        if st.session_state.mode == "batch" and r_a and r_b:
            info_a = CASES[ca]
            info_b = CASES[cb]
            la = info_a["label"].split("(")[0].strip()
            lb = info_b["label"].split("(")[0].strip()

            # ── Case info cards ───────────────────────────────────
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
<div class="nt-card navy">
  <span class="case-badge badge-a">CASE A</span>
  <div style="font-size:1.0rem;font-weight:700;color:#0d1117;margin:6px 0 3px;">
    {info_a['thai']}
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#8fa5b4;
              margin-bottom:8px;">{info_a['label']}</div>
  <div style="font-size:0.84rem;color:#546e7a;">{info_a['desc']}</div>
</div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
<div class="nt-card amber">
  <span class="case-badge badge-b">CASE B</span>
  <div style="font-size:1.0rem;font-weight:700;color:#0d1117;margin:6px 0 3px;">
    {info_b['thai']}
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#8fa5b4;
              margin-bottom:8px;">{info_b['label']}</div>
  <div style="font-size:0.84rem;color:#546e7a;">{info_b['desc']}</div>
</div>""", unsafe_allow_html=True)

            # ── Betti Numbers ─────────────────────────────────────
            st.markdown(f"""
<div class="nt-rule">
  <div class="nt-rule-line"></div>
  <div class="nt-rule-text">Betti Numbers  ·  ε = {epsilon:.3f}</div>
  <div class="nt-rule-line"></div>
</div>""", unsafe_allow_html=True)

            b0_a, b1_a = betti_at(r_a, epsilon)
            b0_b, b1_b = betti_at(r_b, epsilon)
            s_a = topo_stats(r_a, 1)
            s_b = topo_stats(r_b, 1)

            st.markdown(f'<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8fa5b4;margin-bottom:6px;">▌ {la}</div>', unsafe_allow_html=True)
            col1,col2,col3,col4 = st.columns(4)
            col1.metric("β₀  Components",     b0_a)
            col2.metric("β₁  Holes / Loops",  b1_a)
            col3.metric("H₁ Max Persistence", f"{s_a['max_p']:.4f}")
            col4.metric("H₁ Entropy",         f"{s_a['entropy']:.4f}")

            st.markdown('<div style="margin:10px 0;"></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.68rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8fa5b4;margin-bottom:6px;">▌ {lb}</div>', unsafe_allow_html=True)
            col5,col6,col7,col8 = st.columns(4)
            col5.metric("β₀  Components",     b0_b, delta=f"{b0_b-b0_a:+d}")
            col6.metric("β₁  Holes / Loops",  b1_b, delta=f"{b1_b-b1_a:+d}")
            col7.metric("H₁ Max Persistence", f"{s_b['max_p']:.4f}",
                        delta=f"{s_b['max_p']-s_a['max_p']:+.4f}")
            col8.metric("H₁ Entropy",         f"{s_b['entropy']:.4f}",
                        delta=f"{s_b['entropy']-s_a['entropy']:+.4f}")

            # ── Verdict ───────────────────────────────────────────
            if b1_b > b1_a:
                vc, vt, vm = "red","chip-alert", (
                    f"<strong>Topological Anomaly:</strong>  β₁ Case B ({b1_b}) &gt; β₁ Case A ({b1_a})<br>"
                    f"H₁ entropy สูงกว่า ({s_b['entropy']:.3f} vs {s_a['entropy']:.3f}) — "
                    f"บ่งชี้ internal structural complexity สูง สอดคล้องกับ multiple necrotic voids"
                )
            elif b0_b > b0_a:
                vc, vt, vm = "amber","chip-warn", (
                    f"<strong>Fragmentation Pattern:</strong>  β₀ Case B ({b0_b}) &gt; β₀ Case A ({b0_a})<br>"
                    f"จำนวน connected components สูง — อาจบ่งชี้ metastatic pattern"
                )
            else:
                vc, vt, vm = "green","chip-ok", (
                    f"ค่า β₀ และ β₁ ทั้งสอง Case ใกล้เคียงกันที่ ε = {epsilon:.2f}<br>"
                    f"ลองเพิ่มค่า ε ในแถบควบคุมเพื่อดูความแตกต่าง"
                )
            st.markdown(f"""
<div class="nt-card {vc}" style="margin-top:12px;">
  <span class="nt-chip {vt}" style="margin-right:10px;">TOPOLOGY ANALYSIS</span>
  <span style="font-size:0.87rem;">{vm}</span>
</div>""", unsafe_allow_html=True)

            # ── MRI Images + Simplicial Complexes ─────────────────
            st.markdown(f"""
<div class="nt-rule">
  <div class="nt-rule-line"></div>
  <div class="nt-rule-text">MRI Source Image  ·  Simplicial Complex  ·  ε = {epsilon:.3f}</div>
  <div class="nt-rule-line"></div>
</div>""", unsafe_allow_html=True)

            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown(f'<div style="font-size:0.70rem;font-weight:600;color:{info_a["pl_color"]};font-family:monospace;margin-bottom:5px;">● {la}  β₀={b0_a}  β₁={b1_a}</div>', unsafe_allow_html=True)
                # MRI image
                if mri_a:
                    st.image(mri_a, use_container_width=True, caption=f"Synthetic MRI — {la}")
                # Simplicial complex
                st.image(fig_simplicial(p_a, epsilon, la, info_a["pl_color"]),
                         use_container_width=True)

            with col_r:
                st.markdown(f'<div style="font-size:0.70rem;font-weight:600;color:{info_b["pl_color"]};font-family:monospace;margin-bottom:5px;">● {lb}  β₀={b0_b}  β₁={b1_b}</div>', unsafe_allow_html=True)
                # MRI image
                if mri_b:
                    st.image(mri_b, use_container_width=True, caption=f"Synthetic MRI — {lb}")
                # Simplicial complex
                st.image(fig_simplicial(p_b, epsilon, lb, info_b["pl_color"]),
                         use_container_width=True)

            st.markdown("""
<div class="nt-card grey">
  <strong>วิธีอ่านภาพ:</strong>
  ภาพ MRI (บน) แสดงโครงสร้างเนื้อเยื่อจำลอง  ·
  Simplicial Complex (ล่าง) แสดง topology ที่สกัดได้<br>
  แต่ละจุด = ตำแหน่งเนื้อเยื่อที่สุ่มตัวอย่าง  ·
  วงกลมโปร่งใส = ε-ball  ·
  เส้นเชื่อม = 1-simplex  ·
  สามเหลี่ยม = 2-simplex  ·
  โพรงที่ไม่ถูก fill = <strong>H₁ feature (β₁)</strong>
</div>""", unsafe_allow_html=True)

            # Betti curves
            st.markdown("""
<div class="nt-rule">
  <div class="nt-rule-line"></div>
  <div class="nt-rule-text">Betti Curves  ·  Topological Signature</div>
  <div class="nt-rule-line"></div>
</div>""", unsafe_allow_html=True)

            bc1, bc2 = st.columns(2)
            with bc1:
                st.image(fig_betti_curves(r_a, f"Betti Curves — {la}"),
                         use_container_width=True)
            with bc2:
                st.image(fig_betti_curves(r_b, f"Betti Curves — {lb}"),
                         use_container_width=True)

        elif st.session_state.mode == "upload" and r_u and p_u is not None:
            b0_u, b1_u = betti_at(r_u, epsilon)
            s_u = topo_stats(r_u, 1)

            col1,col2,col3,col4 = st.columns(4)
            col1.metric("β₀  Components",     b0_u)
            col2.metric("β₁  Holes / Loops",  b1_u)
            col3.metric("H₁ Max Persistence", f"{s_u['max_p']:.4f}")
            col4.metric("จำนวน Points",        len(p_u))

            if b1_u >= 3:
                st.markdown(f"""<div class="nt-card red">
  <span class="nt-chip chip-alert" style="margin-right:8px;">HIGH COMPLEXITY</span>
  β₁ = {b1_u} — ตรวจพบ multiple persistent holes
  (ผลนี้เป็นส่วนหนึ่งของการวิจัย — ต้องยืนยันด้วยผู้เชี่ยวชาญ)
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="nt-card green">
  <span class="nt-chip chip-ok" style="margin-right:8px;">NORMAL RANGE</span>
  β₁ = {b1_u} — topology ค่อนข้างเรียบง่ายที่ ε = {epsilon:.2f}
</div>""", unsafe_allow_html=True)

            # Show uploaded MRI + point cloud side by side
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**ภาพ MRI ที่อัพโหลด**")
                if st.session_state.img_u is not None:
                    pil_u = Image.fromarray(st.session_state.img_u)
                    st.image(pil_u, use_container_width=True, caption="Uploaded MRI")
            with c2:
                st.markdown("**Simplicial Complex**")
                st.image(fig_simplicial(p_u, epsilon, "MRI Upload", _NAV),
                         use_container_width=True)

            st.image(fig_betti_curves(r_u, "Betti Curves — MRI Upload"),
                     use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — Persistence Diagnostics
# ════════════════════════════════════════════════════════════════
with tab_diag:
    st.markdown("## Primary Diagnostic Outputs")
    st.markdown("""
<div class="nt-card navy">
  <strong>Persistence Barcode</strong> และ <strong>Persistence Diagram</strong>
  คือ output หลักสำหรับการวินิจฉัย topology ของเนื้อเยื่อ<br>
  <strong style="color:#0d6e8a;">สีน้ำเงิน (H₀)</strong> = Connected Components  ·
  <strong style="color:#b45309;">สีเหลือง (H₁)</strong> = Holes / Loops<br>
  ความยาวแถบ / ระยะห่างจากเส้นทแยงมุม = persistence = ความสำคัญ
</div>""", unsafe_allow_html=True)

    if not ready:
        st.info("กรุณา Scan ข้อมูลก่อน")
    elif st.session_state.mode == "batch" and r_a and r_b:
        la = CASES[ca]["label"].split("(")[0].strip()
        lb = CASES[cb]["label"].split("(")[0].strip()

        # Combined 4-panel + MRI images
        c_img1, c_img2 = st.columns(2)
        with c_img1:
            if mri_a:
                st.image(mri_a, use_container_width=True,
                         caption=f"MRI Reference — {la}")
        with c_img2:
            if mri_b:
                st.image(mri_b, use_container_width=True,
                         caption=f"MRI Reference — {lb}")

        st.image(fig_dual_diagnostic(r_a, r_b, la, lb,
                                      CASES[ca]["pl_color"],
                                      CASES[cb]["pl_color"]),
                 use_container_width=True)

        # Summary
        s_a = topo_stats(r_a,1); s_b = topo_stats(r_b,1)
        d0_a=_dim_pairs(r_a,0); d0_b=_dim_pairs(r_b,0)
        st.markdown(f"""
<div class="nt-card amber">
  <strong>Comparative Topology Summary:</strong><br><br>
  <span style="font-family:'DM Mono',monospace;font-size:0.79rem;">
  Case A — H₀: {len(d0_a)} &nbsp;|&nbsp;
            H₁: {s_a['n']} &nbsp;|&nbsp;
            max_persistence: {s_a['max_p']:.4f} &nbsp;|&nbsp;
            entropy: {s_a['entropy']:.4f}<br>
  Case B — H₀: {len(d0_b)} &nbsp;|&nbsp;
            H₁: {s_b['n']} &nbsp;|&nbsp;
            max_persistence: {s_b['max_p']:.4f} &nbsp;|&nbsp;
            entropy: {s_b['entropy']:.4f}
  </span>
</div>""", unsafe_allow_html=True)

    elif st.session_state.mode == "upload" and r_u:
        c1, c2 = st.columns(2)
        with c1:
            if st.session_state.img_u is not None:
                st.image(Image.fromarray(st.session_state.img_u),
                         use_container_width=True, caption="Uploaded MRI")
            st.image(fig_barcode(r_u, "MRI Upload · Persistence Barcode"),
                     use_container_width=True)
        with c2:
            st.image(fig_persistence_diagram(r_u, "MRI Upload · Persistence Diagram"),
                     use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — Filtration Explorer
# ════════════════════════════════════════════════════════════════
with tab_filt:
    st.markdown("## Filtration Explorer")
    st.markdown("""
<div class="nt-card navy">
  <strong>Filtration</strong> คือการเพิ่มค่า ε ทีละขั้น
  สี่ panel แสดงสถานะ Simplicial Complex ณ ε ต่างกัน
  สังเกตวิธีที่ topology เปลี่ยนแปลงเมื่อ scale ใหญ่ขึ้น
</div>""", unsafe_allow_html=True)

    if not ready:
        st.info("กรุณา Scan ก่อน")
    else:
        if st.session_state.mode == "batch" and p_a is not None and p_b is not None:
            la = CASES[ca]["label"].split("(")[0].strip()
            lb = CASES[cb]["label"].split("(")[0].strip()
            st.markdown(f"#### Case A — {la}")
            st.image(fig_filtration_strip(p_a, CASES[ca]["pl_color"]),
                     use_container_width=True)
            st.markdown(f"#### Case B — {lb}")
            st.image(fig_filtration_strip(p_b, CASES[cb]["pl_color"]),
                     use_container_width=True)
        elif p_u is not None:
            st.image(fig_filtration_strip(p_u, _NAV), use_container_width=True)

        st.markdown(f"""
<div class="nt-rule">
  <div class="nt-rule-line"></div>
  <div class="nt-rule-text">Interactive  ·  ε = {epsilon:.3f}</div>
  <div class="nt-rule-line"></div>
</div>""", unsafe_allow_html=True)

        if st.session_state.mode == "batch" and p_a is not None and p_b is not None:
            cx1, cx2 = st.columns(2)
            with cx1:
                st.image(fig_simplicial(p_a, epsilon,
                         CASES[ca]["label"].split("(")[0].strip(),
                         CASES[ca]["pl_color"]), use_container_width=True)
            with cx2:
                st.image(fig_simplicial(p_b, epsilon,
                         CASES[cb]["label"].split("(")[0].strip(),
                         CASES[cb]["pl_color"]), use_container_width=True)
        elif p_u is not None:
            st.image(fig_simplicial(p_u, epsilon, "MRI Upload", _NAV),
                     use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 — Image Pipeline
# ════════════════════════════════════════════════════════════════
with tab_pipe:
    st.markdown("## ท่อประมวลผลภาพ MRI")
    st.markdown("""
<div class="nt-card teal">
  ก่อน TDA ภาพ MRI ผ่าน 4 ขั้นตอน:
  (1) Grayscale Input →
  (2) Gaussian Smoothing →
  (3) Binary Threshold →
  (4) Intensity Heatmap
</div>""", unsafe_allow_html=True)

    if st.session_state.mode == "batch":
        st.markdown("""
<div class="nt-card grey">
  โหมด Batch ใช้ข้อมูลจำลอง — ภาพ MRI สร้างแบบ synthetic<br>
  เปลี่ยนเป็นโหมด <strong>อัพโหลดภาพ MRI</strong> เพื่อดู image pipeline ของภาพจริง
</div>""", unsafe_allow_html=True)
        if mri_a or mri_b:
            st.markdown("**ภาพ MRI จำลองที่ใช้ในการวิเคราะห์**")
            c1, c2 = st.columns(2)
            if mri_a:
                with c1:
                    la = CASES[ca]["label"].split("(")[0].strip()
                    st.image(mri_a, use_container_width=True, caption=f"Case A — {la}")
            if mri_b:
                with c2:
                    lb = CASES[cb]["label"].split("(")[0].strip()
                    st.image(mri_b, use_container_width=True, caption=f"Case B — {lb}")

    elif st.session_state.mode == "upload":
        if st.session_state.img_u is not None:
            st.image(fig_pipeline(st.session_state.img_u, threshold, sigma),
                     use_container_width=True)
            if p_u is not None:
                c1,c2,c3 = st.columns(3)
                b0_u,b1_u = betti_at(r_u,epsilon) if r_u else (0,0)
                c1.metric("β₀  Components",b0_u)
                c2.metric("β₁  Holes",b1_u)
                c3.metric("Points",len(p_u))
                st.image(fig_simplicial(p_u,epsilon,"MRI Upload",_NAV),
                         use_container_width=True)
        else:
            st.info("อัพโหลดภาพและกด ▶ SCAN & ANALYSE")


# ════════════════════════════════════════════════════════════════
# TAB 5 — Theory & Diagnostics
# ════════════════════════════════════════════════════════════════
with tab_theory:
    st.markdown("## Theory & Diagnostics")

    with st.expander("1.  Computational Topology — จาก MRI Pixel สู่ Point Cloud", expanded=True):
        st.markdown("""
<div class="nt-card navy">
  การแปลงภาพ MRI เป็น Point Cloud สำหรับ TDA:
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div class="nt-formula">
Step 1:  I(x,y) → I_σ = (G_σ * I)(x,y)              [Gaussian smoothing]
Step 2:  B(x,y) = 1 if I_σ(x,y) > τ, else 0          [Binary threshold τ]
Step 3:  P = { (x, -y) | B(x,y) = 1 }                 [Extract coordinates]
Step 4:  P_norm ∈ [-1,1]²                              [Normalise]
Step 5:  P_cloud = random_sample(P_norm, n ≤ max_pts)  [Downsample]
</div>""", unsafe_allow_html=True)

    with st.expander("2.  Filtration Process — ε-Ball ขยายตัวเพื่อตรวจจับ Connectivity"):
        st.markdown("""
<div class="nt-formula">
VR(X, ε) = { σ ⊆ X  |  d(xᵢ, xⱼ) ≤ ε  ∀ xᵢ, xⱼ ∈ σ }

Filtration:  VR(X, 0) ⊆ VR(X, ε₁) ⊆ ... ⊆ VR(X, ∞)

birth(f) = ε ที่ feature f ปรากฏ
death(f) = ε ที่ feature f หายไป
persistence(f) = death(f) − birth(f)
</div>""", unsafe_allow_html=True)

    with st.expander("3.  Clinical Significance — ความหมายทางการแพทย์ของ β₀ และ β₁"):
        st.markdown("""
<div class="nt-card teal">
  <strong>β₀ — Tumor Mass Connectivity</strong><br>
  • β₀ = 1 → ก้อนเนื้องอกชิ้นเดียว (Meningioma)<br>
  • β₀ ≥ 2 → เนื้องอกแตกกระจาย / satellite masses<br>
  • β₀ สูงมาก → Metastatic pattern
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div class="nt-card amber">
  <strong>β₁ — Internal Structural Complexity / Voids</strong><br>
  • β₁ = 1 → เยื่อหุ้มปกติ ไม่มีโพรงภายใน<br>
  • β₁ = 2–3 → necrotic core ขนาดเล็ก (LGG)<br>
  • β₁ ≥ 4 → multiple necrotic voids → GBM Grade IV<br><br>
  Crawford et al. (2020) พบว่า persistence features ของ β₁
  มีความสัมพันธ์กับ overall survival ใน GBM
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div class="nt-formula">
╔═══════════════════════════════════════════════════════════╗
║  Tumor Type        │  β₀   │  β₁   │  H₁ Entropy        ║
╠═══════════════════════════════════════════════════════════╣
║  Normal Tissue     │  1    │  1    │  Low               ║
║  Meningioma I      │  1    │  1-2  │  Low               ║
║  LGG Grade II      │  1    │  2-3  │  Moderate          ║
║  GBM Grade IV      │  ≥2   │  ≥4   │  High              ║
║  Brain Metastasis  │  ≥4   │  ≥3   │  High              ║
╚═══════════════════════════════════════════════════════════╝
⚠  ค่าในตารางเป็นแนวทางวิจัย — ยืนยันด้วยพยาธิวิทยาเสมอ
</div>""", unsafe_allow_html=True)

    with st.expander("4.  Selected References"):
        st.markdown("""
<div class="nt-card lav">
• <strong>Carlsson, G. (2009).</strong>  "Topology and Data."
  <em>Bulletin of the AMS</em>, 46(2), 255–308.<br>
• <strong>Crawford, L. et al. (2020).</strong>  "Predicting Clinical Outcomes in Glioblastoma."
  <em>JASA</em>, 115(531), 1139–1150.<br>
• <strong>Cohen-Steiner, D. et al. (2007).</strong>  "Stability of Persistence Diagrams."
  <em>Discrete &amp; Computational Geometry</em>, 37(1), 103–120.<br>
• <strong>Tauzin, G. et al. (2021).</strong>  "giotto-tda."  <em>JMLR</em>, 22(39), 1–6.
</div>""", unsafe_allow_html=True)

    with st.expander("5.  Glossary — อภิธานศัพท์"):
        GLOSSARY = {
            "Simplicial Complex":    "โครงสร้างจากชุดของ simplex ที่ประกอบกัน",
            "Vietoris–Rips":        "วิธีสร้าง complex: เชื่อมจุดที่ระยะ ≤ ε",
            "Filtration":           "ลำดับ complex K₀⊆K₁⊆...⊆Kₙ ตาม ε",
            "Persistent Homology":  "คำนวณ homology ตลอด filtration",
            "β₀ (Betti-0)":        "จำนวน connected components",
            "β₁ (Betti-1)":        "จำนวน independent loops/holes",
            "Birth":                "ค่า ε ที่ feature เริ่มปรากฏ",
            "Death":                "ค่า ε ที่ feature หายไป",
            "Persistence":          "death − birth = ความสำคัญ",
            "Persistence Entropy":  "−Σpᵢlogpᵢ วัดความซับซ้อน",
            "Necrotic Core":        "โพรงตายในเนื้องอก → H₁ feature",
            "Downsampling":         "ลดจำนวนจุดเพื่อประหยัดการคำนวณ",
        }
        for term, defn in GLOSSARY.items():
            st.markdown(
                f'<div style="padding:6px 0;border-bottom:1px solid var(--border);">'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;'
                f'color:{_NAV};font-weight:600;">{term}</span>'
                f'<span style="color:#8fa5b4;"> — </span>'
                f'<span style="font-size:0.84rem;color:#546e7a;">{defn}</span></div>',
                unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 6 — Case Report
# ════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown("## Clinical Case Report")
    st.markdown("""
<div class="nt-card red">
  ⚠️  <strong>Disclaimer:</strong>
  รายงานนี้จัดทำเพื่อการวิจัยและการศึกษาเท่านั้น
  ผลการวิเคราะห์ TDA <strong>ไม่สามารถทดแทนการวินิจฉัยทางคลินิก</strong>
  การวินิจฉัยต้องดำเนินการโดยแพทย์ผู้เชี่ยวชาญ
</div>""", unsafe_allow_html=True)

    if not ready:
        st.info("กรุณา Scan ก่อน")
    else:
        now_str = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")

        if st.session_state.mode == "batch" and r_a and r_b:
            b0_a,b1_a = betti_at(r_a,epsilon)
            b0_b,b1_b = betti_at(r_b,epsilon)
            s_a = topo_stats(r_a,1); s_b = topo_stats(r_b,1)
            la,lb = CASES[ca]["label"],CASES[cb]["label"]
            verdict = "GBM Grade IV (TDA indicator)" if b1_b>=4 else \
                      "LGG / Moderate complexity" if b1_b>=2 else "Benign / Normal"

            report = f"""NeuroTopography v5.0 — TDA Clinical Case Report
═══════════════════════════════════════════════════════
DATE/TIME  :  {now_str}
TDA ENGINE :  {TDA_BACKEND or 'N/A'}
EPSILON    :  {epsilon:.3f}
MAX EDGE   :  {max_edge:.2f}
═══════════════════════════════════════════════════════

CASE A — {la}
  Thai   :  {CASES[ca]['thai']}
  β₀     :  {b0_a}
  β₁     :  {b1_a}
  H₁ n   :  {s_a['n']}
  max_p  :  {s_a['max_p']:.5f}
  entropy:  {s_a['entropy']:.5f}

CASE B — {lb}
  Thai   :  {CASES[cb]['thai']}
  β₀     :  {b0_b}  (Δ {b0_b-b0_a:+d})
  β₁     :  {b1_b}  (Δ {b1_b-b1_a:+d})
  H₁ n   :  {s_b['n']}
  max_p  :  {s_b['max_p']:.5f}  (Δ {s_b['max_p']-s_a['max_p']:+.5f})
  entropy:  {s_b['entropy']:.5f}  (Δ {s_b['entropy']-s_a['entropy']:+.5f})

═══════════════════════════════════════════════════════
TOPOLOGY SUMMARY
  β₁ Delta     :  {b1_b-b1_a:+d}
  Entropy Δ    :  {s_b['entropy']-s_a['entropy']:+.5f}
  Complexity   :  {'HIGH' if b1_b>=3 else 'MODERATE' if b1_b>=2 else 'LOW'}
  TDA Indicator:  {verdict}

NOTES
  • ผลนี้เป็นเชิง TDA research เท่านั้น
  • ต้องยืนยันด้วยการตรวจพยาธิวิทยาและคลินิก
  • Generated by NeuroTopography v5.0
═══════════════════════════════════════════════════════"""

            st.code(report, language="text")

            buf = io.StringIO(report)
            st.download_button(
                "⬇  Download Report (.txt)",
                data=report,
                file_name=f"neurotopography_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

        elif st.session_state.mode == "upload" and r_u and p_u is not None:
            b0_u,b1_u = betti_at(r_u,epsilon)
            s_u = topo_stats(r_u,1)
            report_u = f"""NeuroTopography v5.0 — MRI Upload Analysis
═══════════════════════════════════════════
DATE/TIME   :  {now_str}
BACKEND     :  {TDA_BACKEND}
THRESHOLD   :  {threshold}  |  SIGMA: {sigma:.1f}  |  MAX POINTS: {max_pts}
EPSILON     :  {epsilon:.3f}  |  MAX EDGE: {max_edge:.2f}
═══════════════════════════════════════════
β₀          :  {b0_u}
β₁          :  {b1_u}
H₁ features :  {s_u['n']}
Max persist.:  {s_u['max_p']:.5f}
Entropy H₁  :  {s_u['entropy']:.5f}
Points      :  {len(p_u)}
═══════════════════════════════════════════
Generated by NeuroTopography v5.0"""
            st.code(report_u, language="text")
            st.download_button(
                "⬇  Download Report (.txt)",
                data=report_u,
                file_name=f"neurotopography_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

    # Footer
    st.markdown(f"""
<div style="
  margin-top:40px; padding-top:18px;
  border-top:2px solid #1a3a5c;
  display:flex; align-items:center;
  justify-content:space-between; flex-wrap:wrap; gap:12px;
">
  <div style="font-size:1.0rem;font-weight:700;color:#0d1117;
              letter-spacing:-0.01em;">NeuroTopography</div>
  <div style="font-family:'DM Mono',monospace;font-size:0.60rem;
              color:#8fa5b4;text-align:right;line-height:1.9;">
    Advanced TDA for Precision Neuro-Oncology  ·  v5.0<br>
    Python  ·  Streamlit  ·  {TDA_BACKEND or 'TDA lib missing'}  ·
    NumPy  ·  SciPy  ·  Matplotlib
  </div>
</div>""", unsafe_allow_html=True)