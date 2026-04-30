"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗                               ║
║   ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗                              ║
║   ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║                              ║
║   ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║                              ║
║   ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝                              ║
║   ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝                              ║
║                                                                              ║
║   T O P O G R A P H Y                                                        ║
║   Advanced TDA for Precision Neuro-Oncology                                  ║
║                                                                              ║
║   Medical AI Research Platform  ·  Persistent Homology Engine  ·  v4.0      ║
╚══════════════════════════════════════════════════════════════════════════════╝

สถาปัตยกรรมระบบ (System Architecture):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODULE 1  CONFIG          — Page setup, CSS clinical theme, matplotlib palette
MODULE 2  TDA ENGINE      — Backend detection, persistence computation, caching
MODULE 3  POINT CLOUD     — MRI → binary → downsampled point cloud pipeline
MODULE 4  BATCH SAMPLER   — Synthetic clinical case generator (5 case types)
MODULE 5  VISUALIZATIONS  — All matplotlib figure factories
MODULE 6  UI LAYOUT       — Sidebar, header, 6 diagnostic tabs
"""

# ══════════════════════════════════════════════════════════════════
# MODULE 1 — CONFIGURATION & GLOBAL STYLING
# ══════════════════════════════════════════════════════════════════
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter, label as nd_label
from PIL import Image, ImageEnhance
import io
import warnings
warnings.filterwarnings("ignore")

# ── TDA Backend ───────────────────────────────────────────────────
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

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroTopography | Precision Neuro-Oncology TDA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# Aesthetic direction: High-precision clinical workstation
# Ultra-dark navy-black background, ice-blue primary accent,
# warm amber for H₁ topology, crisp white labels.
# Typography: Syne for display (bold geometric), Fira Code for data.
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Noto+Sans+Thai:wght@300;400;500;600&family=Fira+Code:wght@400;500&display=swap');

/* ── CSS Design Tokens ── */
:root {
  --void:        #050709;
  --deep:        #080c12;
  --panel:       #0d1119;
  --raised:      #111720;
  --card:        #141c28;
  --border:      #1c2636;
  --border-hi:   #263348;
  --ice:         #64d2ff;       /* primary accent — ice blue */
  --ice-glow:    rgba(100,210,255,0.12);
  --ice-dim:     #2a6a8a;
  --amber:       #f59e0b;       /* H₁ topology */
  --amber-dim:   #78510a;
  --teal:        #2dd4bf;       /* H₀ topology */
  --teal-dim:    #0f5a52;
  --coral:       #fb7185;       /* anomaly / warning */
  --jade:        #4ade80;       /* normal / positive */
  --lavender:    #a78bfa;       /* theory / info */
  --snow:        #f0f4f8;       /* primary text */
  --mist:        #8fa3b8;       /* secondary text */
  --slate:       #4a5e72;       /* muted text */
  --font-display:'Syne', sans-serif;
  --font-thai:   'Noto Sans Thai', sans-serif;
  --font-mono:   'Fira Code', monospace;
}

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--void) !important;
  color: var(--snow) !important;
  font-family: var(--font-thai) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--deep) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
  font-family: var(--font-thai) !important;
  color: var(--snow) !important;
}

/* ── Display headings ── */
h1 {
  font-family: var(--font-display) !important;
  font-weight: 800 !important;
  font-size: 1.6rem !important;
  color: var(--snow) !important;
  letter-spacing: -0.02em !important;
  line-height: 1.2 !important;
}
h2 {
  font-family: var(--font-display) !important;
  font-weight: 700 !important;
  font-size: 1.1rem !important;
  color: var(--snow) !important;
  letter-spacing: -0.01em !important;
}
h3 {
  font-family: var(--font-thai) !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  color: var(--mist) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-top: 2px solid var(--ice) !important;
  border-radius: 6px !important;
  padding: 14px 18px !important;
  transition: border-top-color 0.2s;
}
[data-testid="stMetricValue"] {
  font-family: var(--font-mono) !important;
  color: var(--ice) !important;
  font-size: 2.4rem !important;
  font-weight: 500 !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--font-thai) !important;
  color: var(--slate) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] {
  border-bottom: 1px solid var(--border) !important;
  margin-bottom: 0 !important;
}
[data-testid="stTabs"] button {
  font-family: var(--font-thai) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: var(--slate) !important;
  padding: 12px 18px !important;
  letter-spacing: 0.02em !important;
}
[data-testid="stTabs"] button:hover { color: var(--mist) !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--ice) !important;
  border-bottom: 2px solid var(--ice) !important;
  font-weight: 600 !important;
}

/* ── Buttons ── */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--border-hi) !important;
  color: var(--mist) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.75rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border-radius: 4px !important;
  padding: 10px 22px !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  border-color: var(--ice) !important;
  color: var(--ice) !important;
  background: var(--ice-glow) !important;
  box-shadow: 0 0 16px rgba(100,210,255,0.08) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div {
  background: var(--ice) !important;
}
[data-testid="stSlider"] label {
  font-family: var(--font-mono) !important;
  font-size: 0.73rem !important;
  color: var(--slate) !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}

/* ── Select / Radio ── */
[data-testid="stRadio"] label,
[data-testid="stSelectbox"] label {
  font-family: var(--font-thai) !important;
  font-size: 0.84rem !important;
  color: var(--mist) !important;
}

/* ── Clinical Cards ── */
.nt-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 18px 22px;
  margin: 10px 0;
  font-family: var(--font-thai);
  font-size: 0.88rem;
  line-height: 1.85;
  color: var(--mist);
}
.nt-card.ice    { border-left: 3px solid var(--ice);    }
.nt-card.amber  { border-left: 3px solid var(--amber);  }
.nt-card.coral  { border-left: 3px solid var(--coral);  }
.nt-card.jade   { border-left: 3px solid var(--jade);   }
.nt-card.teal   { border-left: 3px solid var(--teal);   }
.nt-card.lav    { border-left: 3px solid var(--lavender);}

/* ── Status Chips ── */
.nt-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-family: var(--font-mono);
  font-size: 0.63rem;
  font-weight: 500;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  padding: 4px 10px;
  border-radius: 3px;
  border: 1px solid;
}
.chip-ice   { color:var(--ice);    border-color:var(--ice-dim);    background:var(--ice-glow);         }
.chip-amber { color:var(--amber);  border-color:var(--amber-dim);  background:rgba(245,158,11,0.08);   }
.chip-coral { color:var(--coral);  border-color:#7a2535;           background:rgba(251,113,133,0.08);  }
.chip-jade  { color:var(--jade);   border-color:#1a5e35;           background:rgba(74,222,128,0.08);   }
.chip-teal  { color:var(--teal);   border-color:var(--teal-dim);   background:rgba(45,212,191,0.08);   }

/* ── Section dividers ── */
.nt-rule {
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 26px 0 16px;
}
.nt-rule-line { flex:1; height:1px; background:var(--border); }
.nt-rule-text {
  font-family: var(--font-mono);
  font-size: 0.60rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--slate);
  white-space: nowrap;
}

/* ── Formula blocks ── */
.nt-formula {
  background: var(--void);
  border: 1px solid var(--border);
  border-left: 3px solid var(--lavender);
  border-radius: 4px;
  padding: 13px 20px;
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: #b8c9f0;
  margin: 9px 0;
  letter-spacing: 0.03em;
  line-height: 2.0;
}

/* ── Case scan cards ── */
.case-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px 16px;
  margin: 6px 0;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
}
.case-card:hover {
  border-color: var(--ice);
  background: var(--card);
}
.case-card-title {
  font-family: var(--font-display);
  font-size: 0.85rem;
  font-weight: 700;
  color: var(--snow);
  margin-bottom: 4px;
}
.case-card-sub {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--slate);
  letter-spacing: 0.08em;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--font-thai) !important;
  font-size: 0.92rem !important;
  font-weight: 600 !important;
  color: var(--snow) !important;
}
[data-testid="stExpanderDetails"] {
  background: var(--panel) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--panel) !important;
  border: 1px dashed var(--border-hi) !important;
  border-radius: 6px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }

/* ── Alerts ── */
hr { border-color: var(--border) !important; }
[data-testid="stAlert"] {
  background: var(--card) !important;
  border-radius: 5px !important;
  font-family: var(--font-thai) !important;
}

/* ── Sidebar section labels ── */
.sb-label {
  font-family: var(--font-mono);
  font-size: 0.60rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--slate);
  padding: 12px 0 4px;
  display: block;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib Clinical Palette ───────────────────────────────────
_BG   = "#050709"
_SURF = "#0d1119"
_GRID = "#1c2636"
_ICE  = "#64d2ff"
_AMB  = "#f59e0b"
_TEAL = "#2dd4bf"
_CORAL= "#fb7185"
_JADE = "#4ade80"
_LAV  = "#a78bfa"
_SNOW = "#f0f4f8"
_MIST = "#8fa3b8"

plt.rcParams.update({
    "figure.facecolor":  _BG,
    "axes.facecolor":    _SURF,
    "axes.edgecolor":    _GRID,
    "axes.labelcolor":   _MIST,
    "xtick.color":       "#4a5e72",
    "ytick.color":       "#4a5e72",
    "text.color":        _SNOW,
    "grid.color":        _GRID,
    "grid.linestyle":    ":",
    "grid.alpha":        0.55,
    "font.family":       "monospace",
    "font.size":         8.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlepad":     9,
})


# ══════════════════════════════════════════════════════════════════
# MODULE 2 — TDA COMPUTATION ENGINE
# ══════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, max_entries=12)
def cached_persistence(pts_bytes: bytes, max_edge: float) -> dict | None:
    """
    Cached Vietoris–Rips persistence computation.
    Key = (serialised point cloud bytes, max_edge) — recomputes only when data changes.
    """
    pts = np.frombuffer(pts_bytes, dtype=np.float64).reshape(-1, 2)
    return _run_persistence(pts, max_edge)


def _run_persistence(pts: np.ndarray, max_edge: float = 1.8) -> dict | None:
    if TDA_BACKEND == "giotto":
        return _persistence_giotto(pts, max_edge)
    elif TDA_BACKEND == "gudhi":
        return _persistence_gudhi(pts, max_edge)
    return None


def _persistence_giotto(pts: np.ndarray, max_edge: float) -> dict:
    try:
        X = pts[np.newaxis, :, :]
        vr = VietorisRipsPersistence(
            homology_dimensions=[0, 1],
            max_edge_length=max_edge,
            collapse_edges=True,
            n_jobs=-1,
        )
        dgm = vr.fit_transform(X)[0]          # shape (n, 3)
        return {"backend": "giotto", "pairs": dgm, "max_edge": max_edge}
    except Exception as err:
        st.warning(f"Giotto error: {err} — switching to GUDHI")
        return _persistence_gudhi(pts, max_edge)


def _persistence_gudhi(pts: np.ndarray, max_edge: float) -> dict:
    import gudhi
    rips    = gudhi.RipsComplex(points=pts, max_edge_length=max_edge)
    st_obj  = rips.create_simplex_tree(max_dimension=2)
    st_obj.compute_persistence()
    cap     = max_edge * 1.15
    finite  = [(b, min(d, cap), dim) for dim, (b, d) in st_obj.persistence() if d != float("inf")]
    inf_pts = [(b, cap, dim)         for dim, (b, d) in st_obj.persistence() if d == float("inf")]
    all_p   = finite + inf_pts
    pairs   = np.array(all_p, dtype=float) if all_p else np.zeros((0, 3))
    return {"backend": "gudhi", "pairs": pairs, "max_edge": max_edge}


def _dim_pairs(result: dict, dim: int) -> list[tuple[float, float]]:
    if result is None or len(result["pairs"]) == 0:
        return []
    p = result["pairs"]
    return [(float(r[0]), float(r[1])) for r in p[p[:, 2] == dim]]


def betti_at_epsilon(result: dict, eps: float) -> tuple[int, int]:
    """
    β₀(ε) = จำนวน connected components ที่ยังมีชีวิต ณ ε
    β₁(ε) = จำนวน independent loops / holes ที่ยังมีชีวิต ณ ε
    """
    if result is None:
        return 0, 0
    b0 = sum(1 for b, d in _dim_pairs(result, 0) if b <= eps < d)
    b1 = sum(1 for b, d in _dim_pairs(result, 1) if b <= eps < d)
    return b0, b1


def topology_stats(result: dict, dim: int = 1) -> dict:
    """Compute summary statistics for a homology dimension."""
    pairs = _dim_pairs(result, dim)
    if not pairs:
        return dict(n=0, max_p=0., mean_p=0., entropy=0., lifetimes=np.array([]))
    lt = np.array([d - b for b, d in pairs if d > b])
    if len(lt) == 0:
        return dict(n=0, max_p=0., mean_p=0., entropy=0., lifetimes=np.array([]))
    p  = lt / (lt.sum() + 1e-12)
    return dict(
        n=len(lt),
        max_p=float(lt.max()),
        mean_p=float(lt.mean()),
        entropy=float(-np.sum(p * np.log(p + 1e-12))),
        lifetimes=lt,
    )


# ══════════════════════════════════════════════════════════════════
# MODULE 3 — POINT CLOUD PIPELINE
# ══════════════════════════════════════════════════════════════════

def mri_to_cloud(
    img: np.ndarray,
    threshold: int,
    max_pts: int = 350,
    sigma: float = 1.3,
) -> np.ndarray:
    """
    แปลงภาพ MRI Grayscale เป็น 2D Point Cloud

    Pipeline:
      1. Gaussian smoothing (sigma) — ลด acquisition noise
      2. Binary thresholding        — คัดเลือก voxel เนื้อเยื่อ
      3. Coordinate normalisation   — [-1, 1] ² invariant to image size
      4. Random downsampling        — O(n²) neighbour search stays fast

    Parameters
    ----------
    img       : (H, W) uint8/float grayscale array
    threshold : pixel intensity cutoff
    max_pts   : maximum point cloud size after downsampling
    sigma     : Gaussian kernel σ

    Returns
    -------
    (N, 2) float64 array  — x (→right), y (→up)
    """
    smooth      = gaussian_filter(img.astype(np.float64), sigma=sigma)
    rows, cols  = np.where(smooth > threshold)
    if len(rows) < 3:
        return np.random.default_rng(0).standard_normal((10, 2)) * 0.1

    pts = np.column_stack([cols.astype(np.float64), -rows.astype(np.float64)])

    # Normalise each axis to [-1, 1]
    for d in range(2):
        lo, hi = pts[:, d].min(), pts[:, d].max()
        if hi > lo:
            pts[:, d] = 2.0 * (pts[:, d] - lo) / (hi - lo) - 1.0

    # Downsampling — uniform random (preserves global structure)
    if len(pts) > max_pts:
        idx = np.random.default_rng(42).choice(len(pts), max_pts, replace=False)
        pts = pts[idx]

    return pts


def pipeline_views(img: np.ndarray, threshold: int, sigma: float = 1.3) -> tuple:
    smooth  = gaussian_filter(img.astype(np.float64), sigma=sigma)
    binary  = (smooth > threshold).astype(np.float64)
    heatmap = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9)
    return img, smooth, binary, heatmap


# ══════════════════════════════════════════════════════════════════
# MODULE 4 — BATCH CLINICAL SAMPLE GENERATOR
# ══════════════════════════════════════════════════════════════════

CASE_CATALOG = {
    "GBM_Grade4": {
        "label": "Glioblastoma Multiforme  (Grade IV)",
        "thai":  "กลิโอบลาสโตมา มัลติฟอร์มี  WHO Grade IV",
        "color": _CORAL,
        "desc":  "เนื้องอกระดับสูงสุด — โครงสร้างซับซ้อน มีโพรงเนื้อตายหลายแห่ง β₁ สูงมาก",
        "chip":  "chip-coral",
    },
    "LGG_Grade2": {
        "label": "Low-Grade Glioma  (Grade II)",
        "thai":  "กลิโอมาระดับต่ำ  WHO Grade II",
        "color": _AMB,
        "desc":  "เนื้องอกเติบโตช้า — โครงสร้างสม่ำเสมอกว่า β₁ ปานกลาง",
        "chip":  "chip-amber",
    },
    "Meningioma": {
        "label": "Meningioma  (Grade I)",
        "thai":  "เนื้องอกเยื่อหุ้มสมอง  WHO Grade I",
        "color": _ICE,
        "desc":  "เนื้องอกขอบเขตชัดเจน มีขอบเรียบ — β₁ ต่ำ β₀ อาจสูงขึ้นเล็กน้อย",
        "chip":  "chip-ice",
    },
    "NormalTissue": {
        "label": "Normal Cerebral Tissue",
        "thai":  "เนื้อเยื่อสมองปกติ",
        "color": _JADE,
        "desc":  "เนื้อเยื่อปกติ — วงกลมสม่ำเสมอ β₀=1 β₁=1",
        "chip":  "chip-jade",
    },
    "Metastasis": {
        "label": "Brain Metastasis  (Multiple)",
        "thai":  "เนื้องอกแพร่กระจายสมอง (หลายก้อน)",
        "color": _LAV,
        "desc":  "ก้อนเนื้องอกหลายตำแหน่งแยกกัน — β₀ สูงมาก บ่งชี้การแพร่กระจาย",
        "chip":  "chip-ice",
    },
}


def generate_batch_case(case_id: str, rng_seed: int = 42) -> np.ndarray:
    """
    สร้าง point cloud จำลองสำหรับแต่ละ clinical case type

    ออกแบบให้มี topological signature เฉพาะตัว:
      GBM_Grade4  → β₀≥2, β₁≥4  (fragmented + multiple voids)
      LGG_Grade2  → β₀=1, β₁=2  (single mass + one cavity)
      Meningioma  → β₀=1, β₁=1  (clean lobulated ring)
      NormalTissue→ β₀=1, β₁=1  (perfect circle)
      Metastasis  → β₀≥4, β₁≥3  (many disconnected masses)
    """
    rng = np.random.default_rng(rng_seed)

    def ring(cx, cy, r, n, noise=0.03, t_range=(0, 2*np.pi)):
        t  = np.linspace(t_range[0], t_range[1], n, endpoint=False)
        rx = r + rng.normal(0, noise, n)
        return np.column_stack([cx + rx*np.cos(t), cy + rx*np.sin(t)])

    def blob(cx, cy, r, n, noise=0.025):
        return ring(cx, cy, r, n, noise=noise)

    if case_id == "NormalTissue":
        return ring(0, 0, 1.0, 180, noise=0.025)

    elif case_id == "Meningioma":
        outer = ring(0, 0, 1.0, 160, noise=0.06)
        bumps = np.vstack([
            ring(0, 0, 1.08 + 0.05*np.sin(k), 20, noise=0.02)
            for k in range(0, 12, 2)])
        return np.vstack([outer, bumps])

    elif case_id == "LGG_Grade2":
        outer  = ring(0, 0, 1.0, 180, noise=0.07)
        cavity = ring(0, 0, 0.38, 55, noise=0.04)
        return np.vstack([outer, cavity])

    elif case_id == "GBM_Grade4":
        outer  = ring(0, 0, 1.0, 200, noise=0.14)
        outer  = outer + rng.normal(0, 0.03, outer.shape)
        cav1   = ring(0.1, 0.1, 0.40, 60, noise=0.05)
        cav2   = ring(-0.15, -0.2, 0.22, 35, noise=0.04)
        sat1   = blob(1.82, 0.55, 0.20, 42, noise=0.03)
        sat2   = blob(-1.65, -0.60, 0.18, 38, noise=0.03)
        finger = ring(0, 0, 0.70, 30,
                      noise=0.04, t_range=(np.pi/4, 3*np.pi/4))
        return np.vstack([outer, cav1, cav2, sat1, sat2, finger])

    elif case_id == "Metastasis":
        centres = [(0,0,0.30), (1.8,0.5,0.22), (-1.7,-0.4,0.20),
                   (0.3,-1.6,0.18), (-0.5,1.7,0.16)]
        parts   = [ring(cx, cy, r, max(25, int(r*150)), noise=0.03)
                   for cx, cy, r in centres]
        return np.vstack(parts)

    return ring(0, 0, 1.0, 150, noise=0.03)  # fallback


# ══════════════════════════════════════════════════════════════════
# MODULE 5 — VISUALIZATION FACTORY
# ══════════════════════════════════════════════════════════════════

def _png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def fig_simplicial_complex(
    pts: np.ndarray,
    eps: float,
    title: str = "",
    color: str = _ICE,
    wh: tuple = (5.2, 5.2),
) -> bytes:
    """Render Vietoris–Rips Simplicial Complex at given ε."""
    fig, ax = plt.subplots(figsize=wh)
    ax.set_facecolor(_SURF)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18)

    D = cdist(pts, pts)
    n = len(pts)

    # ε-balls
    for pt in pts:
        ax.add_patch(Circle(pt, eps/2, color=color, alpha=0.045, zorder=1))

    # 1-simplices
    segs = [[pts[i], pts[j]]
            for i in range(n) for j in range(i+1, n) if D[i,j] <= eps]
    if segs:
        ax.add_collection(LineCollection(segs, color=color, alpha=0.25, lw=0.75, zorder=2))

    # 2-simplices
    for i in range(n):
        for j in range(i+1, n):
            if D[i,j] > eps: continue
            for k in range(j+1, n):
                if D[i,k] <= eps and D[j,k] <= eps:
                    ax.add_patch(plt.Polygon(
                        [pts[i], pts[j], pts[k]],
                        color=color, alpha=0.065, zorder=2))

    # 0-simplices
    ax.scatter(pts[:,0], pts[:,1], s=20, color=color, zorder=6,
               edgecolors=_BG, linewidths=0.35, alpha=0.93)

    pad = 0.35
    ax.set_xlim(pts[:,0].min()-pad, pts[:,0].max()+pad)
    ax.set_ylim(pts[:,1].min()-pad, pts[:,1].max()+pad)
    ax.set_title(f"{title}  |  ε = {eps:.3f}",
                 color=color, fontsize=9.5, fontfamily="monospace")
    fig.tight_layout()
    return _png(fig)


def fig_filtration_strip(
    pts: np.ndarray,
    color: str = _ICE,
    eps_list: tuple = (0.12, 0.42, 0.82, 1.38),
) -> bytes:
    """4-panel filtration evolution strip."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    D = cdist(pts, pts)
    n = len(pts)

    for ax, eps in zip(axes, eps_list):
        ax.set_facecolor(_SURF)
        ax.set_aspect("equal")
        for pt in pts:
            ax.add_patch(Circle(pt, eps/2, color=color, alpha=0.045))
        segs = [[pts[i], pts[j]] for i in range(n) for j in range(i+1,n) if D[i,j]<=eps]
        if segs:
            ax.add_collection(LineCollection(segs, color=color, alpha=0.28, lw=0.82))
        for i in range(n):
            for j in range(i+1,n):
                if D[i,j]>eps: continue
                for k in range(j+1,n):
                    if D[i,k]<=eps and D[j,k]<=eps:
                        ax.add_patch(plt.Polygon([pts[i],pts[j],pts[k]], color=color, alpha=0.07))
        ax.scatter(pts[:,0], pts[:,1], s=14, color=color, zorder=5,
                   edgecolors=_BG, linewidths=0.28)
        ax.set_title(f"ε = {eps:.2f}", color=color, fontsize=10, fontfamily="monospace")
        ax.set_xlim(pts[:,0].min()-0.28, pts[:,0].max()+0.28)
        ax.set_ylim(pts[:,1].min()-0.28, pts[:,1].max()+0.28)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Filtration Process  —  Simplicial Complex Growth",
                 color=_MIST, fontsize=10, y=1.02)
    fig.tight_layout()
    return _png(fig)


def fig_barcode(result: dict, title: str = "Persistence Barcode") -> bytes:
    """Persistence Barcode — primary diagnostic output."""
    if result is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "ไม่พบข้อมูล — ติดตั้ง gudhi หรือ giotto-tda",
                ha="center", va="center", color=_CORAL, fontsize=11)
        return _png(fig)

    d0 = _dim_pairs(result, 0)
    d1 = _dim_pairs(result, 1)
    total = len(d0) + len(d1)
    if total == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "ไม่พบ persistence features", ha="center", va="center", color=_AMB)
        return _png(fig)

    h = max(3.5, total * 0.30 + 1.8)
    fig, ax = plt.subplots(figsize=(9.0, h))
    ax.set_facecolor(_SURF)
    ax.grid(axis="x", alpha=0.18)

    y = 0
    for i, (b, d) in enumerate(sorted(d0, key=lambda x: x[1]-x[0], reverse=True)):
        ax.barh(y, d-b, left=b, height=0.62, color=_TEAL, alpha=0.82, zorder=3,
                label=f"H₀  Connected Components  (n={len(d0)})" if i==0 else "_nl")
        y += 1
    for i, (b, d) in enumerate(sorted(d1, key=lambda x: x[1]-x[0], reverse=True)):
        ax.barh(y, d-b, left=b, height=0.62, color=_AMB, alpha=0.84, zorder=3,
                label=f"H₁  Loops / Holes  (n={len(d1)})" if i==0 else "_nl")
        y += 1

    ax.axvline(0, color=_GRID, lw=1)
    ax.set_xlabel("Filtration parameter  ε", labelpad=7)
    ax.set_yticks([])
    ax.set_title(title, color=_MIST, fontsize=10)
    ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    return _png(fig)


def fig_persistence_diagram(result: dict, title: str = "Persistence Diagram") -> bytes:
    """Persistence Diagram — birth × death scatter — primary diagnostic output."""
    if result is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.text(0.5, 0.5, "ไม่พบข้อมูล", ha="center", va="center", color=_CORAL)
        return _png(fig)

    d0 = [(b,d) for b,d in _dim_pairs(result,0) if d < float("inf")]
    d1 = [(b,d) for b,d in _dim_pairs(result,1) if d < float("inf")]
    all_v = [v for pair in d0+d1 for v in pair]
    mv = (max(all_v) if all_v else 1.5) * 1.12

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.set_facecolor(_SURF)

    ax.plot([0,mv],[0,mv], color=_GRID, lw=1.4, zorder=1)
    ax.fill_between([0,mv],[0,mv], alpha=0.035, color="white")

    if d0:
        bv, dv = zip(*d0)
        ax.scatter(bv, dv, s=55, color=_TEAL, alpha=0.88, zorder=5,
                   edgecolors=_BG, lw=0.4, label=f"H₀  n={len(d0)}")
    if d1:
        bv, dv = zip(*d1)
        ax.scatter(bv, dv, s=68, color=_AMB, alpha=0.88, zorder=5,
                   edgecolors=_BG, lw=0.4, marker="D", label=f"H₁  n={len(d1)}")

    ax.set_xlabel("Birth  ε", labelpad=7)
    ax.set_ylabel("Death  ε", labelpad=7)
    ax.set_xlim(0,mv); ax.set_ylim(0,mv)
    ax.set_aspect("equal")
    ax.set_title(title, color=_MIST, fontsize=10)
    ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=8.5)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    return _png(fig)


def fig_betti_curves(result: dict, title: str = "Betti Curves") -> bytes:
    """β₀(ε) and β₁(ε) across full filtration range."""
    if result is None:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "ไม่พบข้อมูล", ha="center", va="center", color=_CORAL)
        return _png(fig)

    eps_r  = np.linspace(0, result["max_edge"], 300)
    b0s, b1s = [], []
    for e in eps_r:
        b0, b1 = betti_at_epsilon(result, e)
        b0s.append(b0); b1s.append(b1)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.set_facecolor(_SURF)
    ax.step(eps_r, b0s, where="post", color=_TEAL, lw=2.0, label="β₀  Connected Components", zorder=3)
    ax.step(eps_r, b1s, where="post", color=_AMB,  lw=2.0, label="β₁  Holes / Loops",        zorder=3)
    ax.fill_between(eps_r, b0s, step="post", alpha=0.07, color=_TEAL)
    ax.fill_between(eps_r, b1s, step="post", alpha=0.07, color=_AMB)
    ax.set_xlabel("Filtration parameter  ε", labelpad=7)
    ax.set_ylabel("Betti Number", labelpad=7)
    ax.set_title(title, color=_MIST, fontsize=10)
    ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=9)
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    return _png(fig)


def fig_pipeline(img: np.ndarray, threshold: int, sigma: float) -> bytes:
    """4-panel MRI pre-processing pipeline visualization."""
    orig, smooth, binary, hmap = pipeline_views(img, threshold, sigma)
    panels = [
        (orig,   "gray",    f"(1) Grayscale Input\nOriginal MRI Slice"),
        (smooth, "gray",    f"(2) Gaussian Smoothing\nσ = {sigma:.1f}"),
        (binary, "gray",    f"(3) Binary Threshold\nt = {threshold}"),
        (hmap,   "inferno", "(4) Intensity Heatmap\nNormalized Signal"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    for ax, (data, cmap, lbl) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_title(lbl, color=_MIST, fontsize=8.5, pad=8)
        ax.axis("off")
        if cmap == "inferno":
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("ท่อประมวลผลภาพ MRI  |  Pre-processing Pipeline before TDA",
                 color=_MIST, fontsize=10, y=1.02)
    fig.tight_layout()
    return _png(fig)


def fig_dual_diagnostic(r_a: dict, r_b: dict,
                         label_a: str, label_b: str,
                         color_a: str = _TEAL, color_b: str = _AMB) -> bytes:
    """
    Side-by-side diagnostic panel:
    Top row:    Barcode comparison
    Bottom row: Persistence Diagram comparison
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    def _draw_barcode(ax, result, color, lbl):
        ax.set_facecolor(_SURF)
        ax.grid(axis="x", alpha=0.18)
        d0 = _dim_pairs(result, 0) if result else []
        d1 = _dim_pairs(result, 1) if result else []
        y = 0
        for i,(b,d) in enumerate(sorted(d0, key=lambda x:x[1]-x[0], reverse=True)):
            ax.barh(y, d-b, left=b, height=0.60, color=_TEAL, alpha=0.80, zorder=3,
                    label=f"H₀  n={len(d0)}" if i==0 else "_nl")
            y += 1
        for i,(b,d) in enumerate(sorted(d1, key=lambda x:x[1]-x[0], reverse=True)):
            ax.barh(y, d-b, left=b, height=0.60, color=color, alpha=0.84, zorder=3,
                    label=f"H₁  n={len(d1)}" if i==0 else "_nl")
            y += 1
        ax.set_yticks([])
        ax.set_xlabel("ε", labelpad=5)
        ax.set_title(f"Barcode — {lbl}", color=color, fontsize=9.5)
        if d0 or d1:
            ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=7.5)

    def _draw_diagram(ax, result, color, lbl):
        ax.set_facecolor(_SURF)
        d0 = [(b,d) for b,d in (_dim_pairs(result,0) if result else []) if d<float("inf")]
        d1 = [(b,d) for b,d in (_dim_pairs(result,1) if result else []) if d<float("inf")]
        all_v = [v for pair in d0+d1 for v in pair]
        mv    = (max(all_v) if all_v else 1.5) * 1.12
        ax.plot([0,mv],[0,mv], color=_GRID, lw=1.2, zorder=1)
        if d0:
            bv,dv = zip(*d0)
            ax.scatter(bv,dv, s=50, color=_TEAL, alpha=0.88, zorder=5,
                       edgecolors=_BG, lw=0.35, label=f"H₀  n={len(d0)}")
        if d1:
            bv,dv = zip(*d1)
            ax.scatter(bv,dv, s=62, color=color, alpha=0.88, zorder=5,
                       edgecolors=_BG, lw=0.35, marker="D", label=f"H₁  n={len(d1)}")
        ax.set_xlim(0,mv); ax.set_ylim(0,mv)
        ax.set_aspect("equal")
        ax.set_xlabel("Birth ε", labelpad=5); ax.set_ylabel("Death ε", labelpad=5)
        ax.set_title(f"Diagram — {lbl}", color=color, fontsize=9.5)
        ax.grid(True, alpha=0.18)
        if d0 or d1:
            ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=7.5)

    _draw_barcode  (fig.add_subplot(gs[0,0]), r_a, color_a, label_a)
    _draw_barcode  (fig.add_subplot(gs[0,1]), r_b, color_b, label_b)
    _draw_diagram  (fig.add_subplot(gs[1,0]), r_a, color_a, label_a)
    _draw_diagram  (fig.add_subplot(gs[1,1]), r_b, color_b, label_b)

    fig.suptitle("Primary Diagnostic Outputs  —  Persistence Barcode & Diagram",
                 color=_MIST, fontsize=11, y=1.01)
    return _png(fig)


def fig_lifetime_compare(r_a: dict, r_b: dict,
                          label_a: str, label_b: str,
                          color_a=_TEAL, color_b=_AMB) -> bytes:
    """H₁ lifetime histogram comparison."""
    lt_a = topology_stats(r_a, 1)["lifetimes"] if r_a else np.array([])
    lt_b = topology_stats(r_b, 1)["lifetimes"] if r_b else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, lt, color, lbl in zip(axes, [lt_a, lt_b], [color_a, color_b], [label_a, label_b]):
        ax.set_facecolor(_SURF)
        if len(lt) > 0:
            bins = min(20, max(5, len(lt)))
            ax.hist(lt, bins=bins, color=color, alpha=0.72,
                    edgecolor=_BG, linewidth=0.5)
            ax.axvline(lt.mean(), color=_SNOW, lw=1.3, ls="--",
                       label=f"mean = {lt.mean():.4f}")
            ax.legend(facecolor=_BG, edgecolor=_GRID, fontsize=8.5)
        else:
            ax.text(0.5, 0.5, "ไม่มีข้อมูล", ha="center", va="center", color=_CORAL)
        ax.set_title(f"H₁ Lifetime Distribution\n{lbl}", color=_MIST, fontsize=9)
        ax.set_xlabel("Persistence (lifetime)", labelpad=6)
        ax.set_ylabel("Frequency", labelpad=6)
        ax.grid(True, alpha=0.18)

    fig.tight_layout()
    return _png(fig)


# ══════════════════════════════════════════════════════════════════
# MODULE 6 — STREAMLIT UI
# ══════════════════════════════════════════════════════════════════

# ── Session state ─────────────────────────────────────────────────
_DEFAULTS = {
    "result_a": None,  "result_b": None,   "result_upload": None,
    "pts_a": None,     "pts_b": None,       "pts_upload": None,
    "img_upload": None,
    "case_a": "NormalTissue",
    "case_b": "GBM_Grade4",
    "mode": "batch",
    "ready": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Chip for backend ──────────────────────────────────────────────
_CHIP = {
    "giotto": '<span class="nt-chip chip-ice">giotto-tda · ONLINE</span>',
    "gudhi":  '<span class="nt-chip chip-teal">GUDHI · ONLINE</span>',
    None:     '<span class="nt-chip chip-coral">TDA LIB · OFFLINE</span>',
}[TDA_BACKEND]

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="
  padding: 0 0 24px 0;
  border-bottom: 1px solid #1c2636;
  margin-bottom: 30px;
">
  <!-- Eyebrow -->
  <div style="
    font-family:'Fira Code',monospace;
    font-size:0.60rem;
    letter-spacing:0.22em;
    text-transform:uppercase;
    color:#4a5e72;
    margin-bottom:12px;
  ">Medical AI Research Platform  ·  Persistent Homology Engine  ·  v4.0</div>

  <!-- Wordmark row -->
  <div style="display:flex; align-items:flex-end; gap:18px; flex-wrap:wrap;">
    <div>
      <div style="
        font-family:'Syne',sans-serif;
        font-size:2.1rem;
        font-weight:800;
        color:#f0f4f8;
        letter-spacing:-0.03em;
        line-height:1;
      ">NeuroTopography</div>
      <div style="
        font-family:'Syne',sans-serif;
        font-size:0.78rem;
        font-weight:400;
        color:#4a5e72;
        letter-spacing:0.14em;
        text-transform:uppercase;
        margin-top:4px;
      ">Advanced TDA for Precision Neuro-Oncology</div>
    </div>
    <div style="
      padding-bottom:5px;
      display:flex;
      align-items:center;
      gap:8px;
      flex-wrap:wrap;
    ">
      {_CHIP}
      <span class="nt-chip chip-ice">🧠 Brain Tumor · TDA</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if TDA_BACKEND is None:
    st.error("⚠️  **ไม่พบไลบรารี TDA** — กรุณาติดตั้ง: `pip install gudhi`  หรือ  `pip install giotto-tda`")

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<span class="sb-label">โหมดการสแกน</span>', unsafe_allow_html=True)
    mode = st.radio(
        "",
        ["🔬  Batch Case Scanning", "🖼️  อัพโหลดภาพ MRI"],
        label_visibility="collapsed",
        key="mode_radio",
    )
    st.session_state.mode = "batch" if "Batch" in mode else "upload"

    st.markdown("---")

    if st.session_state.mode == "batch":
        st.markdown('<span class="sb-label">เลือก Case สำหรับการเปรียบเทียบ</span>',
                    unsafe_allow_html=True)
        case_options = list(CASE_CATALOG.keys())
        case_labels  = [CASE_CATALOG[k]["thai"] for k in case_options]

        idx_a = st.selectbox("Case A", range(len(case_options)),
                              format_func=lambda i: case_labels[i],
                              index=case_options.index("NormalTissue"))
        idx_b = st.selectbox("Case B", range(len(case_options)),
                              format_func=lambda i: case_labels[i],
                              index=case_options.index("GBM_Grade4"))
        st.session_state.case_a = case_options[idx_a]
        st.session_state.case_b = case_options[idx_b]
    else:
        st.markdown('<span class="sb-label">อัพโหลดภาพ</span>', unsafe_allow_html=True)
        upload_file = st.file_uploader("MRI slice (.jpg / .png)",
                                        type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.markdown('<span class="sb-label">พารามิเตอร์การประมวลผลภาพ</span>',
                unsafe_allow_html=True)
    threshold = st.slider("Pixel Threshold",   30, 230, 115, 5)
    sigma     = st.slider("Gaussian  σ",       0.5, 3.0, 1.3, 0.1)
    max_pts   = st.slider("Max Point Cloud",   50, 700, 300, 25)

    st.markdown("---")
    st.markdown('<span class="sb-label">พารามิเตอร์ Filtration</span>',
                unsafe_allow_html=True)
    epsilon   = st.slider("ε  (Epsilon)",      0.05, 2.0, 0.45, 0.05)
    max_edge  = st.slider("Max Edge Length",   0.5,  3.0, 1.8,  0.1)

    st.markdown("---")
    run_btn = st.button("▶  SCAN & ANALYSE", use_container_width=True)

    st.markdown("---")
    st.markdown("""
<div style="font-family:'Fira Code',monospace;font-size:0.60rem;color:#4a5e72;line-height:2.1;">
PIPELINE<br>
MRI Slice<br>
→ Gaussian Smooth<br>
→ Binary Threshold<br>
→ Point Cloud<br>
→ Rips Filtration<br>
→ Persistence Pairs<br>
→ Betti Numbers<br>
→ Clinical Report
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ANALYSIS EXECUTION
# ══════════════════════════════════════════════════════════════════
if run_btn:
    if st.session_state.mode == "batch":
        with st.spinner("กำลังสร้างข้อมูลจำลองและคำนวณ Persistent Homology..."):
            ca  = st.session_state.case_a
            cb  = st.session_state.case_b
            pts_a = generate_batch_case(ca)
            pts_b = generate_batch_case(cb)
            st.session_state.pts_a    = pts_a
            st.session_state.pts_b    = pts_b
            st.session_state.result_a = _run_persistence(pts_a, max_edge)
            st.session_state.result_b = _run_persistence(pts_b, max_edge)
            st.session_state.ready    = True
        st.sidebar.success("✅  Scan complete")

    elif st.session_state.mode == "upload":
        if "upload_file" in dir() and upload_file is not None:
            with st.spinner("กำลังประมวลผลภาพและคำนวณ TDA..."):
                img_arr  = np.array(Image.open(upload_file).convert("L"))
                pts_u    = mri_to_cloud(img_arr, threshold, max_pts, sigma)
                st.session_state.img_upload    = img_arr
                st.session_state.pts_upload    = pts_u
                st.session_state.result_upload = _run_persistence(pts_u, max_edge)
                st.session_state.ready         = True
            st.sidebar.success("✅  Analysis complete")
        else:
            st.sidebar.warning("⚠️  กรุณาอัพโหลดภาพก่อน")

# ── Shortcuts ─────────────────────────────────────────────────────
r_a  = st.session_state.result_a
r_b  = st.session_state.result_b
r_u  = st.session_state.result_upload
p_a  = st.session_state.pts_a
p_b  = st.session_state.pts_b
p_u  = st.session_state.pts_upload
ca   = st.session_state.case_a
cb   = st.session_state.case_b
ready = st.session_state.ready

# ══════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════
(tab_scan, tab_diagnostic, tab_filtration,
 tab_pipeline, tab_theory, tab_report) = st.tabs([
    "📊  Clinical Scan Dashboard",
    "🔬  Persistence Diagnostics",
    "🌀  Filtration Explorer",
    "🌡️  Image Pipeline",
    "📐  Theory & Diagnostics",
    "📋  Case Report",
])


# ════════════════════════════════════════════════════════════════
# TAB 1 — Clinical Scan Dashboard
# ════════════════════════════════════════════════════════════════
with tab_scan:

    # Empty state
    if not ready:
        st.markdown(f"""
<div class="nt-card ice" style="text-align:center; padding:56px 28px;">
  <div style="font-size:3rem; margin-bottom:16px;">🧠</div>
  <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700;
              color:#f0f4f8; margin-bottom:10px;">
    NeuroTopography  ·  Precision Neuro-Oncology TDA
  </div>
  <div style="font-size:0.9rem; color:#4a5e72; line-height:1.9;">
    เลือก Case หรืออัพโหลดภาพ MRI ในแถบควบคุมด้านซ้าย<br>
    จากนั้นกดปุ่ม
    <span style="color:#64d2ff; font-family:monospace; font-size:0.82rem;">
      ▶ SCAN &amp; ANALYSE
    </span>
    เพื่อเริ่มการสแกน
  </div>
</div>""", unsafe_allow_html=True)

    else:
        # ── Case Info Banner ─────────────────────────────────────
        if st.session_state.mode == "batch":
            info_a = CASE_CATALOG[ca]
            info_b = CASE_CATALOG[cb]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
<div class="nt-card ice">
  <span class="nt-chip {info_a['chip']}" style="margin-bottom:8px;display:inline-block;">
    CASE A
  </span>
  <div style="font-family:'Syne',sans-serif;font-size:1.0rem;font-weight:700;
              color:#f0f4f8;margin:6px 0 4px;">{info_a['thai']}</div>
  <div style="font-family:'Fira Code',monospace;font-size:0.68rem;color:#4a5e72;
              margin-bottom:8px;">{info_a['label']}</div>
  <div style="font-size:0.85rem;color:#8fa3b8;">{info_a['desc']}</div>
</div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
<div class="nt-card amber">
  <span class="nt-chip {info_b['chip']}" style="margin-bottom:8px;display:inline-block;">
    CASE B
  </span>
  <div style="font-family:'Syne',sans-serif;font-size:1.0rem;font-weight:700;
              color:#f0f4f8;margin:6px 0 4px;">{info_b['thai']}</div>
  <div style="font-family:'Fira Code',monospace;font-size:0.68rem;color:#4a5e72;
              margin-bottom:8px;">{info_b['label']}</div>
  <div style="font-size:0.85rem;color:#8fa3b8;">{info_b['desc']}</div>
</div>""", unsafe_allow_html=True)

        # ── Betti Numbers ─────────────────────────────────────────
        st.markdown(f"""
<div class="nt-rule">
  <div class="nt-rule-line"></div>
  <div class="nt-rule-text">Betti Numbers  ·  ε = {epsilon:.3f}</div>
  <div class="nt-rule-line"></div>
</div>""", unsafe_allow_html=True)

        if st.session_state.mode == "batch" and r_a and r_b:
            b0_a, b1_a = betti_at_epsilon(r_a, epsilon)
            b0_b, b1_b = betti_at_epsilon(r_b, epsilon)
            s_a = topology_stats(r_a, 1)
            s_b = topology_stats(r_b, 1)

            label_a = CASE_CATALOG[ca]["label"].split("(")[0].strip()
            label_b = CASE_CATALOG[cb]["label"].split("(")[0].strip()

            # Metrics — Case A
            st.markdown(f'<div style="font-family:monospace;font-size:0.65rem;letter-spacing:0.14em;color:#4a5e72;text-transform:uppercase;margin-bottom:6px;">▌ {label_a}</div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("β₀  Components",       b0_a)
            col2.metric("β₁  Holes / Loops",    b1_a)
            col3.metric("H₁ Max Persistence",  f"{s_a['max_p']:.4f}")
            col4.metric("H₁ Entropy",          f"{s_a['entropy']:.4f}")

            st.markdown('<div style="margin:10px 0;"></div>', unsafe_allow_html=True)

            # Metrics — Case B
            st.markdown(f'<div style="font-family:monospace;font-size:0.65rem;letter-spacing:0.14em;color:#4a5e72;text-transform:uppercase;margin-bottom:6px;">▌ {label_b}</div>', unsafe_allow_html=True)
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("β₀  Components",      b0_b,  delta=f"{b0_b-b0_a:+d}")
            col6.metric("β₁  Holes / Loops",   b1_b,  delta=f"{b1_b-b1_a:+d}")
            col7.metric("H₁ Max Persistence", f"{s_b['max_p']:.4f}",
                        delta=f"{s_b['max_p']-s_a['max_p']:+.4f}")
            col8.metric("H₁ Entropy",         f"{s_b['entropy']:.4f}",
                        delta=f"{s_b['entropy']-s_a['entropy']:+.4f}")

            # ── Clinical verdict ──
            if b1_b > b1_a:
                vc, vt, vm = "amber", "chip-amber", (
                    f"<b>Topological Anomaly Detected:</b>  β₁ Case B ({b1_b}) &gt; β₁ Case A ({b1_a})<br>"
                    f"H₁ entropy สูงกว่า {s_b['entropy']:.3f} vs {s_a['entropy']:.3f} — "
                    f"บ่งชี้ internal structural complexity สูง ซึ่งสอดคล้องกับ multiple necrotic voids "
                    f"ที่พบใน high-grade glioma"
                )
            elif b0_b > b0_a:
                vc, vt, vm = "coral", "chip-coral", (
                    f"<b>Fragmentation Pattern:</b>  β₀ Case B ({b0_b}) &gt; β₀ Case A ({b0_a})<br>"
                    f"จำนวน connected components สูง — บ่งชี้โครงสร้างเนื้องอกที่แตกกระจาย "
                    f"อาจสอดคล้องกับ metastatic pattern หรือ satellite masses"
                )
            else:
                vc, vt, vm = "jade", "chip-jade", (
                    f"ค่า β₀ และ β₁ ทั้งสอง Case ใกล้เคียงกันที่ ε = {epsilon:.2f}<br>"
                    f"ลองเพิ่มค่า ε เพื่อเห็นความแตกต่างที่มาตราส่วนใหญ่ขึ้น"
                )

            st.markdown(f"""
<div class="nt-card {vc}" style="margin-top:14px;">
  <span class="nt-chip {vt}" style="margin-right:10px;">TOPOLOGY ANALYSIS</span>
  <span style="font-size:0.88rem;">{vm}</span>
</div>""", unsafe_allow_html=True)

            # ── Point cloud side-by-side ──
            st.markdown(f"""
<div class="nt-rule">
  <div class="nt-rule-line"></div>
  <div class="nt-rule-text">Simplicial Complex  ·  ε = {epsilon:.3f}</div>
  <div class="nt-rule-line"></div>
</div>""", unsafe_allow_html=True)

            cl, cr = st.columns(2)
            ca_color = CASE_CATALOG[ca]["color"]
            cb_color = CASE_CATALOG[cb]["color"]
            with cl:
                st.markdown(f'<div style="font-family:monospace;font-size:0.70rem;color:{ca_color};margin-bottom:5px;">● {label_a}  β₀={b0_a}  β₁={b1_a}</div>', unsafe_allow_html=True)
                st.image(fig_simplicial_complex(p_a, epsilon, label_a, ca_color),
                         use_container_width=True)
            with cr:
                st.markdown(f'<div style="font-family:monospace;font-size:0.70rem;color:{cb_color};margin-bottom:5px;">● {label_b}  β₀={b0_b}  β₁={b1_b}</div>', unsafe_allow_html=True)
                st.image(fig_simplicial_complex(p_b, epsilon, label_b, cb_color),
                         use_container_width=True)

            st.markdown("""
<div class="nt-card ice">
  <b>หลักการอ่านภาพ:</b>
  แต่ละจุดแทนตำแหน่งบนเยื่อหุ้มเนื้อเยื่อที่สุ่มตัวอย่าง
  วงกลมโปร่งใสคือ ε-ball รอบแต่ละจุด
  เมื่อ ε-balls ซ้อนกัน → เส้นเชื่อมจุด (1-simplex)
  สามเหลี่ยมสีจาง = 2-simplex ที่จุดสามจุดเชื่อมกันครบทุกคู่
  โพรง (holes) ที่ไม่ถูก fill = <b>H₁ features</b>
</div>""", unsafe_allow_html=True)

            # ── Betti Curves ──
            st.markdown("""
<div class="nt-rule">
  <div class="nt-rule-line"></div>
  <div class="nt-rule-text">Betti Curves  ·  Topological Signature</div>
  <div class="nt-rule-line"></div>
</div>""", unsafe_allow_html=True)

            bc1, bc2 = st.columns(2)
            with bc1:
                st.image(fig_betti_curves(r_a, f"Betti Curves — {label_a}"),
                         use_container_width=True)
            with bc2:
                st.image(fig_betti_curves(r_b, f"Betti Curves — {label_b}"),
                         use_container_width=True)

        elif st.session_state.mode == "upload" and r_u and p_u is not None:
            b0_u, b1_u = betti_at_epsilon(r_u, epsilon)
            s_u = topology_stats(r_u, 1)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("β₀  Components",     b0_u)
            col2.metric("β₁  Holes / Loops",  b1_u)
            col3.metric("H₁ Max Persistence", f"{s_u['max_p']:.4f}")
            col4.metric("จำนวน Points",        len(p_u))

            if b1_u >= 3:
                st.markdown(f"""<div class="nt-card coral">
  <span class="nt-chip chip-coral" style="margin-right:8px;">HIGH COMPLEXITY</span>
  β₁ = {b1_u} — ตรวจพบ multiple persistent holes
  ซึ่งอาจบ่งชี้ถึง necrotic cores หรือ internal voids
  (ผลนี้เป็นส่วนหนึ่งของการวิจัย — ต้องยืนยันด้วยผู้เชี่ยวชาญ)
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="nt-card jade">
  <span class="nt-chip chip-jade" style="margin-right:8px;">NORMAL RANGE</span>
  β₁ = {b1_u} — topology ค่อนข้างเรียบง่ายที่ ε = {epsilon:.2f}
</div>""", unsafe_allow_html=True)

            st.image(fig_simplicial_complex(p_u, epsilon, "MRI Upload", _ICE),
                     use_container_width=True)
            st.image(fig_betti_curves(r_u, "Betti Curves — MRI Upload"),
                     use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — Persistence Diagnostics
# ════════════════════════════════════════════════════════════════
with tab_diagnostic:
    st.markdown("## Primary Diagnostic Outputs")
    st.markdown("""
<div class="nt-card ice">
  <b>Persistence Barcode</b> และ <b>Persistence Diagram</b>
  คือ output หลักสำหรับการวินิจฉัย topology ของเนื้อเยื่อ
  ทั้งสองรูปแสดงข้อมูลเดียวกันในรูปแบบต่างกัน
  <br><br>
  <b>เกณฑ์การอ่าน:</b>
  แถบ / จุด สีเขียว = H₀ (Connected Components)  ·
  แถบ / จุด สีเหลือง = H₁ (Holes / Loops)<br>
  ความยาวแถบ = persistence = ความสำคัญทางสถิติ  ·
  จุดไกลเส้นทแยงมุม = feature มีนัยสำคัญสูง
</div>""", unsafe_allow_html=True)

    if not ready:
        st.info("กรุณา Scan ข้อมูลก่อน")
    elif st.session_state.mode == "batch" and r_a and r_b:
        la = CASE_CATALOG[ca]["label"].split("(")[0].strip()
        lb = CASE_CATALOG[cb]["label"].split("(")[0].strip()

        # Combined 4-panel figure
        st.image(fig_dual_diagnostic(r_a, r_b, la, lb,
                                      CASE_CATALOG[ca]["color"],
                                      CASE_CATALOG[cb]["color"]),
                 use_container_width=True)

        # H₁ Lifetime histogram
        st.markdown("""
<div class="nt-rule">
  <div class="nt-rule-line"></div>
  <div class="nt-rule-text">H₁ Lifetime Distribution Comparison</div>
  <div class="nt-rule-line"></div>
</div>""", unsafe_allow_html=True)
        st.image(fig_lifetime_compare(r_a, r_b, la, lb,
                                       CASE_CATALOG[ca]["color"],
                                       CASE_CATALOG[cb]["color"]),
                 use_container_width=True)

        # Summary table
        s_a = topology_stats(r_a, 1)
        s_b = topology_stats(r_b, 1)
        d0_a = _dim_pairs(r_a, 0); d0_b = _dim_pairs(r_b, 0)
        st.markdown(f"""
<div class="nt-card amber">
  <b>Comparative Topology Summary:</b><br><br>
  <span style="font-family:'Fira Code',monospace;font-size:0.80rem;">
  Case A — H₀: {len(d0_a)} features &nbsp;|&nbsp;
            H₁: {s_a['n']} features &nbsp;|&nbsp;
            max persistence: {s_a['max_p']:.4f} &nbsp;|&nbsp;
            entropy: {s_a['entropy']:.4f}<br>
  Case B — H₀: {len(d0_b)} features &nbsp;|&nbsp;
            H₁: {s_b['n']} features &nbsp;|&nbsp;
            max persistence: {s_b['max_p']:.4f} &nbsp;|&nbsp;
            entropy: {s_b['entropy']:.4f}
  </span>
</div>""", unsafe_allow_html=True)

    elif st.session_state.mode == "upload" and r_u:
        c1, c2 = st.columns(2)
        with c1:
            st.image(fig_barcode(r_u, "MRI Upload · Persistence Barcode"),
                     use_container_width=True)
        with c2:
            st.image(fig_persistence_diagram(r_u, "MRI Upload · Persistence Diagram"),
                     use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — Filtration Explorer
# ════════════════════════════════════════════════════════════════
with tab_filtration:
    st.markdown("## Filtration Explorer")
    st.markdown("""
<div class="nt-card ice">
  <b>Filtration</b> คือการเพิ่มค่า ε จาก 0 จนถึง max_edge ทีละขั้น
  ในแต่ละขั้นจะเพิ่ม simplices ใหม่ตามกฎ:
  เชื่อมจุดสองจุดด้วย edge ถ้าระยะห่างน้อยกว่า ε
  โดยสี่ panel แสดงสถานะ Simplicial Complex ที่ ε ต่างกัน
</div>""", unsafe_allow_html=True)

    if not ready:
        st.info("กรุณา Scan ข้อมูลก่อน")
    else:
        if st.session_state.mode == "batch" and p_a is not None and p_b is not None:
            la = CASE_CATALOG[ca]["label"].split("(")[0].strip()
            lb = CASE_CATALOG[cb]["label"].split("(")[0].strip()
            st.markdown(f"#### Case A — {la}")
            st.image(fig_filtration_strip(p_a, CASE_CATALOG[ca]["color"]),
                     use_container_width=True)
            st.markdown(f"#### Case B — {lb}")
            st.image(fig_filtration_strip(p_b, CASE_CATALOG[cb]["color"]),
                     use_container_width=True)
        elif p_u is not None:
            st.image(fig_filtration_strip(p_u, _ICE), use_container_width=True)

        # Interactive slider view
        st.markdown(f"""
<div class="nt-rule">
  <div class="nt-rule-line"></div>
  <div class="nt-rule-text">Interactive  ·  ε = {epsilon:.3f}  (ปรับจาก Sidebar)</div>
  <div class="nt-rule-line"></div>
</div>""", unsafe_allow_html=True)

        if st.session_state.mode == "batch" and p_a is not None and p_b is not None:
            cx1, cx2 = st.columns(2)
            with cx1:
                st.image(fig_simplicial_complex(p_a, epsilon,
                         CASE_CATALOG[ca]["label"].split("(")[0].strip(),
                         CASE_CATALOG[ca]["color"]), use_container_width=True)
            with cx2:
                st.image(fig_simplicial_complex(p_b, epsilon,
                         CASE_CATALOG[cb]["label"].split("(")[0].strip(),
                         CASE_CATALOG[cb]["color"]), use_container_width=True)
        elif p_u is not None:
            st.image(fig_simplicial_complex(p_u, epsilon, "MRI Upload", _ICE),
                     use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 — Image Pipeline
# ════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("## ท่อประมวลผลภาพ MRI")
    st.markdown("""
<div class="nt-card teal">
  ก่อน TDA ภาพ MRI ผ่าน pipeline 4 ขั้นตอน:<br>
  <b>(1)</b> Grayscale Input → ภาพต้นฉบับ pixel intensity 0–255<br>
  <b>(2)</b> Gaussian Smoothing → กรอง acquisition noise ด้วย Gaussian kernel (σ ปรับได้)<br>
  <b>(3)</b> Binary Threshold → คัดเลือก voxels เนื้อเยื่อ (bright = tissue)<br>
  <b>(4)</b> Intensity Heatmap → แสดง signal distribution ด้วย inferno colormap<br>
  <br>
  จุดที่ผ่าน threshold แต่ละจุดกลายเป็นหนึ่งจุดใน Point Cloud ที่ใช้คำนวณ TDA
</div>""", unsafe_allow_html=True)

    if st.session_state.mode == "batch":
        st.markdown("""
<div class="nt-card amber">
  โหมด Batch Scanning ใช้ข้อมูลจำลองที่สร้างจากสมการ — ไม่มีภาพ MRI จริง<br>
  เปลี่ยนเป็นโหมด <b>อัพโหลดภาพ MRI</b> เพื่อดู Image Pipeline ของภาพจริง
</div>""", unsafe_allow_html=True)

    elif st.session_state.mode == "upload":
        if st.session_state.img_upload is not None:
            st.image(fig_pipeline(st.session_state.img_upload, threshold, sigma),
                     use_container_width=True)
            if p_u is not None:
                st.markdown(f"""
<div class="nt-rule">
  <div class="nt-rule-line"></div>
  <div class="nt-rule-text">Point Cloud Stats</div>
  <div class="nt-rule-line"></div>
</div>""", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                b0_u, b1_u = betti_at_epsilon(r_u, epsilon) if r_u else (0, 0)
                c1.metric("β₀  Components", b0_u)
                c2.metric("β₁  Holes",      b1_u)
                c3.metric("Points in Cloud", len(p_u))
                st.image(fig_simplicial_complex(p_u, epsilon, "MRI Upload", _ICE),
                         use_container_width=True)
        else:
            st.info("อัพโหลดภาพและกด ▶ SCAN & ANALYSE")


# ════════════════════════════════════════════════════════════════
# TAB 5 — Theory & Diagnostics
# ════════════════════════════════════════════════════════════════
with tab_theory:
    st.markdown("## Theory & Diagnostics")
    st.markdown("### ทฤษฎีและการประยุกต์ใช้ Topological Data Analysis ในงานวิจัย Neuro-Oncology")

    with st.expander("1.  Computational Topology — จาก MRI Pixel สู่ Point Cloud", expanded=True):
        st.markdown("""
<div class="nt-card ice">
  <b>ขั้นตอนการแปลงภาพ MRI เป็น Point Cloud สำหรับ TDA:</b>
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div class="nt-formula">
Step 1:  I(x,y) → I_σ(x,y) = (G_σ * I)(x,y)          [Gaussian smoothing]
Step 2:  B(x,y) = 1 if I_σ(x,y) > τ, else 0             [Binary threshold τ]
Step 3:  P = { (x, -y) | B(x,y) = 1 }                   [Coordinate extraction]
Step 4:  P_norm = 2(P - min(P)) / (max(P) - min(P)) - 1  [Normalise to [-1,1]²]
Step 5:  P_cloud = random_sample(P_norm, n = max_pts)     [Downsampling]
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div class="nt-card lav">
  <b>เหตุใดต้อง Downsample?</b><br>
  การคำนวณ Rips complex มีความซับซ้อน O(n²) ถึง O(n³)
  ภาพ MRI ความละเอียด 512×512 อาจมีจุดมากกว่า 100,000 จุด
  การ downsample เป็น 200–400 จุดลดเวลาคำนวณจาก ชั่วโมง เป็น วินาที
  โดยรักษา topological structure หลักไว้ได้ เนื่องจาก
  persistence diagram มีความ stable ต่อการสุ่มตัวอย่าง (Stability Theorem)
</div>""", unsafe_allow_html=True)

    with st.expander("2.  Filtration Process — การขยาย ε-Ball เพื่อตรวจจับ Connectivity"):
        st.markdown("""
<div class="nt-card teal">
  <b>แนวคิดพื้นฐานของ Vietoris–Rips Filtration:</b><br>
  ลองจินตนาการว่าเราค่อยๆ "พองลม" บอลลูนรอบแต่ละจุดในภาพ
  เมื่อบอลลูนสองอันสัมผัสกัน เราเชื่อมจุดทั้งสองด้วยเส้น
  เมื่อสามจุดล้อมรอบกัน เราเติมสามเหลี่ยม
  โพรง (holes) ที่ยังไม่ถูกเติม = β₁ features
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div class="nt-formula">
VR(X, ε) = { σ ⊆ X  |  d(xᵢ, xⱼ) ≤ ε  ∀ xᵢ, xⱼ ∈ σ }

Filtration:   VR(X, 0) ⊆ VR(X, ε₁) ⊆ VR(X, ε₂) ⊆ ... ⊆ VR(X, ∞)

Topological event:
  birth(f) = ε ที่ feature f ปรากฏขึ้นครั้งแรก
  death(f) = ε ที่ feature f หายไป (ถูก fill หรือ merge)
  persistence(f) = death(f) - birth(f)
</div>""", unsafe_allow_html=True)

    with st.expander("3.  Clinical Significance — ความหมายทางการแพทย์ของ β₀ และ β₁"):
        st.markdown("""
<div class="nt-card amber">
  <b>β₀ — Tumor Mass Connectivity</b><br>
  β₀ นับจำนวน connected components ณ มาตราส่วน ε ที่กำหนด
  ในบริบทของเนื้องอกสมอง:<br>
  • β₀ = 1 → ก้อนเนื้องอกชิ้นเดียว ขอบเขตชัดเจน (เช่น Meningioma)<br>
  • β₀ ≥ 2 → เนื้องอกแตกเป็นหลายก้อน หรือมี satellite masses<br>
  • β₀ สูงมาก → อาจบ่งชี้ metastatic pattern หรือ multifocal GBM
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div class="nt-card coral">
  <b>β₁ — Internal Structural Complexity / Voids</b><br>
  β₁ นับจำนวน independent loops หรือ holes ที่ไม่ถูก fill ณ ε นั้น
  ในบริบทของ neuro-oncology:<br>
  • β₁ = 1 → เยื่อหุ้มเนื้อเยื่อปกติ วงเดียว ไม่มีโพรงภายใน<br>
  • β₁ = 2–3 → อาจมี necrotic core 1–2 แห่ง (Low-grade glioma / early GBM)<br>
  • β₁ ≥ 4 → multiple necrotic voids → ลักษณะของ GBM Grade IV<br><br>
  <b>ความสัมพันธ์กับ WHO Grading:</b><br>
  งานวิจัย (Crawford et al., 2020) พบว่า persistence features ของ β₁
  มีความสัมพันธ์กับ overall survival ใน GBM อย่างมีนัยสำคัญทางสถิติ
</div>""", unsafe_allow_html=True)

        # Reference table
        st.markdown("""
<div class="nt-formula" style="overflow-x:auto;">
╔══════════════════════════════════════════════════════════════════╗
║  Tumor Type          │  β₀       │  β₁      │  H₁ Entropy      ║
╠══════════════════════════════════════════════════════════════════╣
║  Normal Tissue       │  1        │  1        │  Low             ║
║  Meningioma Grade I  │  1        │  1–2      │  Low–Moderate    ║
║  LGG Grade II        │  1        │  2–3      │  Moderate        ║
║  GBM Grade IV        │  ≥2       │  ≥4       │  High            ║
║  Brain Metastasis    │  ≥4       │  ≥3       │  High            ║
╚══════════════════════════════════════════════════════════════════╝
  ⚠  ค่าในตารางเป็นแนวทางวิจัย — ต้องยืนยันด้วยพยาธิวิทยาเสมอ
</div>""", unsafe_allow_html=True)

    with st.expander("4.  Persistence Stability Theorem"):
        st.markdown("""
<div class="nt-card lav">
  <b>Bottleneck Stability (Cohen-Steiner et al., 2007):</b><br>
  ถ้า ‖f − g‖∞ ≤ δ แล้ว W∞(Dgm(f), Dgm(g)) ≤ δ<br>
  โดย W∞ คือ bottleneck distance ระหว่าง persistence diagrams
  <br><br>
  <b>ความหมายในทางปฏิบัติ:</b>
  Persistence diagram มีความ robust ต่อ noise ในภาพ MRI
  การรบกวนเล็กน้อย (quantum noise, motion artifact)
  จะเปลี่ยนแปลง diagram เพียงเล็กน้อยเท่านั้น
  นี่คือข้อได้เปรียบหลักของ TDA เหนือวิธีวิเคราะห์ภาพแบบดั้งเดิม
</div>""", unsafe_allow_html=True)

    with st.expander("5.  Selected References"):
        st.markdown("""
<div class="nt-card lav">

• <b>Carlsson, G. (2009).</b>
  "Topology and Data."
  <em>Bulletin of the American Mathematical Society</em>, 46(2), 255–308.<br>

• <b>Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007).</b>
  "Stability of Persistence Diagrams."
  <em>Discrete & Computational Geometry</em>, 37(1), 103–120.<br>

• <b>Crawford, L., et al. (2020).</b>
  "Predicting Clinical Outcomes in Glioblastoma: An Application of Topological
  and Functional Data Analysis."
  <em>Journal of the American Statistical Association</em>, 115(531), 1139–1150.<br>

• <b>Nicolau, M., Levine, A. J., & Carlsson, G. (2011).</b>
  "Topology based data analysis identifies a subgroup of breast cancers with a
  unique mutational profile."
  <em>PNAS</em>, 108(17), 7265–7270.<br>

• <b>Tauzin, G., et al. (2021).</b>
  "giotto-tda: A Topological Data Analysis Toolkit for Machine Learning and
  Data Exploration."
  <em>Journal of Machine Learning Research</em>, 22(39), 1–6.<br>
</div>""", unsafe_allow_html=True)

    with st.expander("6.  Glossary — อภิธานศัพท์เทคนิค"):
        GLOSSARY = {
            "Simplicial Complex":       "โครงสร้างจากชุดของ simplex ที่ประกอบกัน",
            "Vietoris–Rips Complex":    "VR(X,ε): simplex σ ∈ VR ⟺ d(x,y)≤ε ∀ x,y∈σ",
            "Filtration":               "ลำดับ K₀⊆K₁⊆...⊆Kₙ ที่เพิ่มขึ้นตาม ε",
            "Persistent Homology":      "การคำนวณ homology groups ตลอด filtration",
            "β₀  (Betti-0)":           "จำนวน connected components",
            "β₁  (Betti-1)":           "จำนวน independent loops / holes ใน 2D",
            "Birth":                    "ค่า ε ที่ topological feature เริ่มปรากฏ",
            "Death":                    "ค่า ε ที่ topological feature หายไป",
            "Persistence":              "death − birth = ความสำคัญของ feature",
            "Persistence Diagram":      "scatter plot ของ (birth, death) ทุก feature",
            "Persistence Barcode":      "กราฟแถบ [birth, death] ของแต่ละ feature",
            "Persistence Entropy":      "H = −Σpᵢlogpᵢ วัดความซับซ้อนของ diagram",
            "Bottleneck Distance":      "W∞ — ระยะห่างระหว่าง persistence diagrams",
            "Necrotic Core":            "บริเวณตายในเนื้องอก → H₁ feature ที่มีนัยสำคัญ",
            "GBM":                      "Glioblastoma Multiforme — มะเร็งสมองระดับสูงสุด",
            "WHO Grade":                "ระบบจัดระดับเนื้องอก I–IV ตาม WHO classification",
        }
        for term, defn in GLOSSARY.items():
            st.markdown(
                f'<div style="padding:7px 0;border-bottom:1px solid #1c2636;">'
                f'<span style="font-family:\'Fira Code\',monospace;font-size:0.78rem;'
                f'color:{_ICE};">{term}</span>'
                f'<span style="color:#4a5e72;"> — </span>'
                f'<span style="font-size:0.85rem;color:#8fa3b8;">{defn}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════
# TAB 6 — Case Report
# ════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown("## Clinical Case Report")
    st.markdown("""
<div class="nt-card amber">
  ⚠️  <b>ข้อจำกัดความรับผิดชอบ (Disclaimer)</b><br>
  รายงานนี้จัดทำขึ้นเพื่อวัตถุประสงค์ทางการวิจัยและการศึกษาเท่านั้น
  ผลการวิเคราะห์ TDA <b>ไม่สามารถใช้ทดแทนการวินิจฉัยทางคลินิก</b>
  ได้ในทุกกรณี การวินิจฉัยขั้นสุดท้ายต้องดำเนินการโดยแพทย์รังสีวิทยา
  หรือประสาทศัลยแพทย์ผู้เชี่ยวชาญ ร่วมกับข้อมูลทางคลินิกและผลพยาธิวิทยา
</div>""", unsafe_allow_html=True)

    if not ready:
        st.info("กรุณา Scan ข้อมูลก่อนเพื่อสร้าง Case Report")
    else:
        from datetime import datetime
        now_str = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")

        if st.session_state.mode == "batch" and r_a and r_b:
            b0_a, b1_a = betti_at_epsilon(r_a, epsilon)
            b0_b, b1_b = betti_at_epsilon(r_b, epsilon)
            s_a = topology_stats(r_a, 1)
            s_b = topology_stats(r_b, 1)
            la  = CASE_CATALOG[ca]["label"]
            lb  = CASE_CATALOG[cb]["label"]
            d_a = CASE_CATALOG[ca]["thai"]
            d_b = CASE_CATALOG[cb]["thai"]

            verdict = "GBM" if (b1_b >= 4 or s_b['entropy'] > 1.5) else \
                      "LGG" if (b1_b >= 2) else "Benign/Normal"

            st.markdown(f"""
<div style="
  background:#0d1119;
  border:1px solid #1c2636;
  border-radius:8px;
  padding:28px 32px;
  font-family:'Fira Code',monospace;
  font-size:0.78rem;
  color:#8fa3b8;
  line-height:2.0;
">
<div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
            color:#f0f4f8;margin-bottom:16px;letter-spacing:0.05em;">
  NeuroTopography · TDA Clinical Case Report
</div>

<span style="color:#4a5e72;">═══════════════════════════════════════════════════════════════</span><br>
<span style="color:#64d2ff;">SCAN DATE/TIME</span>   :  {now_str}<br>
<span style="color:#64d2ff;">TDA BACKEND    </span>   :  {TDA_BACKEND or "N/A"}<br>
<span style="color:#64d2ff;">EPSILON        </span>   :  {epsilon:.3f}<br>
<span style="color:#64d2ff;">MAX EDGE       </span>   :  {max_edge:.2f}<br>
<span style="color:#4a5e72;">═══════════════════════════════════════════════════════════════</span><br>

<br>
<span style="color:#f59e0b;font-weight:600;">CASE A — {la}</span><br>
<span style="color:#4a5e72;">    Thai label  :</span>  {d_a}<br>
<span style="color:#4a5e72;">    β₀           :</span>  {b0_a}  (connected components)<br>
<span style="color:#4a5e72;">    β₁           :</span>  {b1_a}  (holes / loops)<br>
<span style="color:#4a5e72;">    H₁ features  :</span>  {s_a['n']}<br>
<span style="color:#4a5e72;">    Max persist. :</span>  {s_a['max_p']:.5f}<br>
<span style="color:#4a5e72;">    Entropy H₁   :</span>  {s_a['entropy']:.5f}<br>

<br>
<span style="color:#f59e0b;font-weight:600;">CASE B — {lb}</span><br>
<span style="color:#4a5e72;">    Thai label  :</span>  {d_b}<br>
<span style="color:#4a5e72;">    β₀           :</span>  {b0_b}  (connected components)<br>
<span style="color:#4a5e72;">    β₁           :</span>  {b1_b}  (holes / loops)<br>
<span style="color:#4a5e72;">    H₁ features  :</span>  {s_b['n']}<br>
<span style="color:#4a5e72;">    Max persist. :</span>  {s_b['max_p']:.5f}<br>
<span style="color:#4a5e72;">    Entropy H₁   :</span>  {s_b['entropy']:.5f}<br>

<br>
<span style="color:#4a5e72;">═══════════════════════════════════════════════════════════════</span><br>
<span style="color:#fb7185;">TOPOLOGY SUMMARY</span><br>
<span style="color:#4a5e72;">    β₁ Delta     :</span>  {b1_b - b1_a:+d}  (Case B vs Case A)<br>
<span style="color:#4a5e72;">    Entropy Δ    :</span>  {s_b['entropy'] - s_a['entropy']:+.5f}<br>
<span style="color:#4a5e72;">    Complexity   :</span>  {"HIGH — Multiple persistent voids detected" if b1_b >= 3 else "MODERATE" if b1_b >= 2 else "LOW"}<br>
<span style="color:#4a5e72;">    TDA Indicator:</span>  {verdict}<br>

<br>
<span style="color:#4a5e72;">═══════════════════════════════════════════════════════════════</span><br>
<span style="color:#a78bfa;">NOTES</span><br>
<span style="color:#4a5e72;">  • ผลนี้เป็นเชิง TDA research เท่านั้น</span><br>
<span style="color:#4a5e72;">  • ต้องยืนยันด้วยการตรวจพยาธิวิทยาและคลินิก</span><br>
<span style="color:#4a5e72;">  • Generated by NeuroTopography v4.0</span><br>
<span style="color:#4a5e72;">═══════════════════════════════════════════════════════════════</span>
</div>""", unsafe_allow_html=True)

        elif st.session_state.mode == "upload" and r_u and p_u is not None:
            b0_u, b1_u = betti_at_epsilon(r_u, epsilon)
            s_u = topology_stats(r_u, 1)

            st.markdown(f"""
<div style="
  background:#0d1119;border:1px solid #1c2636;border-radius:8px;
  padding:28px 32px;font-family:'Fira Code',monospace;
  font-size:0.78rem;color:#8fa3b8;line-height:2.0;
">
<div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
            color:#f0f4f8;margin-bottom:16px;">
  NeuroTopography · MRI Upload Analysis Report
</div>
<span style="color:#4a5e72;">═══════════════════════════════════════</span><br>
<span style="color:#64d2ff;">DATE/TIME     :</span>  {now_str}<br>
<span style="color:#64d2ff;">BACKEND       :</span>  {TDA_BACKEND}<br>
<span style="color:#64d2ff;">THRESHOLD     :</span>  {threshold}<br>
<span style="color:#64d2ff;">SIGMA         :</span>  {sigma:.1f}<br>
<span style="color:#64d2ff;">MAX POINTS    :</span>  {max_pts}<br>
<span style="color:#64d2ff;">EPSILON       :</span>  {epsilon:.3f}<br>
<span style="color:#4a5e72;">═══════════════════════════════════════</span><br>
<br>
<span style="color:#64d2ff;">β₀ (Components)   :</span>  {b0_u}<br>
<span style="color:#64d2ff;">β₁ (Holes)        :</span>  {b1_u}<br>
<span style="color:#64d2ff;">H₁ Features       :</span>  {s_u['n']}<br>
<span style="color:#64d2ff;">Max Persistence   :</span>  {s_u['max_p']:.5f}<br>
<span style="color:#64d2ff;">Entropy H₁        :</span>  {s_u['entropy']:.5f}<br>
<span style="color:#64d2ff;">Point Cloud Size  :</span>  {len(p_u)}<br>
<span style="color:#4a5e72;">═══════════════════════════════════════</span><br>
<span style="color:#a78bfa;">Generated by NeuroTopography v4.0</span><br>
<span style="color:#4a5e72;">═══════════════════════════════════════</span>
</div>""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────
    st.markdown(f"""
<div style="
  margin-top:44px;
  padding-top:22px;
  border-top:1px solid #1c2636;
  display:flex;
  align-items:center;
  justify-content:space-between;
  flex-wrap:wrap;
  gap:12px;
">
  <div style="font-family:'Syne',sans-serif;font-size:1.0rem;font-weight:700;
              color:#f0f4f8;letter-spacing:-0.01em;">
    NeuroTopography
  </div>
  <div style="font-family:'Fira Code',monospace;font-size:0.60rem;color:#4a5e72;
              text-align:right;line-height:1.9;">
    Advanced TDA for Precision Neuro-Oncology  ·  v4.0  ·  Medical AI Research Platform<br>
    Python · Streamlit · {TDA_BACKEND or 'TDA lib not found'} · NumPy · SciPy · Matplotlib
  </div>
</div>""", unsafe_allow_html=True)