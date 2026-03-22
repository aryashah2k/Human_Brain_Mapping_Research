"""
Shared utilities for the VBM analysis pipeline.
All paths are relative to the project root (hbm1/).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ---------------------------------------------------------------------------
# Publication-quality matplotlib style
# ---------------------------------------------------------------------------
FONT_FAMILY = "Arial"
PALETTE_GROUP = ["#2166AC", "#D6604D"]   # blue=good/low, red=poor/high
PALETTE_CORR  = "#4D9221"                 # green for correlations
FIG_DPI       = 300

def set_pub_style() -> None:
    """Apply publication-ready matplotlib rcParams."""
    matplotlib.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    [FONT_FAMILY, "DejaVu Sans"],
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "axes.linewidth":     1.0,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "figure.dpi":         FIG_DPI,
        "savefig.dpi":        FIG_DPI,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype":       42,   # embed fonts (required for journals)
        "ps.fonttype":        42,
    })


# ---------------------------------------------------------------------------
# Path helpers (all relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # hbm1/

DATA_DIR     = PROJECT_ROOT                           # raw data files live here
RESULTS_DIR  = PROJECT_ROOT / "results"
TABLES_DIR   = RESULTS_DIR  / "tables"
FIGURES_DIR  = RESULTS_DIR  / "figures"
INTERIM_DIR  = RESULTS_DIR  / "interim"              # processed parquet files

AABC_DIR     = DATA_DIR     / "AABC_Release2_StructuralIDPs"

# File paths
HCP_CSV          = DATA_DIR  / "HCP_YA_subjects_2026.csv"
AABC_DEMO_CSV    = AABC_DIR  / "AABC_Demographics.csv"
AABC_THICK_CSV   = AABC_DIR  / "Cortical_Areal_Thicknesses.csv"
AABC_VOL_CSV     = AABC_DIR  / "Cortical_Areal_Volumes.csv"
AABC_AREA_CSV    = AABC_DIR  / "Cortical_Areal_Surface_Areas.csv"
AABC_MYELIN_CSV  = AABC_DIR  / "Cortical_Areal_Myelin.csv"
ATLAS_CSV        = DATA_DIR  / "HCP-MMP1_UniqueRegionList.csv"
AABC_XL_CSV      = AABC_DIR  / "AABC_Release2_Non-imaging_Data-XL.csv"

# Interim parquet files written by prep scripts
HCP_PARQUET      = INTERIM_DIR / "hcp_clean.parquet"
AABC_PARQUET     = INTERIM_DIR / "aabc_clean.parquet"


# ---------------------------------------------------------------------------
# Desikan-Killiany (DK) atlas region names (34 per hemisphere)
# Exactly as they appear in HCP_YA_subjects_2026.csv column names.
# ---------------------------------------------------------------------------
DK_REGIONS: list[str] = [
    "Bankssts", "Caudalanteriorcingulate", "Caudalmiddlefrontal",
    "Cuneus", "Entorhinal", "Fusiform", "Inferiorparietal",
    "Inferiortemporal", "Isthmuscingulate", "Lateraloccipital",
    "Lateralorbitofrontal", "Lingual", "Medialorbitofrontal",
    "Middletemporal", "Parahippocampal", "Paracentral",
    "Parsopercularis", "Parsorbitalis", "Parstriangularis",
    "Pericalcarine", "Postcentral", "Posteriorcingulate",
    "Precentral", "Precuneus", "Rostralanteriorcingulate",
    "Rostralmiddlefrontal", "Superiorfrontal", "Superiorparietal",
    "Superiortemporal", "Supramarginal", "Frontalpole",
    "Temporalpole", "Transversetemporal", "Insula",
]


def ensure_dirs() -> None:
    """Create all output directories, do nothing if they already exist."""
    for d in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR, INTERIM_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d for two independent groups (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1,  var2  = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_sd


def pearson_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Return (lower, upper) 95 % CI for a Pearson r via Fisher z-transform.
    """
    from scipy import stats as st
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = st.norm.ppf(1 - alpha / 2)
    z_lo, z_hi = z - crit * se, z + crit * se
    return float(np.tanh(z_lo)), float(np.tanh(z_hi))


# ---------------------------------------------------------------------------
# Save figure to both PDF and PNG
# ---------------------------------------------------------------------------
def save_fig(fig: plt.Figure, stem: str, subdir: str = "") -> None:
    """
    Save *fig* as <stem>.pdf and <stem>.png in FIGURES_DIR / subdir.

    Parameters
    ----------
    fig    : matplotlib Figure
    stem   : filename without extension (e.g. 'sleep_bar')
    subdir : optional subdirectory within FIGURES_DIR
    """
    out_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# APA-style table formatter
# ---------------------------------------------------------------------------
def format_p(p: float) -> str:
    """Return APA-formatted p-value string."""
    if p < 0.001:
        return "< .001"
    return f"= {p:.3f}".lstrip("0")


def apa_ttest_row(roi: str, t: float, df: float, p: float, d: float,
                  ci_lo: float, ci_hi: float, p_fdr: float,
                  sig_fdr: bool) -> dict:
    """Return one row dict for an APA t-test summary table."""
    return {
        "ROI":       roi,
        "t":         f"{t:.2f}",
        "df":        int(df),
        "p":         format_p(p),
        "d":         f"{d:.2f}",
        "95% CI [d]": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
        "p_FDR":     format_p(p_fdr),
        "FDR sig.":  "✓" if sig_fdr else "",
    }


def apa_corr_row(roi: str, r: float, p: float, n: int,
                 ci_lo: float, ci_hi: float, p_fdr: float,
                 sig_fdr: bool) -> dict:
    """Return one row dict for an APA correlation summary table."""
    return {
        "ROI":       roi,
        "r":         f"{r:.3f}",
        "n":         n,
        "p":         format_p(p),
        "95% CI [r]": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
        "p_FDR":     format_p(p_fdr),
        "FDR sig.":  "✓" if sig_fdr else "",
    }
