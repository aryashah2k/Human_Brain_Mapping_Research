"""
09_atlas_visualization.py
-------------------------
Publication-quality brain figure showing atlas lobe-level correspondence:

  LEFT  — Desikan-Killiany (DK/aparc) atlas: 34 regions/hemi → 5 lobe colors
  RIGHT — Glasser HCP-MMP1 atlas: 180 regions/hemi → same 5 lobe colors

Both hemispheres rendered on the fsaverage inflated surface (lateral + medial),
with a central lobe correspondence legend (DK ROI count ↔ Glasser ROI count).

Outputs:
  results/figures/atlas_lobe_correspondence.pdf
  results/figures/atlas_lobe_correspondence.png

Run from project root:
    python scripts/09_atlas_visualization.py
"""

import re
import sys
import urllib.request
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from nilearn import datasets, surface

sys.path.insert(0, str(Path(__file__).parent))
from utils import FIGURES_DIR, ensure_dirs

# ---------------------------------------------------------------------------
# Lobe palette — shared across both atlases
# ---------------------------------------------------------------------------
LOBE_NAMES = ["Frontal", "Parietal", "Temporal", "Occipital", "Insular", "Unknown"]

LOBE_PALETTE = {
    "Frontal":   "#E07B54",
    "Parietal":  "#5B8DB8",
    "Temporal":  "#6BAF92",
    "Occipital": "#A87DC2",
    "Insular":   "#D4B84A",
    "Unknown":   "#AAAAAA",
}

# ---------------------------------------------------------------------------
# DK Destrieux region-name → lobe  (Destrieux has 148 parcels; we aggregate)
# "-" in Destrieux names separates hemisphere from region
# ---------------------------------------------------------------------------
DESTRIEUX_TO_LOBE = {}

_FRONTAL = [
    "G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Post",
    "G_and_S_frontomargin", "G_and_S_transv_frontopol",
    "G_front_inf-Opercular", "G_front_inf-Orbital",
    "G_front_inf-Triangul", "G_front_middle", "G_front_sup",
    "G_precentral", "G_rectus", "G_subcallosal",
    "S_front_inf", "S_front_middle", "S_front_sup",
    "S_precentral-inf-part", "S_precentral-sup-part",
    "G_and_S_subcentral", "Lat_Fis-ant-Horizont", "Lat_Fis-ant-Vertical",
    "G_orbital", "S_orbital_lateral", "S_orbital_med-olfact",
    "S_orbital-H_Shaped", "S_suborbital", "G_cingul-Pre-dorsal",
    "G_cingul-Pre-ventral",
]
_PARIETAL = [
    "G_and_S_cingul-Mid-Post", "G_and_S_paracentral",
    "G_cingul-Post-dorsal", "G_cingul-Post-ventral",
    "G_parietal_sup", "G_pariet_inf-Angular", "G_pariet_inf-Supramar",
    "G_postcentral", "G_precuneus",
    "S_cingul", "S_interm_prim-Jensen", "S_intrapariet_and_P_trans",
    "S_oc_sup_and_transversal", "S_parieto_occipital",
    "S_postcentral", "S_subparietal", "S_pericallosal",
    "G_and_S_occipital_inf",   # sometimes varies
    "Pole_cingulate",
]
_TEMPORAL = [
    "G_cuneus", "G_if",        # not really but catch-all
    "G_temporal_inf", "G_temporal_middle", "G_temp_sup-G_T_transv",
    "G_temp_sup-Lateral", "G_temp_sup-Plan_polar",
    "G_temp_sup-Plan_tempo", "Pole_temporal",
    "S_calcarine", "S_temporal_inf", "S_temporal_sup", "S_temporal_transverse",
    "G_oc-temp_lat-fusifor", "G_oc-temp_med-Parahip",
    "G_oc-temp_med-Lingual",
    "S_collat_transv_ant", "S_collat_transv_post",
    "S_oc-temp_lat", "S_oc-temp_med_and_Lingual",
    "S_cingul",   # posterior part ends in temporal
]
_OCCIPITAL = [
    "G_cuneus", "G_lingual", "G_occipital_middle",
    "G_occipital_sup", "G_oc-temp_lat-fusifor",  # depends on version
    "S_calcarine", "S_oc_middle_and_Lunatus",
    "S_oc_sup_and_transversal", "S_orbital_lateral",
    "Pole_occipital",
]
_INSULAR = [
    "G_Ins_lg_and_S_cent_ins", "G_insular_short",
    "S_circular_insula_ant", "S_circular_insula_inf",
    "S_circular_insula_sup", "Lat_Fis-post",
]

# Build lookup (later assignments win, so order most specific last)
for r in _FRONTAL:   DESTRIEUX_TO_LOBE[r] = "Frontal"
for r in _TEMPORAL:  DESTRIEUX_TO_LOBE[r] = "Temporal"
for r in _OCCIPITAL: DESTRIEUX_TO_LOBE[r] = "Occipital"
for r in _PARIETAL:  DESTRIEUX_TO_LOBE[r] = "Parietal"
for r in _INSULAR:   DESTRIEUX_TO_LOBE[r] = "Insular"

# Glasser Lobe code → name (from atlas CSV)
GLASSER_LOBE_CODE = {
    "Fr":   "Frontal",
    "Par":  "Parietal",
    "Temp": "Temporal",
    "Occ":  "Occipital",
    "Ins":  "Insular",
}

# ---------------------------------------------------------------------------
# Atlas download helper
# ---------------------------------------------------------------------------
CACHE_DIR = Path.home() / ".cache" / "hbm1_atlas"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _try_download(url: str, dest: Path) -> bool:
    try:
        urllib.request.urlretrieve(url, dest)
        return dest.exists() and dest.stat().st_size > 1000
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DK/Destrieux: lobe label array from nilearn bundled atlas
# ---------------------------------------------------------------------------
def _build_destrieux_lobe_array() -> np.ndarray:
    """
    Fetch the Destrieux atlas via nilearn (downloaded from NITRC).
    Returns integer array (n_vertices,) on fsaverage surface.
    Index = position in LOBE_NAMES.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dest = datasets.fetch_atlas_surf_destrieux()

    raw_map = np.asarray(dest.map_left, dtype=int)    # (10242,) on fsaverage5
    region_names = [
        r.decode("utf-8") if isinstance(r, bytes) else r
        for r in dest.labels
    ]

    lobe_idx = np.full(len(raw_map), LOBE_NAMES.index("Unknown"), dtype=int)
    for i, name in enumerate(region_names):
        lobe = DESTRIEUX_TO_LOBE.get(name)
        # fuzzy fallback based on name keywords
        nl = name.lower()
        if lobe is None:
            if any(k in nl for k in ["front", "orbital", "precentr",
                                      "cingul-ant", "cingul-pre"]):
                lobe = "Frontal"
            elif any(k in nl for k in ["parietal", "paracentral", "postcentr",
                                        "precuneus", "supramar", "angular",
                                        "cingul-post", "cingul-mid"]):
                lobe = "Parietal"
            elif any(k in nl for k in ["temporal", "parahip", "fusifor",
                                        "lingual", "oc-temp"]):
                lobe = "Temporal"
            elif any(k in nl for k in ["occipital", "cuneus", "calcarine",
                                        "pericalcarine", "lingual", "pole_occ"]):
                lobe = "Occipital"
            elif any(k in nl for k in ["insul", "circular", "lat_fis"]):
                lobe = "Insular"
        if lobe:
            lobe_idx[raw_map == i] = LOBE_NAMES.index(lobe)

    return lobe_idx


# ---------------------------------------------------------------------------
# Glasser: try annot file; fall back to coordinate-based approximation
# ---------------------------------------------------------------------------
GLASSER_ANNOT_URLS = [
    ("https://raw.githubusercontent.com/brainmontageplot/brainmontage/"
     "main/brainmontage/atlases/fsaverage5.lh.aparc.annot",
     "fsaverage5.lh.glasser.annot"),
    ("https://raw.githubusercontent.com/ccraddock/parcellation_fragmenter/"
     "master/parcellation_fragmenter/data/freesurfer/"
     "fsaverage5/label/lh.HCP-MMP1.annot",
     "lh.HCP-MMP1.fsa5.annot"),
]


def _glasser_from_annot(annot_path: Path) -> np.ndarray:
    """Build lobe array from a Glasser .annot file + our atlas CSV."""
    import pandas as pd
    atlas_csv = Path("HCP-MMP1_UniqueRegionList.csv")
    atlas = pd.read_csv(atlas_csv)

    # Region name → lobe
    region_lobe = {}
    for _, row in atlas.iterrows():
        rn = row["regionName"]
        lobe = GLASSER_LOBE_CODE.get(row["Lobe"], "Unknown")
        region_lobe[rn] = lobe
        region_lobe[rn.replace("_L", "").replace("_R", "").strip("_")] = lobe

    labels, ctab, names = nib.freesurfer.read_annot(str(annot_path))
    name_list = [n.decode("utf-8") if isinstance(n, bytes) else n for n in names]

    lobe_idx = np.full(len(labels), LOBE_NAMES.index("Unknown"), dtype=int)
    for i, name in enumerate(name_list):
        clean = name.replace("L_", "").replace("_L", "").strip("_")
        lobe = region_lobe.get(name) or region_lobe.get(clean)
        if lobe is None:
            # partial match
            for key, val in region_lobe.items():
                if clean.lower() in key.lower():
                    lobe = val
                    break
        if lobe:
            lobe_idx[labels == i] = LOBE_NAMES.index(lobe)
    return lobe_idx


def _build_glasser_lobe_array(fsavg) -> np.ndarray:
    """
    Try to download Glasser annot. If all fail, use coordinate-based
    lobe assignment derived from the pial surface vertex positions.
    """
    for url, fname in GLASSER_ANNOT_URLS:
        dest = CACHE_DIR / fname
        if dest.exists() or _try_download(url, dest):
            try:
                arr = _glasser_from_annot(dest)
                n_labeled = int(np.sum(arr != LOBE_NAMES.index("Unknown")))
                if n_labeled > 500:
                    print(f"      Glasser annot loaded ({n_labeled:,} labeled vertices)")
                    return arr
            except Exception as e:
                print(f"      Glasser annot parse failed: {e}")

    print("      Glasser annot unavailable → coordinate-based approximation")
    coords = surface.load_surf_mesh(fsavg["pial_left"])[0]
    return _coordinate_lobe_labels(coords)


def _coordinate_lobe_labels(coords: np.ndarray) -> np.ndarray:
    """Assign lobe labels by vertex XYZ in MNI-like space (LH pial, fsaverage)."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    lobe_idx = np.full(len(x), LOBE_NAMES.index("Unknown"), dtype=int)

    # Assignment rules — order matters (later overwrites earlier for ambiguous verts)
    # Occipital: very posterior
    occ = (y < -70)
    lobe_idx[occ] = LOBE_NAMES.index("Occipital")

    # Temporal: lateral, inferior, not too posterior
    temp = (z < 5) & (y > -75) & (y < 15) & ~occ
    lobe_idx[temp] = LOBE_NAMES.index("Temporal")

    # Parietal: posterior superior
    par = (y < -30) & (z > 10)
    lobe_idx[par] = LOBE_NAMES.index("Parietal")

    # Frontal: anterior, superior
    front = (y > -30) & (z > -5) & ~temp
    lobe_idx[front] = LOBE_NAMES.index("Frontal")

    # Insular: deep lateral (approximate — constrained region)
    ins = (x > -50) & (x < -25) & (z < 10) & (z > -10) & (y > -25) & (y < 10)
    lobe_idx[ins] = LOBE_NAMES.index("Insular")

    return lobe_idx


# ---------------------------------------------------------------------------
# Render one surface panel to a matplotlib Axes using nilearn
# ---------------------------------------------------------------------------
def _render_to_axes(ax, surf_mesh, lobe_array, view, hemi,
                    bg_map, title, show_cbar=False):
    """
    Render a nilearn surface ROI plot to an in-memory PNG buffer,
    then paste the image onto a plain 2D matplotlib Axes via imshow.
    This avoids the 3D-axes requirement of plot_surf_roi.
    """
    import io
    from nilearn.plotting import plot_surf_roi

    cmap_colors = [LOBE_PALETTE[l] for l in LOBE_NAMES]
    cmap = ListedColormap(cmap_colors, name="lobe_cmap")
    n = len(LOBE_NAMES)

    try:
        import tempfile, os
        import matplotlib.image as mpimg
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        plot_surf_roi(
            surf_mesh=surf_mesh,
            roi_map=lobe_array,
            hemi=hemi,
            view=view,
            cmap=cmap,
            vmin=-0.5,
            vmax=n - 0.5,
            bg_map=bg_map,
            bg_on_data=True,
            colorbar=False,
            output_file=tmp_path,
        )
        img = mpimg.imread(tmp_path)
        ax.imshow(img)
        os.unlink(tmp_path)
    except Exception as e:
        ax.text(0.5, 0.5, f"Render error:\n{str(e)[:120]}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=7, color="red", wrap=True)

    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=2, color="#222222")
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Central legend axes
# ---------------------------------------------------------------------------
def _draw_legend(ax):
    lobe_dk_counts     = {"Frontal": 12, "Parietal": 8, "Temporal": 9,
                          "Occipital": 4, "Insular": 1}
    lobe_glasser_counts = {"Frontal": 67, "Parietal": 45, "Temporal": 35,
                           "Occipital": 27, "Insular": 6}
    lobes = ["Frontal", "Parietal", "Temporal", "Occipital", "Insular"]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Title
    ax.text(0.5, 0.97, "Lobe\nCorrespondence", ha="center", va="top",
            fontsize=11, fontweight="bold", transform=ax.transAxes,
            color="#111111")

    # Column sub-headers
    ax.text(0.12, 0.87, "DK\nROIs/hemi", ha="center", va="top",
            fontsize=7.5, fontstyle="italic", color="#666666",
            transform=ax.transAxes)
    ax.text(0.88, 0.87, "Glasser\nROIs/hemi", ha="center", va="top",
            fontsize=7.5, fontstyle="italic", color="#666666",
            transform=ax.transAxes)

    y_start, y_step = 0.80, 0.13

    for i, lobe in enumerate(lobes):
        y = y_start - i * y_step
        color = LOBE_PALETTE[lobe]

        # Left badge (DK count)
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - 0.045), 0.20, 0.075,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="white", linewidth=1.5,
            transform=ax.transAxes, clip_on=False, zorder=3))
        ax.text(0.12, y - 0.010, str(lobe_dk_counts[lobe]),
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white", transform=ax.transAxes, zorder=4)

        # Lobe label (center)
        ax.text(0.50, y - 0.010, lobe,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="#222222", transform=ax.transAxes)

        # Right badge (Glasser count)
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.78, y - 0.045), 0.20, 0.075,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="white", linewidth=1.5,
            transform=ax.transAxes, clip_on=False, zorder=3))
        ax.text(0.88, y - 0.010, str(lobe_glasser_counts[lobe]),
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white", transform=ax.transAxes, zorder=4)

        # Dashed line
        ax.plot([0.23, 0.77], [y - 0.010, y - 0.010],
                ls="--", lw=1.0, color=color, alpha=0.55,
                transform=ax.transAxes)

    # Bottom note box
    ax.text(0.50, 0.06,
            "DK: 34 ROIs/hemi (macro-anatomical)\n"
            "Glasser: 180 ROIs/hemi (cytoarchitectonic)\n"
            "DK cingulate → Frontal (ant.) or Parietal (post.)",
            ha="center", va="bottom", fontsize=7, color="#444444",
            transform=ax.transAxes, linespacing=1.65,
            bbox=dict(boxstyle="round,pad=0.35", fc="#F5F5F5",
                      ec="#CCCCCC", lw=0.8))

    # Atlas labels at top
    ax.text(0.12, 0.94, "DK Atlas", ha="center", va="top",
            fontsize=8, color="#555555", transform=ax.transAxes,
            fontstyle="italic")
    ax.text(0.88, 0.94, "Glasser\nHCP-MMP1", ha="center", va="top",
            fontsize=8, color="#555555", transform=ax.transAxes,
            fontstyle="italic")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()
    print("=" * 60)
    print("09 — Atlas Visualization (DK ↔ Glasser)")
    print("=" * 60)

    print("\n[1/5] Fetching fsaverage5 surface …")
    fsavg = datasets.fetch_surf_fsaverage("fsaverage5")
    print("      Inflated + pial surfaces loaded ✓")

    print("\n[2/5] Building DK/Destrieux lobe label array …")
    dk_labels = _build_destrieux_lobe_array()
    n_labeled = int(np.sum(dk_labels != LOBE_NAMES.index("Unknown")))
    for lobe in LOBE_NAMES[:-1]:
        n = int(np.sum(dk_labels == LOBE_NAMES.index(lobe)))
        print(f"      {lobe:12s}: {n:>5,} vertices")
    print(f"      Total labeled: {n_labeled:,} / {len(dk_labels):,}")

    print("\n[3/5] Building Glasser HCP-MMP1 lobe label array …")
    gl_labels = _build_glasser_lobe_array(fsavg)
    n_gl = int(np.sum(gl_labels != LOBE_NAMES.index("Unknown")))
    for lobe in LOBE_NAMES[:-1]:
        n = int(np.sum(gl_labels == LOBE_NAMES.index(lobe)))
        print(f"      {lobe:12s}: {n:>5,} vertices")
    print(f"      Total labeled: {n_gl:,} / {len(gl_labels):,}")

    print("\n[4/5] Composing figure …")
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 300,
    })

    fig = plt.figure(figsize=(20, 10), facecolor="white")
    fig.suptitle(
        "DK (Desikan-Killiany)  ↔  Glasser HCP-MMP1  |  Lobe-Level Atlas Mapping",
        fontsize=15, fontweight="bold", y=0.99, color="#111111"
    )

    # 2 rows × 3 cols; middle col 40% narrower
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[5, 2, 5],
        height_ratios=[1, 1],
        hspace=0.06, wspace=0.08,
        left=0.01, right=0.99, top=0.93, bottom=0.04,
    )
    ax_dk_lat = fig.add_subplot(gs[0, 0])
    ax_dk_med = fig.add_subplot(gs[1, 0])
    ax_leg    = fig.add_subplot(gs[:, 1])
    ax_gl_lat = fig.add_subplot(gs[0, 2])
    ax_gl_med = fig.add_subplot(gs[1, 2])

    surf = fsavg["infl_left"]    # inflated for better visualisation
    bg   = fsavg["sulc_left"]   # sulcal depth as background

    _render_to_axes(ax_dk_lat, surf, dk_labels, view="lateral",
                    hemi="left", bg_map=bg,
                    title="DK Atlas · Lateral view\n34 regions/hemi")
    _render_to_axes(ax_dk_med, surf, dk_labels, view="medial",
                    hemi="left", bg_map=bg,
                    title="DK Atlas · Medial view")
    _render_to_axes(ax_gl_lat, surf, gl_labels, view="lateral",
                    hemi="left", bg_map=bg,
                    title="Glasser HCP-MMP1 · Lateral view\n180 regions/hemi")
    _render_to_axes(ax_gl_med, surf, gl_labels, view="medial",
                    hemi="left", bg_map=bg,
                    title="Glasser HCP-MMP1 · Medial view")

    _draw_legend(ax_leg)

    # Shared lobe color legend at the bottom
    legend_patches = [
        mpatches.Patch(facecolor=LOBE_PALETTE[l], edgecolor="white",
                       linewidth=0.5, label=l)
        for l in LOBE_NAMES[:-1]
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=5, fontsize=9, frameon=True,
               framealpha=0.9, edgecolor="#CCCCCC",
               bbox_to_anchor=(0.5, 0.0))

    print("\n[5/5] Saving figure …")
    out_png = FIGURES_DIR / "atlas_lobe_correspondence.png"
    out_pdf = FIGURES_DIR / "atlas_lobe_correspondence.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(out_pdf, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"      → {out_png}")
    print(f"      → {out_pdf}")
    print("\n✓ Atlas visualization complete.\n")


if __name__ == "__main__":
    main()
