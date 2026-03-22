"""
07_atlas_mapping.py
-------------------
Build a cross-atlas correspondence table mapping Desikan-Killiany (DK) regions
to Glasser HCP-MMP1 lobes, based on anatomically verified lobe assignments.

Glasser atlas (from HCP-MMP1_UniqueRegionList.csv) uses exactly 5 lobe codes:
  Fr   = Frontal  (includes anterior/mid cingulate cortices)
  Par  = Parietal (includes posterior cingulate cortex)
  Temp = Temporal
  Occ  = Occipital (includes posterior cingulate cortex)
  Ins  = Insular

DK Cingulate regions are reassigned to their Glasser-equivalent lobes:
  Anterior cingulate (caudal/rostral)  → Frontal  (matches Glasser 'Fr')
  Posterior + isthmus cingulate        → Parietal (matches Glasser 'Par')

Outputs:
  results/tables/atlas_dk_to_glasser_map.csv
  results/tables/atlas_glasser_regions.csv
  results/tables/atlas_cross_dataset_summary.csv
  results/tables/atlas_glasser_cortex_summary.csv

Run from project root:
    python scripts/07_atlas_mapping.py
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import ATLAS_CSV, TABLES_DIR, ensure_dirs, DK_REGIONS

# ---------------------------------------------------------------------------
# Verified DK → Glasser lobe mapping
# Glasser uses exactly 5 lobes: Frontal, Parietal, Temporal, Occipital, Insular
# Cingulate parcels in Glasser are absorbed into Frontal (anterior) and Parietal
# (posterior/isthmus) — confirmed from HCP-MMP1_UniqueRegionList.csv:
#   Fr cortices include: 'Anterior_Cingulate_and_Medial_Prefrontal',
#                        'Paracentral_Lobular_and_Mid_Cingulate'
#   Par cortices include: 'Posterior_Cingulate'
# ---------------------------------------------------------------------------
DK_LOBE_MAP = {
    # ── Frontal ──────────────────────────────────────────────
    "Caudalmiddlefrontal":      "Frontal",
    "Rostralmiddlefrontal":     "Frontal",
    "Superiorfrontal":          "Frontal",
    "Lateralorbitofrontal":     "Frontal",
    "Medialorbitofrontal":      "Frontal",
    "Parsopercularis":          "Frontal",
    "Parsorbitalis":            "Frontal",
    "Parstriangularis":         "Frontal",
    "Precentral":               "Frontal",
    "Frontalpole":              "Frontal",
    # Cingulate → Frontal (anterior/mid cingulate; Glasser: Fr)
    "Caudalanteriorcingulate":  "Frontal",
    "Rostralanteriorcingulate": "Frontal",

    # ── Parietal ─────────────────────────────────────────────
    "Superiorparietal":         "Parietal",
    "Inferiorparietal":         "Parietal",
    "Supramarginal":            "Parietal",
    "Postcentral":              "Parietal",
    "Paracentral":              "Parietal",
    "Precuneus":                "Parietal",
    # Cingulate → Parietal (posterior/isthmus cingulate; Glasser: Par)
    "Posteriorcingulate":       "Parietal",
    "Isthmuscingulate":         "Parietal",

    # ── Temporal ─────────────────────────────────────────────
    "Superiortemporal":         "Temporal",
    "Middletemporal":           "Temporal",
    "Inferiortemporal":         "Temporal",
    "Temporalpole":             "Temporal",
    "Transversetemporal":       "Temporal",
    "Fusiform":                 "Temporal",
    "Parahippocampal":          "Temporal",
    "Bankssts":                 "Temporal",
    "Entorhinal":               "Temporal",

    # ── Occipital ────────────────────────────────────────────
    "Cuneus":                   "Occipital",
    "Lateraloccipital":         "Occipital",
    "Lingual":                  "Occipital",
    "Pericalcarine":            "Occipital",

    # ── Insular ──────────────────────────────────────────────
    "Insula":                   "Insular",
}

# Direct mapping from Glasser CSV 'Lobe' codes to readable names
GLASSER_LOBE_CODE = {
    "Fr":   "Frontal",
    "Par":  "Parietal",
    "Temp": "Temporal",
    "Occ":  "Occipital",
    "Ins":  "Insular",
}

# Glasser cortex names that contain cingulate parcels (for cross-reference table)
GLASSER_CINGULATE_CORTICES = {
    "Anterior_Cingulate_and_Medial_Prefrontal": "Frontal",
    "Paracentral_Lobular_and_Mid_Cingulate":    "Frontal",
    "Posterior_Cingulate":                       "Parietal",
}


def build_dk_table() -> pd.DataFrame:
    """DK regions with their Glasser-equivalent lobe assignments."""
    rows = []
    for region in DK_REGIONS:
        lobe = DK_LOBE_MAP.get(region)
        if lobe is None:
            raise ValueError(f"DK region '{region}' missing from DK_LOBE_MAP — edit the map.")
        rows.append({
            "dk_region":         region,
            "glasser_lobe":      lobe,
            "hcp_thck_col_L":   f"FS_L_{region}_Thck",
            "hcp_thck_col_R":   f"FS_R_{region}_Thck",
            "hcp_area_col_L":   f"FS_L_{region}_Area",
            "hcp_area_col_R":   f"FS_R_{region}_Area",
        })
    return pd.DataFrame(rows)


def build_glasser_table() -> pd.DataFrame:
    """
    Load HCP-MMP1_UniqueRegionList.csv, map the 5-code Lobe field to readable
    names, flag which Glasser parcels correspond to cingulate cortices.
    """
    if not ATLAS_CSV.exists():
        raise FileNotFoundError(f"Cannot find {ATLAS_CSV}")
    df = pd.read_csv(ATLAS_CSV)

    unknown_lobes = set(df["Lobe"].unique()) - set(GLASSER_LOBE_CODE.keys())
    if unknown_lobes:
        raise ValueError(
            f"Unexpected Lobe codes in Glasser CSV: {unknown_lobes}. "
            "Update GLASSER_LOBE_CODE in 07_atlas_mapping.py."
        )

    df["broad_lobe"] = df["Lobe"].map(GLASSER_LOBE_CODE)
    df["is_cingulate"] = df["cortex"].isin(GLASSER_CINGULATE_CORTICES.keys())

    return df[["regionName", "regionLongName", "region", "LR",
               "Lobe", "broad_lobe", "cortex", "is_cingulate", "regionID"]]


def build_cross_dataset_summary(dk_table: pd.DataFrame,
                                 glasser_table: pd.DataFrame) -> pd.DataFrame:
    """
    Per-lobe correspondence table:
      - DK ROI count per lobe (both hemispheres combined)
      - Glasser ROI count per lobe (LH only, since parcels are symmetric)
      - Glasser cortex names included in each lobe
      - Methodological note on atlas differences
    """
    dk_counts = dk_table.groupby("glasser_lobe").size().rename("dk_n_rois_per_hemi")
    dk_counts = dk_counts.reset_index().rename(columns={"glasser_lobe": "lobe"})

    glasser_lh = glasser_table[glasser_table["LR"] == "L"]
    gl_counts = (glasser_lh.groupby("broad_lobe")
                 .size().rename("glasser_n_rois_lh").reset_index()
                 .rename(columns={"broad_lobe": "lobe"}))

    gl_cingulate_note = (glasser_lh.groupby("broad_lobe")[["is_cingulate", "cortex"]]
                         .apply(lambda x: "; ".join(sorted(x.loc[x["is_cingulate"], "cortex"].unique())),
                                include_groups=False)
                         .reset_index()
                         .rename(columns={"broad_lobe": "lobe", 0: "glasser_cingulate_cortices_included"}))

    gl_cortices = (glasser_lh.groupby("broad_lobe")["cortex"]
                   .apply(lambda x: "; ".join(sorted(x.unique())))
                   .reset_index()
                   .rename(columns={"broad_lobe": "lobe", "cortex": "glasser_cortex_names"}))

    summary = (dk_counts
               .merge(gl_counts,           on="lobe", how="outer")
               .merge(gl_cingulate_note,   on="lobe", how="left")
               .merge(gl_cortices,         on="lobe", how="left")
               .sort_values("lobe"))

    summary["cross_atlas_note"] = (
        "Lobe-level comparison only. DK (34 ROIs/hemi) uses macro-anatomical lobe "
        "boundaries. Glasser (180 ROIs/hemi) uses fine cytoarchitectonic parcels. "
        "DK cingulate ROIs are redistributed to Frontal (anterior/mid) and Parietal "
        "(posterior/isthmus) to match Glasser lobe conventions."
    )
    return summary


def build_glasser_cortex_summary(glasser_table: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cortex summary for the Glasser atlas (LH), showing parcel counts
    and their broad lobe assignment. Useful for methods section reporting.
    """
    lh = glasser_table[glasser_table["LR"] == "L"]
    summary = (lh.groupby(["broad_lobe", "cortex"])
                 .agg(n_parcels=("regionName", "count"))
                 .reset_index()
                 .sort_values(["broad_lobe", "cortex"]))
    return summary


def main() -> None:
    ensure_dirs()

    print("=" * 60)
    print("07 — Atlas Mapping (DK ↔ Glasser)")
    print("=" * 60)

    print("\n[1/4] Building DK region table …")
    dk_table = build_dk_table()
    lobe_dist = dk_table.groupby("glasser_lobe").size().to_dict()
    print(f"      {len(dk_table)} DK regions → Glasser lobes: {lobe_dist}")

    print("\n[2/4] Loading and processing Glasser atlas …")
    glasser_table = build_glasser_table()
    lh_only = glasser_table[glasser_table["LR"] == "L"]
    lobe_dist_gl = lh_only.groupby("broad_lobe").size().to_dict()
    n_cing = lh_only["is_cingulate"].sum()
    print(f"      {len(glasser_table)} total entries ({len(lh_only)} LH)")
    print(f"      Glasser lobe distribution (LH): {lobe_dist_gl}")
    print(f"      Glasser cingulate parcels (LH): {n_cing} "
          f"(distributed across Frontal and Parietal lobes)")

    print("\n[3/4] Building cortex-level Glasser summary …")
    cortex_summary = build_glasser_cortex_summary(glasser_table)

    print("\n[4/4] Saving tables …")
    dk_path = TABLES_DIR / "atlas_dk_to_glasser_map.csv"
    dk_table.to_csv(dk_path, index=False)
    print(f"      DK map → {dk_path.name}")

    gl_path = TABLES_DIR / "atlas_glasser_regions.csv"
    glasser_table.to_csv(gl_path, index=False)
    print(f"      Glasser regions → {gl_path.name}")

    summary = build_cross_dataset_summary(dk_table, glasser_table)
    sum_path = TABLES_DIR / "atlas_cross_dataset_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"      Cross-dataset summary → {sum_path.name}")

    cx_path = TABLES_DIR / "atlas_glasser_cortex_summary.csv"
    cortex_summary.to_csv(cx_path, index=False)
    print(f"      Glasser cortex summary → {cx_path.name}")

    print("\n--- Cross-dataset lobe correspondence (LH ROI counts) ---")
    print(f"{'Lobe':<12} {'DK ROIs':>10} {'Glasser ROIs':>14}  Glasser cingulate cortices")
    print("-" * 72)
    for _, row in summary.iterrows():
        cing = row.get("glasser_cingulate_cortices_included", "") or ""
        gl_n = int(row["glasser_n_rois_lh"]) if pd.notna(row.get("glasser_n_rois_lh")) else "—"
        dk_n = int(row["dk_n_rois_per_hemi"]) if pd.notna(row.get("dk_n_rois_per_hemi")) else "—"
        print(f"  {row['lobe']:<10} {dk_n:>10} {gl_n:>14}  {cing}")

    print("\n✓ Atlas mapping complete.\n")


if __name__ == "__main__":
    main()
