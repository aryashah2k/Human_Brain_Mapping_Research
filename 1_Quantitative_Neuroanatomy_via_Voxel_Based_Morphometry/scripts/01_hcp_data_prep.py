"""
01_hcp_data_prep.py
-------------------
Load HCP_YA_subjects_2026.csv, validate all required columns, exclude subjects
with missing values in any analysis variable, and write a clean parquet file
to results/interim/hcp_clean.parquet.

Run from the project root (hbm1/):
    python scripts/01_hcp_data_prep.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Make sure scripts/ on the path so utils imports cleanly regardless of cwd
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    HCP_CSV, HCP_PARQUET, ensure_dirs, TABLES_DIR
)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
BEHAV_COLS = ["PSQI_Score", "PMAT24_A_CR", "NEOFAC_N",
              "NEOFAC_A", "NEOFAC_O", "NEOFAC_C", "NEOFAC_E"]

DK_REGIONS = [
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

THCK_COLS = ([f"FS_L_{r}_Thck" for r in DK_REGIONS] +
             [f"FS_R_{r}_Thck" for r in DK_REGIONS])

AREA_COLS = ([f"FS_L_{r}_Area" for r in DK_REGIONS] +
             [f"FS_R_{r}_Area" for r in DK_REGIONS])

STRUCT_COLS = THCK_COLS + AREA_COLS + ["FS_IntraCranial_Vol"]

ID_COLS = ["Subject", "Gender", "Age"]

ALL_REQUIRED = ID_COLS + BEHAV_COLS + STRUCT_COLS


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required columns in HCP CSV.\n"
            f"First 5 missing: {missing[:5]}"
        )


def main() -> None:
    ensure_dirs()

    print("=" * 60)
    print("01 — HCP Data Preparation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load raw CSV
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading {HCP_CSV.name} …")
    if not HCP_CSV.exists():
        raise FileNotFoundError(f"Cannot find {HCP_CSV}")
    df_raw = pd.read_csv(HCP_CSV, low_memory=False)
    print(f"      Loaded {len(df_raw):,} rows × {len(df_raw.columns):,} columns")

    # ------------------------------------------------------------------
    # 2. Validate columns
    # ------------------------------------------------------------------
    print("\n[2/5] Validating required columns …")
    validate_columns(df_raw, ALL_REQUIRED)
    print(f"      All {len(ALL_REQUIRED)} required columns present ✓")

    # ------------------------------------------------------------------
    # 3. Subset and coerce types
    # ------------------------------------------------------------------
    print("\n[3/5] Subsetting and coercing types …")
    df = df_raw[ALL_REQUIRED].copy()

    # Coerce numeric — anything non-parseable becomes NaN
    num_cols = BEHAV_COLS + STRUCT_COLS
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Gender: keep only M/F, recode to 0/1
    df = df[df["Gender"].isin(["M", "F"])].copy()
    df["sex_binary"] = (df["Gender"] == "F").astype(int)

    # Age: HCP stores as bin strings like "22-25" → use midpoint
    age_midpoints = {
        "22-25": 23.5, "26-30": 28.0, "31-35": 33.0, "36+": 38.0,
    }
    df["age_mid"] = df["Age"].map(age_midpoints)
    if df["age_mid"].isna().any():
        unknown = df.loc[df["age_mid"].isna(), "Age"].unique()
        raise ValueError(f"Unexpected Age bin values: {unknown}")

    # ------------------------------------------------------------------
    # 4. Exclude subjects with any missing value in analysis variables
    # ------------------------------------------------------------------
    print("\n[4/5] Excluding subjects with missing values …")
    analysis_cols = BEHAV_COLS + STRUCT_COLS
    n_before = len(df)
    df = df.dropna(subset=analysis_cols)
    n_after = len(df)
    n_dropped = n_before - n_after
    print(f"      Excluded {n_dropped:,} subjects with ≥1 missing value")
    print(f"      Retained {n_after:,} subjects")

    # Group size sanity checks
    good_sleepers = (df["PSQI_Score"] <= 5).sum()
    poor_sleepers = (df["PSQI_Score"] > 5).sum()
    print(f"\n      PSQI groups: good ≤5 n={good_sleepers}, poor >5 n={poor_sleepers}")
    if good_sleepers < 20 or poor_sleepers < 20:
        raise RuntimeError("PSQI group sizes too small (< 20). Check data.")

    n1_lo = (df["NEOFAC_N"] <= df["NEOFAC_N"].quantile(1/3)).sum()
    n1_hi = (df["NEOFAC_N"] >= df["NEOFAC_N"].quantile(2/3)).sum()
    print(f"      NEOFAC_N tertiles: bottom n={n1_lo}, top n={n1_hi}")
    if n1_lo < 20 or n1_hi < 20:
        raise RuntimeError("Neuroticism tertile sizes too small. Check data.")

    # ------------------------------------------------------------------
    # 5. Write clean parquet + summary
    # ------------------------------------------------------------------
    print(f"\n[5/5] Writing clean dataset to {HCP_PARQUET.relative_to(HCP_PARQUET.parents[3])} …")
    df.reset_index(drop=True).to_parquet(HCP_PARQUET, index=False)

    # Write column manifest for reference
    manifest_path = TABLES_DIR / "hcp_column_manifest.txt"
    with open(manifest_path, "w") as f:
        for col in df.columns:
            f.write(col + "\n")
    print(f"      Column manifest → {manifest_path.name}")

    # Summary stats for key variables
    summary = df[["PSQI_Score", "PMAT24_A_CR", "NEOFAC_N",
                  "age_mid"]].describe().round(2)
    print("\n--- Key variable summary ---")
    print(summary.to_string())

    print("\n✓ HCP data preparation complete.\n")


if __name__ == "__main__":
    main()
