"""
05_aabc_data_prep.py
--------------------
Load AABC_Demographics.csv, AABC_Release2_Non-imaging_Data-XL.csv, and
structural CSVs (thickness, volume, surface area, myelin).
Filter to baseline visit (_V1), merge, validate, and write to
results/interim/aabc_clean.parquet.

Subject ID notes (verified from actual files):
  - All files use 'id_event' column holding 'HCA6000030_V1' format.
  - Structural CSVs: first column is named 'x___' — handled automatically.
  - Non-imaging XL: header row 0 = descriptions; actual data starts at row 1.
  - Merge key: full 'id_event' string (including _V1 suffix).

Behavioral columns in XL file (pre-computed):
  - psqi_global : PSQI global score (0–21; >5 = poor sleep quality)
  - neo_n       : NEO-FFI Neuroticism factor score
  - neo_a, neo_c, neo_e, neo_o: other NEO domains

Run from project root:
    python scripts/05_aabc_data_prep.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    AABC_DEMO_CSV, AABC_THICK_CSV, AABC_VOL_CSV, AABC_AREA_CSV,
    AABC_MYELIN_CSV, AABC_XL_CSV, AABC_PARQUET, TABLES_DIR, ensure_dirs,
)

VISIT_SUFFIX = "_V1"

# Behavioral columns we need from the XL file
XL_BEHAV_COLS = [
    "psqi_global",   # PSQI total score (0–21)
    "neo_n",         # Neuroticism
    "neo_a",         # Agreeableness
    "neo_c",         # Conscientiousness
    "neo_e",         # Extraversion
    "neo_o",         # Openness
]


def _load_structural(csv_path: Path, label: str) -> pd.DataFrame:
    """
    Load a structural CSV. The first column (named 'x___') holds id_event values.
    Prefix all ROI columns with `label_` for disambiguation.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing AABC structural file: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    raw_id_col = df.columns[0]
    df = df.rename(columns={raw_id_col: "id_event"})
    rename = {c: f"{label}_{c}" for c in df.columns if c != "id_event"}
    df = df.rename(columns=rename)
    value_cols = [c for c in df.columns if c != "id_event"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    return df


def _load_xl_behavioral(xl_path: Path) -> pd.DataFrame:
    """
    Load the non-imaging XL CSV.
    Row 0 in the raw file is a label/description row — use header=1
    so the actual column names are read from row 1.
    Filter to _V1 rows, extract id_event + behavioral summary columns.
    """
    if not xl_path.exists():
        raise FileNotFoundError(f"Cannot find {xl_path}")
    df = pd.read_csv(xl_path, low_memory=False, header=1)

    if "id_event" not in df.columns:
        raise ValueError("'id_event' column not found in XL file after header=1 read.")

    # Filter to baseline
    df_v1 = df[df["id_event"].astype(str).str.endswith(VISIT_SUFFIX)].copy()

    # Verify required columns present
    missing = [c for c in XL_BEHAV_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"These behavioral columns not found in XL file: {missing}\n"
            f"Available PSQI cols: {[c for c in df.columns if 'psqi' in c.lower()][:10]}"
        )

    cols = ["id_event"] + XL_BEHAV_COLS
    df_v1 = df_v1[cols].copy()
    df_v1[XL_BEHAV_COLS] = df_v1[XL_BEHAV_COLS].apply(pd.to_numeric, errors="coerce")
    return df_v1


def main() -> None:
    ensure_dirs()

    print("=" * 60)
    print("05 — AABC Data Preparation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load demographics
    # ------------------------------------------------------------------
    print(f"\n[1/7] Loading demographics …")
    if not AABC_DEMO_CSV.exists():
        raise FileNotFoundError(f"Cannot find {AABC_DEMO_CSV}")
    demo = pd.read_csv(AABC_DEMO_CSV, low_memory=False)
    print(f"      {len(demo):,} rows from {AABC_DEMO_CSV.name}")

    for col in ["id_event", "age_open", "sex"]:
        if col not in demo.columns:
            raise ValueError(f"Column '{col}' missing from demographics.")

    # ------------------------------------------------------------------
    # 2. Filter to V1
    # ------------------------------------------------------------------
    print(f"\n[2/7] Filtering to baseline visit ({VISIT_SUFFIX}) …")
    demo_v1 = demo[demo["id_event"].astype(str).str.endswith(VISIT_SUFFIX)].copy()
    print(f"      Retained {len(demo_v1):,} baseline rows")

    if len(demo_v1) < 50:
        raise RuntimeError(f"Too few V1 rows ({len(demo_v1)}). Check id_event format.")

    demo_v1["age_open"] = pd.to_numeric(demo_v1["age_open"], errors="coerce")

    # Sex recoding: AABC uses 'F'/'M' (verified from Release 2)
    sex_vals = demo_v1["sex"].astype(str).str.strip().str.upper()
    demo_v1["sex_binary"] = (sex_vals == "F").astype(int)
    pct_f = demo_v1["sex_binary"].mean() * 100
    if pct_f < 1:
        raise ValueError(
            f"Sex recoding yielded {pct_f:.1f}% female. "
            f"Unique values: {demo_v1['sex'].unique()[:5].tolist()}"
        )

    demo_v1 = demo_v1[["id_event", "age_open", "sex", "sex_binary"]].copy()

    # ------------------------------------------------------------------
    # 3. Load non-imaging behavioral data
    # ------------------------------------------------------------------
    print(f"\n[3/7] Loading non-imaging XL behavioral data …")
    xl_behav = _load_xl_behavioral(AABC_XL_CSV)
    print(f"      {len(xl_behav):,} _V1 rows with behavioral variables")
    print(f"      Columns: {XL_BEHAV_COLS}")

    # Summary
    for col in XL_BEHAV_COLS:
        n_valid = xl_behav[col].notna().sum()
        print(f"      {col}: {n_valid} / {len(xl_behav)} non-missing "
              f"(range {xl_behav[col].min():.0f}–{xl_behav[col].max():.0f})")

    # ------------------------------------------------------------------
    # 4. Load structural CSVs
    # ------------------------------------------------------------------
    print("\n[4/7] Loading structural CSVs …")
    thick  = _load_structural(AABC_THICK_CSV,  "thick")
    vol    = _load_structural(AABC_VOL_CSV,    "vol")
    area   = _load_structural(AABC_AREA_CSV,   "area")
    myelin = _load_structural(AABC_MYELIN_CSV, "myelin")

    for name, df_ in [("thickness", thick), ("volume", vol),
                      ("area", area), ("myelin", myelin)]:
        v1_n = (df_["id_event"].str.endswith(VISIT_SUFFIX)).sum()
        print(f"      {name:10s}: {v1_n:,} _V1 rows, {len(df_.columns) - 1} ROI cols")

    # ------------------------------------------------------------------
    # 5. Filter structural CSVs to V1 and merge everything
    # ------------------------------------------------------------------
    print("\n[5/7] Filtering structural CSVs to _V1 and merging …")
    thick_v1  = thick[thick["id_event"].str.endswith(VISIT_SUFFIX)].copy()
    vol_v1    = vol[vol["id_event"].str.endswith(VISIT_SUFFIX)].copy()
    area_v1   = area[area["id_event"].str.endswith(VISIT_SUFFIX)].copy()
    myelin_v1 = myelin[myelin["id_event"].str.endswith(VISIT_SUFFIX)].copy()

    merged = demo_v1.merge(xl_behav,  on="id_event", how="inner")
    print(f"      After demo × behavioral merge:  {len(merged):,}")
    merged = merged.merge(thick_v1,   on="id_event", how="inner")
    print(f"      After thickness merge:           {len(merged):,}")
    merged = merged.merge(vol_v1,     on="id_event", how="inner")
    merged = merged.merge(area_v1,    on="id_event", how="inner")
    merged = merged.merge(myelin_v1,  on="id_event", how="inner")
    print(f"      After all structural merges:     {len(merged):,}")

    if len(merged) < 50:
        raise RuntimeError(
            f"Only {len(merged)} subjects after full merge. "
            "Check subject ID formats across all files."
        )

    # ------------------------------------------------------------------
    # 6. Drop subjects missing key variables
    # ------------------------------------------------------------------
    print("\n[6/7] Dropping subjects with missing age, sex, or behavioral scores …")
    required_non_null = ["age_open", "sex_binary"] + XL_BEHAV_COLS
    n_before = len(merged)
    merged = merged.dropna(subset=required_non_null)
    print(f"      Dropped {n_before - len(merged):,} → retained {len(merged):,}")

    print(f"\n      Age:  mean={merged['age_open'].mean():.1f},  "
          f"SD={merged['age_open'].std():.1f},  "
          f"range=[{merged['age_open'].min():.0f}, {merged['age_open'].max():.0f}]")
    print(f"      Sex:  {merged['sex_binary'].mean()*100:.1f}% female")

    good_sleep = (merged["psqi_global"] <= 5).sum()
    poor_sleep = (merged["psqi_global"] > 5).sum()
    print(f"      PSQI: good ≤5 n={good_sleep}, poor >5 n={poor_sleep}")
    n_lo = (merged["neo_n"] <= merged["neo_n"].quantile(1/3)).sum()
    n_hi = (merged["neo_n"] >= merged["neo_n"].quantile(2/3)).sum()
    print(f"      NEO-N tertiles: bottom n={n_lo}, top n={n_hi}")

    # ------------------------------------------------------------------
    # 7. Write parquet + manifest
    # ------------------------------------------------------------------
    print(f"\n[7/7] Writing {AABC_PARQUET.name} …")
    merged.reset_index(drop=True).to_parquet(AABC_PARQUET, index=False)

    manifest_path = TABLES_DIR / "aabc_column_manifest.txt"
    with open(manifest_path, "w") as f:
        for col in merged.columns:
            f.write(col + "\n")
    n_thick = len([c for c in merged.columns if c.startswith("thick_")])
    n_vol   = len([c for c in merged.columns if c.startswith("vol_")])
    print(f"      {len(merged.columns):,} total columns "
          f"({n_thick} thick, {n_vol} vol, + behavioral + demo)")
    print(f"      Column manifest → {manifest_path.name}")

    print("\n✓ AABC data preparation complete.\n")


if __name__ == "__main__":
    main()
