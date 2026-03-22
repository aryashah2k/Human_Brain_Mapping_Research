"""
08_report_tables.py
-------------------
Assemble unified APA-format summary tables from the outputs of all previous
analysis scripts. Produces one master Excel workbook with one sheet per
analysis, and individual publication-ready CSV files.

Outputs:
  results/tables/VBM_Analysis_Summary.xlsx    (multi-sheet workbook)

Run from project root:
    python scripts/08_report_tables.py
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import TABLES_DIR, ensure_dirs

# Expected input files (produced by earlier scripts)
INPUT_FILES = {
    "Sleep (all ROIs)":           "hcp_sleep_ttest_all.csv",
    "Sleep (sig. ROIs)":          "hcp_sleep_ttest_significant.csv",
    "Cognition (all ROIs)":       "hcp_cognition_corr_all.csv",
    "Cognition (sig. ROIs)":      "hcp_cognition_corr_significant.csv",
    "Neuroticism Thck (all)":     "hcp_neuroticism_thck_all.csv",
    "Neuroticism Thck (sig.)":    "hcp_neuroticism_thck_significant.csv",
    "Neuroticism Area (all)":     "hcp_neuroticism_area_all.csv",
    "Neuroticism Area (sig.)":    "hcp_neuroticism_area_significant.csv",
    "AABC Thick Corr (all)":      "aabc_aging_thick_corr_all.csv",
    "AABC Thick Corr (sig.)":     "aabc_aging_thick_corr_significant.csv",
    "AABC Vol Corr (all)":        "aabc_aging_vol_corr_all.csv",
    "AABC Vol Corr (sig.)":       "aabc_aging_vol_corr_significant.csv",
    "AABC ANOVA (thick)":         "aabc_aging_anova_thick.csv",
    "AABC ANOVA (vol)":           "aabc_aging_anova_vol.csv",
    "Atlas DK→Glasser map":       "atlas_dk_to_glasser_map.csv",
    "Atlas cross-dataset summary":"atlas_cross_dataset_summary.csv",
    "AABC Sleep (all ROIs)":      "aabc_sleep_ttest_all.csv",
    "AABC Sleep (sig. ROIs)":     "aabc_sleep_ttest_significant.csv",
    "AABC Neuro Thck (all)":      "aabc_neuroticism_thck_all.csv",
    "AABC Neuro Thck (sig.)":     "aabc_neuroticism_thck_significant.csv",
}

XLSX_PATH = TABLES_DIR / "VBM_Analysis_Summary.xlsx"


def add_method_note(sheet_name: str) -> str:
    notes = {
        "Sleep (all ROIs)":
            "Welch t-test (good sleepers PSQI≤5 vs. poor sleepers PSQI>5). "
            "FDR correction: Benjamini–Hochberg (q<.05). Effect size: Cohen's d.",
        "Cognition (all ROIs)":
            "Pearson r (PMAT24_A_CR × DK surface area). "
            "FDR correction: Benjamini–Hochberg (q<.05). 95% CI via Fisher z-transform.",
        "Neuroticism Thck (all)":
            "Welch t-test (bottom vs. top NEOFAC_N tertile). "
            "FDR correction: Benjamini–Hochberg (q<.05). Effect size: Cohen's d.",
        "AABC Thick Corr (all)":
            "Pearson r (age_open × cortical thickness). "
            "FDR correction: Benjamini–Hochberg (q<.05). 95% CI via Fisher z-transform.",
        "AABC ANOVA (thick)":
            "One-way ANOVA (age decade) on FDR-significant ROIs. Post-hoc: Tukey HSD.",
        "Atlas DK→Glasser map":
            "Lobe classification based on Desikan et al. (2006) and Glasser et al. (2016). "
            "Cross-atlas comparison is at lobe level only.",
    }
    for key, note in notes.items():
        if sheet_name.startswith(key.split("(")[0].strip()):
            return note
    return ""


def main() -> None:
    ensure_dirs()

    print("=" * 60)
    print("08 — Report Tables (Master Excel Workbook)")
    print("=" * 60)

    missing = []
    for label, fname in INPUT_FILES.items():
        path = TABLES_DIR / fname
        if not path.exists():
            missing.append(fname)

    if missing:
        print(f"\n  ⚠  {len(missing)} table(s) not yet generated:")
        for f in missing:
            print(f"      • {f}")
        print(
            "\n  This script must be run AFTER all analysis scripts (01–07).\n"
            "  Missing sheets will be skipped; available sheets will still be written."
        )

    print(f"\nWriting {XLSX_PATH.name} …")

    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as writer:
        for label, fname in INPUT_FILES.items():
            path = TABLES_DIR / fname
            if not path.exists():
                print(f"  SKIP  {label} ({fname} not found)")
                continue

            df = pd.read_csv(path)

            # Truncate sheet name to Excel's 31-char limit
            sheet = label[:31]

            # Add method note as first row if available
            note = add_method_note(label)
            if note:
                meta = pd.DataFrame([{"Method note": note}])
                meta.to_excel(writer, sheet_name=sheet, index=False, startrow=0)
                df.to_excel(writer, sheet_name=sheet, index=False, startrow=2)
            else:
                df.to_excel(writer, sheet_name=sheet, index=False)

            print(f"  ✓  {sheet} ({len(df)} rows)")

    print(f"\nWorkbook saved → {XLSX_PATH}")
    print("\n✓ Report tables complete.\n")


if __name__ == "__main__":
    main()
