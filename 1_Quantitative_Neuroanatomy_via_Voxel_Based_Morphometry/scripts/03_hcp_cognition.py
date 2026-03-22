"""
03_hcp_cognition.py
-------------------
Analysis 2: Fluid intelligence (PMAT24_A_CR) × cortical surface area.
Statistics: Pearson correlation per DK ROI, FDR-BH, 95% CI via Fisher z.
Outputs:
  results/tables/hcp_cognition_corr_all.csv
  results/tables/hcp_cognition_corr_significant.csv
  results/figures/hcp/cognition_scatter_top<N>.pdf/.png  (top-5 by |r|)

Run from project root:
    python scripts/03_hcp_cognition.py
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    HCP_PARQUET, TABLES_DIR, ensure_dirs, set_pub_style, save_fig,
    pearson_ci, apa_corr_row, PALETTE_CORR, DK_REGIONS,
)

AREA_COLS = ([f"FS_L_{r}_Area" for r in DK_REGIONS] +
             [f"FS_R_{r}_Area" for r in DK_REGIONS])

FDR_ALPHA   = 0.05
TOP_N       = 5      # number of scatter plots to produce
SUBDIR      = "hcp"


def _pretty_roi(col: str) -> str:
    parts = col.split("_")   # ['FS', 'L', 'Region', 'Area']
    return f"{parts[1]} {parts[2]}"


def run_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in AREA_COLS:
        valid = df[["PMAT24_A_CR", col]].dropna()
        n = len(valid)
        r, pval = stats.pearsonr(valid["PMAT24_A_CR"], valid[col])
        ci_lo, ci_hi = pearson_ci(r, n)
        rows.append({
            "roi_col":   col,
            "roi_label": _pretty_roi(col),
            "n":         n,
            "r":         r,
            "p":         pval,
            "r_ci_lo":   ci_lo,
            "r_ci_hi":   ci_hi,
        })

    results = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(results["p"].values, alpha=FDR_ALPHA, method="fdr_bh")
    results["p_fdr"]   = p_fdr
    results["sig_fdr"] = reject
    return results


def plot_scatter_grid(df: pd.DataFrame, results: pd.DataFrame) -> plt.Figure:
    """
    Grid of scatter plots for the top-N ROIs sorted by |r|.
    PMAT score on x-axis, surface area on y-axis.
    """
    top = results.nlargest(TOP_N, "r", keep="all").reset_index(drop=True)
    # Sort by |r|
    top = results.assign(abs_r=results["r"].abs()).nlargest(TOP_N, "abs_r").reset_index(drop=True)

    set_pub_style()
    ncols = min(TOP_N, 5)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.5), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, top.iterrows()):
        col = row["roi_col"]
        valid = df[["PMAT24_A_CR", col]].dropna()
        x_vals = valid["PMAT24_A_CR"].values
        y_vals = valid[col].values

        ax.scatter(x_vals, y_vals, alpha=0.25, s=12,
                   color=PALETTE_CORR, linewidths=0)

        # Regression line
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.plot(x_line, m * x_line + b, color="black", lw=1.5)

        r_str = f"r = {row['r']:.3f}"
        ci_str = f"95% CI [{row['r_ci_lo']:.3f}, {row['r_ci_hi']:.3f}]"
        p_str = "p < .001" if row["p_fdr"] < 0.001 else f"p = {row['p_fdr']:.3f}"
        ax.set_title(f"{row['roi_label']}\n{r_str}, {p_str}", fontsize=10)
        ax.set_xlabel("PMAT24 correct responses", fontsize=9)
        ax.set_ylabel("Surface area (mm²)", fontsize=9)
        ax.annotate(f"n = {row['n']}", xy=(0.05, 0.93), xycoords="axes fraction",
                    fontsize=8, color="gray")

    fig.suptitle(
        f"Fluid intelligence × cortical surface area\nTop {TOP_N} ROIs by |r| (FDR-corrected)",
        y=1.02, fontsize=12)
    fig.tight_layout()
    return fig


def main() -> None:
    ensure_dirs()
    set_pub_style()

    print("=" * 60)
    print("03 — HCP Cognition × Cortical Surface Area")
    print("=" * 60)

    print(f"\n[1/4] Loading {HCP_PARQUET.name} …")
    if not HCP_PARQUET.exists():
        raise FileNotFoundError("Run 01_hcp_data_prep.py first.")
    df = pd.read_parquet(HCP_PARQUET)
    print(f"      N = {len(df):,} subjects")

    print("\n[2/4] Computing Pearson correlations (PMAT24 × 68 area ROIs) …")
    results = run_correlations(df)
    n_sig = results["sig_fdr"].sum()
    print(f"      FDR-significant ROIs (q < {FDR_ALPHA}): {n_sig} / {len(results)}")

    print("\n[3/4] Saving tables …")
    apa_rows = [
        apa_corr_row(
            roi=row["roi_label"], r=row["r"], p=row["p"], n=int(row["n"]),
            ci_lo=row["r_ci_lo"], ci_hi=row["r_ci_hi"],
            p_fdr=row["p_fdr"], sig_fdr=row["sig_fdr"]
        )
        for _, row in results.iterrows()
    ]
    apa_df = pd.DataFrame(apa_rows)
    all_path = TABLES_DIR / "hcp_cognition_corr_all.csv"
    apa_df.to_csv(all_path, index=False)
    print(f"      Full table → {all_path.name}")

    sig_df = apa_df[results["sig_fdr"].values].copy()
    sig_path = TABLES_DIR / "hcp_cognition_corr_significant.csv"
    sig_df.to_csv(sig_path, index=False)
    print(f"      Significant table → {sig_path.name}")

    if not sig_df.empty:
        print("\n      FDR-significant ROIs (top 10 by |r|):")
        top10 = results[results["sig_fdr"]].assign(
            abs_r=lambda x: x["r"].abs()).nlargest(10, "abs_r")
        print(top10[["roi_label", "r", "p", "p_fdr"]].to_string(index=False))

    print("\n[4/4] Generating scatter plots …")
    fig = plot_scatter_grid(df, results)
    save_fig(fig, f"cognition_scatter_top{TOP_N}", subdir=SUBDIR)
    print(f"      Scatter grid → results/figures/{SUBDIR}/cognition_scatter_top{TOP_N}.pdf/.png")

    print("\n✓ Cognition analysis complete.\n")


if __name__ == "__main__":
    main()
