"""
06_aabc_aging.py
----------------
Analysis 4: Age × cortical structure in AABC.
Uses thickness and volume for all Glasser HCP-MMP1 ROIs (from merged parquet).
Statistics:
  - Pearson correlation (age × each ROI) with FDR-BH correction
  - One-way ANOVA across age-decade bins, with Tukey HSD post-hoc
Outputs:
  results/tables/aabc_aging_thick_corr_all.csv
  results/tables/aabc_aging_thick_corr_significant.csv
  results/tables/aabc_aging_vol_corr_all.csv
  results/tables/aabc_aging_vol_corr_significant.csv
  results/tables/aabc_aging_anova_thick.csv
  results/tables/aabc_aging_anova_vol.csv
  results/figures/aabc/aging_heatmap_thick.pdf/.png
  results/figures/aabc/aging_heatmap_vol.pdf/.png
  results/figures/aabc/aging_scatter_top5_thick.pdf/.png
  results/figures/aabc/aging_scatter_top5_vol.pdf/.png

Run from project root:
    python scripts/06_aabc_aging.py
"""

import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    AABC_PARQUET, TABLES_DIR, ensure_dirs, set_pub_style, save_fig,
    pearson_ci, apa_corr_row, PALETTE_CORR, FIG_DPI,
)

FDR_ALPHA = 0.05
TOP_N     = 5
SUBDIR    = "aabc"


def _get_roi_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    """Return all columns starting with `prefix` (e.g. 'thick_', 'vol_')."""
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    if not cols:
        raise ValueError(f"No columns with prefix '{prefix}_' found in parquet. "
                         "Check 05_aabc_data_prep.py output.")
    return cols


def _pretty_roi(col: str, prefix: str) -> str:
    """Strip prefix: 'thick_L_V1' → 'L_V1'."""
    return col.removeprefix(prefix + "_")


def run_correlations(df: pd.DataFrame, roi_cols: list[str],
                     prefix: str) -> pd.DataFrame:
    rows = []
    for col in roi_cols:
        valid = df[["age_open", col]].dropna()
        n = len(valid)
        if n < 10:
            continue
        r, pval = stats.pearsonr(valid["age_open"], valid[col])
        ci_lo, ci_hi = pearson_ci(r, n)
        rows.append({
            "roi_col":   col,
            "roi_label": _pretty_roi(col, prefix),
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


def run_anova(df: pd.DataFrame, roi_cols: list[str], prefix: str,
              max_rois: int = 20) -> pd.DataFrame:
    """
    One-way ANOVA (age decade) for up to max_rois ROIs that are FDR-significant,
    plus Tukey HSD post-hoc. Returns summary DataFrame.
    """
    df = df.copy()
    df["age_decade"] = (df["age_open"] // 10 * 10).astype(int).astype(str) + "s"

    corr = run_correlations(df, roi_cols, prefix)
    sig_cols = corr.loc[corr["sig_fdr"], "roi_col"].tolist()[:max_rois]

    rows = []
    for col in sig_cols:
        valid = df[["age_decade", col]].dropna()
        groups = [grp[col].values for _, grp in valid.groupby("age_decade")]
        if len(groups) < 2:
            continue
        F, pval = stats.f_oneway(*groups)
        # Tukey HSD post-hoc: count how many pairwise comparisons are significant
        tukey = pairwise_tukeyhsd(endog=valid[col], groups=valid["age_decade"], alpha=0.05)
        n_sig_pairs = int(np.sum(tukey.reject))
        rows.append({
            "roi":          _pretty_roi(col, prefix),
            "F":            round(F, 3),
            "p_anova":      round(pval, 4),
            "n_age_groups": len(groups),
            "n_sig_pairs":  n_sig_pairs,
        })

    return pd.DataFrame(rows)


def plot_heatmap(results: pd.DataFrame, title: str, stem: str) -> None:
    """
    Heatmap of Pearson r values across all Glasser ROIs (rows) sorted by r.
    Left column = left hemisphere, right column = right hemisphere.
    """
    # Split into L and R
    left  = results[results["roi_label"].str.startswith("L_")].copy()
    right = results[results["roi_label"].str.startswith("R_")].copy()

    for df_h, hemi_label in [(left, "LH"), (right, "RH")]:
        if df_h.empty:
            continue

        df_h = df_h.sort_values("r")
        r_vals = df_h["r"].values.reshape(-1, 1)
        labels = [lbl.removeprefix("L_").removeprefix("R_") for lbl in df_h["roi_label"]]

        set_pub_style()
        fig_h = max(6, len(labels) * 0.22)
        fig, ax = plt.subplots(figsize=(2.8, fig_h))
        cmap = plt.cm.RdBu_r
        norm = mcolors.TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6)
        im = ax.imshow(r_vals, aspect="auto", cmap=cmap, norm=norm)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xticks([])
        ax.set_xlabel("")

        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
        cbar.set_label("Pearson r (age)", fontsize=9)

        # Mark significant ROIs with a dot
        sig_mask = df_h["sig_fdr"].values
        sig_idx  = np.where(sig_mask)[0]
        ax.scatter(np.zeros(len(sig_idx)), sig_idx,
                   marker="*", color="black", s=20, zorder=3)

        ax.set_title(f"{title}\n{hemi_label} — * FDR q<.05", fontsize=10)
        fig.tight_layout()
        save_fig(fig, f"{stem}_{hemi_label.lower()}", subdir=SUBDIR)
        print(f"      Heatmap → results/figures/{SUBDIR}/{stem}_{hemi_label.lower()}.pdf/.png")


def plot_scatter_top(df: pd.DataFrame, results: pd.DataFrame,
                     prefix: str, stem: str, title: str) -> None:
    top = results.assign(abs_r=results["r"].abs()).nlargest(TOP_N, "abs_r").reset_index(drop=True)

    set_pub_style()
    fig, axes = plt.subplots(1, TOP_N, figsize=(4 * TOP_N, 4.5))
    if TOP_N == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, top.iterrows()):
        col   = row["roi_col"]
        valid = df[["age_open", col]].dropna()
        x_v   = valid["age_open"].values
        y_v   = valid[col].values

        ax.scatter(x_v, y_v, alpha=0.25, s=10, color=PALETTE_CORR, linewidths=0)
        m, b = np.polyfit(x_v, y_v, 1)
        x_line = np.linspace(x_v.min(), x_v.max(), 200)
        ax.plot(x_line, m * x_line + b, color="black", lw=1.5)

        r_str = f"r = {row['r']:.3f}"
        p_str = "p < .001" if row["p_fdr"] < 0.001 else f"p = {row['p_fdr']:.3f}"
        ax.set_title(f"{row['roi_label']}\n{r_str}, {p_str}", fontsize=9)
        ax.set_xlabel("Age (years)", fontsize=9)
        ax.set_ylabel("Thickness (mm)" if "thick" in prefix else "Volume (mm³)", fontsize=9)
        ax.annotate(f"n = {row['n']}", xy=(0.05, 0.93),
                    xycoords="axes fraction", fontsize=8, color="gray")

    fig.suptitle(f"{title}\nTop {TOP_N} ROIs by |r| (FDR-corrected)", fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, stem, subdir=SUBDIR)
    print(f"      Scatter → results/figures/{SUBDIR}/{stem}.pdf/.png")


def _save_tables(results: pd.DataFrame, name_all: str, name_sig: str) -> None:
    apa_rows = [
        apa_corr_row(
            roi=row["roi_label"], r=row["r"], p=row["p"], n=int(row["n"]),
            ci_lo=row["r_ci_lo"], ci_hi=row["r_ci_hi"],
            p_fdr=row["p_fdr"], sig_fdr=row["sig_fdr"]
        )
        for _, row in results.iterrows()
    ]
    apa_df = pd.DataFrame(apa_rows)
    apa_df.to_csv(TABLES_DIR / name_all, index=False)
    apa_df[results["sig_fdr"].values].to_csv(TABLES_DIR / name_sig, index=False)
    print(f"      → {name_all}  |  {name_sig}")


def main() -> None:
    ensure_dirs()
    set_pub_style()

    print("=" * 60)
    print("06 — AABC Age × Cortical Structure")
    print("=" * 60)

    print(f"\n[1/5] Loading {AABC_PARQUET.name} …")
    if not AABC_PARQUET.exists():
        raise FileNotFoundError("Run 05_aabc_data_prep.py first.")
    df = pd.read_parquet(AABC_PARQUET)
    print(f"      N = {len(df):,}   Age: {df['age_open'].min():.0f}–{df['age_open'].max():.0f} yrs")

    thick_cols = _get_roi_cols(df, "thick")
    vol_cols   = _get_roi_cols(df, "vol")
    print(f"      {len(thick_cols)} thickness ROIs, {len(vol_cols)} volume ROIs")

    print("\n[2/5] Pearson correlations (age × thickness) …")
    res_thick = run_correlations(df, thick_cols, "thick")
    print(f"      FDR-sig: {res_thick['sig_fdr'].sum()} / {len(res_thick)}")

    print("\n[3/5] Pearson correlations (age × volume) …")
    res_vol = run_correlations(df, vol_cols, "vol")
    print(f"      FDR-sig: {res_vol['sig_fdr'].sum()} / {len(res_vol)}")

    print("\n[4/5] Saving tables …")
    _save_tables(res_thick, "aabc_aging_thick_corr_all.csv", "aabc_aging_thick_corr_significant.csv")
    _save_tables(res_vol,   "aabc_aging_vol_corr_all.csv",   "aabc_aging_vol_corr_significant.csv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anova_thick = run_anova(df, thick_cols, "thick")
        anova_vol   = run_anova(df, vol_cols,   "vol")
    anova_thick.to_csv(TABLES_DIR / "aabc_aging_anova_thick.csv", index=False)
    anova_vol.to_csv(  TABLES_DIR / "aabc_aging_anova_vol.csv",   index=False)
    print("      → aabc_aging_anova_thick.csv  |  aabc_aging_anova_vol.csv")

    print("\n[5/5] Generating figures …")
    plot_heatmap(res_thick, "Age × Cortical Thickness (AABC)", "aging_heatmap_thick")
    plot_heatmap(res_vol,   "Age × Cortical Volume (AABC)",    "aging_heatmap_vol")
    plot_scatter_top(df, res_thick, "thick", "aging_scatter_top5_thick",
                     "Age × Cortical Thickness (AABC)")
    plot_scatter_top(df, res_vol,   "vol",   "aging_scatter_top5_vol",
                     "Age × Cortical Volume (AABC)")

    print("\n✓ AABC aging analysis complete.\n")


if __name__ == "__main__":
    main()
