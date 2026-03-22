"""
04_hcp_neuroticism.py
---------------------
Analysis 3: Neuroticism (NEOFAC_N) × cortical thickness and surface area.
Groups: top tertile (high neuroticism) vs. bottom tertile (low neuroticism).
Statistics: Welch t-test per DK ROI, FDR-BH, Cohen's d, 95% CI.
Outputs:
  results/tables/hcp_neuroticism_thck_all.csv
  results/tables/hcp_neuroticism_thck_significant.csv
  results/tables/hcp_neuroticism_area_all.csv
  results/tables/hcp_neuroticism_area_significant.csv
  results/figures/hcp/neuroticism_violin_significant.pdf/.png

Run from project root:
    python scripts/04_hcp_neuroticism.py
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
    cohens_d, apa_ttest_row, PALETTE_GROUP, DK_REGIONS,
)

THCK_COLS = ([f"FS_L_{r}_Thck" for r in DK_REGIONS] +
             [f"FS_R_{r}_Thck" for r in DK_REGIONS])
AREA_COLS = ([f"FS_L_{r}_Area" for r in DK_REGIONS] +
             [f"FS_R_{r}_Area" for r in DK_REGIONS])

FDR_ALPHA = 0.05
SUBDIR    = "hcp"
MAX_VIOLIN = 12   # cap violin panels to keep the figure legible


def _pretty_roi(col: str) -> str:
    parts = col.split("_")
    return f"{parts[1]} {parts[2]}"


def tertile_groups(df: pd.DataFrame):
    """Return (low_df, high_df) as bottom- and top-tertile NEOFAC_N groups."""
    t33 = df["NEOFAC_N"].quantile(1 / 3)
    t67 = df["NEOFAC_N"].quantile(2 / 3)
    low  = df[df["NEOFAC_N"] <= t33]
    high = df[df["NEOFAC_N"] >= t67]
    return low, high


def run_ttests(df: pd.DataFrame, roi_cols: list[str]) -> pd.DataFrame:
    low, high = tertile_groups(df)
    rows = []
    for col in roi_cols:
        g_lo = low[col].dropna().values
        g_hi = high[col].dropna().values
        t, pval = stats.ttest_ind(g_hi, g_lo, equal_var=False)
        df_stat = len(g_hi) + len(g_lo) - 2
        d = cohens_d(g_hi, g_lo)
        se_d = np.sqrt((len(g_hi) + len(g_lo)) /
                       (len(g_hi) * len(g_lo)) + d**2 / (2 * (len(g_hi) + len(g_lo))))
        rows.append({
            "roi_col":    col,
            "roi_label":  _pretty_roi(col),
            "n_low":      len(g_lo),
            "n_high":     len(g_hi),
            "mean_low":   np.mean(g_lo),
            "mean_high":  np.mean(g_hi),
            "t":          t,
            "df":         df_stat,
            "p":          pval,
            "d":          d,
            "d_ci_lo":    d - 1.96 * se_d,
            "d_ci_hi":    d + 1.96 * se_d,
        })

    results = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(results["p"].values, alpha=FDR_ALPHA, method="fdr_bh")
    results["p_fdr"]   = p_fdr
    results["sig_fdr"] = reject
    return results


def plot_violin(df: pd.DataFrame, results: pd.DataFrame,
                metric: str) -> plt.Figure | None:
    """
    Violin plot for FDR-significant ROIs (capped at MAX_VIOLIN).
    metric: 'Thck' or 'Area'
    """
    sig = results[results["sig_fdr"]].nlargest(MAX_VIOLIN, "d", keep="all").reset_index(drop=True)
    if sig.empty:
        return None

    low, high = tertile_groups(df)
    n_rois = len(sig)

    set_pub_style()
    fig, axes = plt.subplots(1, n_rois, figsize=(max(8, n_rois * 1.8), 5.5))
    if n_rois == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sig.iterrows()):
        col    = row["roi_col"]
        lo_data = low[col].dropna().values
        hi_data = high[col].dropna().values

        parts = ax.violinplot([lo_data, hi_data], positions=[0, 1],
                              showmedians=True, showextrema=False,
                              widths=0.7)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)
        for body, colour in zip(parts["bodies"], PALETTE_GROUP):
            body.set_facecolor(colour)
            body.set_alpha(0.75)
            body.set_edgecolor("none")

        # Overlay jittered strip
        for i, (data, col_) in enumerate(zip([lo_data, hi_data], PALETTE_GROUP)):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(data))
            ax.scatter(np.full(len(data), i) + jitter, data,
                       s=4, alpha=0.3, color=col_, linewidths=0, zorder=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low N", "High N"], fontsize=9)
        p_label = "***" if row["p_fdr"] < 0.001 else ("**" if row["p_fdr"] < 0.01 else "*")
        ax.set_title(f"{row['roi_label']}\nd = {row['d']:.2f}{p_label}", fontsize=9)
        ax.set_ylabel("mm" if metric == "Thck" else "mm²", fontsize=8)

    unit = "cortical thickness" if metric == "Thck" else "surface area"
    fig.suptitle(
        f"Neuroticism × {unit}\n(FDR-significant ROIs; High N = top tertile, Low N = bottom tertile)",
        fontsize=12, y=1.01)
    n_lo = int(sig["n_low"].iloc[0])
    n_hi = int(sig["n_high"].iloc[0])
    fig.text(0.01, 0.01,
             f"Low neuroticism n = {n_lo}; High neuroticism n = {n_hi}. "
             f"Violins = KDE; white line = median. * p_FDR < .05  ** p_FDR < .01  *** p_FDR < .001",
             fontsize=7, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def _save_table(results: pd.DataFrame, name_all: str, name_sig: str) -> None:
    apa_rows = [
        apa_ttest_row(
            roi=row["roi_label"], t=row["t"], df=row["df"], p=row["p"],
            d=row["d"], ci_lo=row["d_ci_lo"], ci_hi=row["d_ci_hi"],
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
    print("04 — HCP Neuroticism × Cortical Structure")
    print("=" * 60)

    print(f"\n[1/4] Loading {HCP_PARQUET.name} …")
    if not HCP_PARQUET.exists():
        raise FileNotFoundError("Run 01_hcp_data_prep.py first.")
    df = pd.read_parquet(HCP_PARQUET)

    low, high = tertile_groups(df)
    print(f"      Low-N tertile n = {len(low)}, High-N tertile n = {len(high)}")

    print("\n[2/4] Running t-tests (thickness + area, 68 ROIs each) …")
    res_thck = run_ttests(df, THCK_COLS)
    res_area = run_ttests(df, AREA_COLS)
    print(f"      Thickness FDR-sig: {res_thck['sig_fdr'].sum()} / {len(res_thck)}")
    print(f"      Area      FDR-sig: {res_area['sig_fdr'].sum()} / {len(res_area)}")

    print("\n[3/4] Saving tables …")
    _save_table(res_thck,
                "hcp_neuroticism_thck_all.csv",
                "hcp_neuroticism_thck_significant.csv")
    _save_table(res_area,
                "hcp_neuroticism_area_all.csv",
                "hcp_neuroticism_area_significant.csv")

    print("\n[4/4] Generating violin plots …")
    for results, metric in [(res_thck, "Thck"), (res_area, "Area")]:
        fig = plot_violin(df, results, metric)
        if fig is not None:
            stem = f"neuroticism_violin_{metric.lower()}_significant"
            save_fig(fig, stem, subdir=SUBDIR)
            print(f"      → results/figures/{SUBDIR}/{stem}.pdf/.png")
        else:
            print(f"      No FDR-significant {metric} ROIs — violin plot skipped.")

    print("\n✓ Neuroticism analysis complete.\n")


if __name__ == "__main__":
    main()
