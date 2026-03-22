"""
02_hcp_sleep.py
---------------
Analysis 1: Sleep quality (PSQI) × cortical thickness.
Groups: good sleepers (PSQI ≤ 5) vs. poor sleepers (PSQI > 5).
Metrics: 68 DK-atlas cortical thickness ROIs (34 L + 34 R).
Statistics: independent-samples t-test, FDR-BH correction, Cohen's d, 95% CI.
Outputs:
  results/tables/hcp_sleep_ttest_all.csv
  results/tables/hcp_sleep_ttest_significant.csv
  results/figures/hcp/sleep_bar_significant.pdf/.png

Run from project root:
    python scripts/02_hcp_sleep.py
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    HCP_PARQUET, TABLES_DIR, ensure_dirs, set_pub_style, save_fig,
    cohens_d, pearson_ci, apa_ttest_row, PALETTE_GROUP, FIG_DPI,
    DK_REGIONS,
)

# Re-use DK_REGIONS from utils via the prep script's exported list
THCK_COLS = ([f"FS_L_{r}_Thck" for r in DK_REGIONS] +
             [f"FS_R_{r}_Thck" for r in DK_REGIONS])

PSQI_THRESHOLD = 5   # ≤ 5 = good sleep quality (Buysse et al., 1989)
FDR_ALPHA      = 0.05
SUBDIR         = "hcp"


def _pretty_roi(col: str) -> str:
    """Convert 'FS_L_Superiorfrontal_Thck' -> 'L Superiorfrontal'."""
    parts = col.split("_")   # ['FS', 'L', 'Superiorfrontal', 'Thck']
    return f"{parts[1]} {parts[2]}"


def run_ttests(df: pd.DataFrame) -> pd.DataFrame:
    """Run t-tests for each thickness ROI; return full results DataFrame."""
    good = df[df["PSQI_Score"] <= PSQI_THRESHOLD]
    poor = df[df["PSQI_Score"] >  PSQI_THRESHOLD]

    rows = []
    for col in THCK_COLS:
        g = good[col].dropna().values
        p = poor[col].dropna().values
        t, pval = stats.ttest_ind(g, p, equal_var=False)   # Welch's t-test
        df_stat = (len(g) + len(p) - 2)
        d = cohens_d(g, p)
        # 95 % CI for Cohen's d (non-central t approximation; simple version)
        se_d = np.sqrt((len(g) + len(p)) / (len(g) * len(p)) + d**2 / (2 * (len(g) + len(p))))
        ci_lo = d - 1.96 * se_d
        ci_hi = d + 1.96 * se_d
        rows.append({
            "roi_col":  col,
            "roi_label": _pretty_roi(col),
            "n_good":    len(g),
            "n_poor":    len(p),
            "mean_good": np.mean(g),
            "mean_poor": np.mean(p),
            "se_good":   np.std(g, ddof=1) / np.sqrt(len(g)),
            "se_poor":   np.std(p, ddof=1) / np.sqrt(len(p)),
            "t":         t,
            "df":        df_stat,
            "p":         pval,
            "d":         d,
            "d_ci_lo":   ci_lo,
            "d_ci_hi":   ci_hi,
        })

    results = pd.DataFrame(rows)

    # FDR correction
    reject, p_fdr, _, _ = multipletests(results["p"].values, alpha=FDR_ALPHA, method="fdr_bh")
    results["p_fdr"]   = p_fdr
    results["sig_fdr"] = reject
    return results


def plot_bar(results: pd.DataFrame, df_full: pd.DataFrame) -> plt.Figure:
    """
    Bar chart of mean thickness ± SE for all FDR-significant ROIs,
    sorted by Cohen's d (descending absolute value).
    """
    sig = results[results["sig_fdr"]].copy()
    if sig.empty:
        print("  No FDR-significant ROIs — bar plot not generated.")
        return None

    sig = sig.reindex(sig["d"].abs().sort_values(ascending=False).index)

    set_pub_style()
    n_rois = len(sig)
    fig_w  = max(8, n_rois * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    x   = np.arange(n_rois)
    w   = 0.35
    colours = PALETTE_GROUP

    bars_good = ax.bar(x - w/2, sig["mean_good"], w,
                       yerr=sig["se_good"], capsize=3,
                       color=colours[0], label=f"Good sleepers (PSQI ≤ {PSQI_THRESHOLD})",
                       error_kw=dict(elinewidth=0.8, ecolor="dimgray"))
    bars_poor = ax.bar(x + w/2, sig["mean_poor"], w,
                       yerr=sig["se_poor"], capsize=3,
                       color=colours[1], label=f"Poor sleepers (PSQI > {PSQI_THRESHOLD})",
                       error_kw=dict(elinewidth=0.8, ecolor="dimgray"))

    ax.set_xticks(x)
    ax.set_xticklabels(sig["roi_label"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean cortical thickness (mm)")
    ax.set_title("Cortical thickness: good vs. poor sleepers\n(FDR-significant ROIs, sorted by |Cohen's d|)",
                 pad=10)
    ax.legend(frameon=False, loc="upper right")
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # Significance stars above each pair
    y_max = ax.get_ylim()[1]
    for i, (_, row) in enumerate(sig.iterrows()):
        p_label = "***" if row["p_fdr"] < 0.001 else ("**" if row["p_fdr"] < 0.01 else "*")
        bar_top = max(row["mean_good"] + row["se_good"],
                      row["mean_poor"] + row["se_poor"])
        ax.text(i, bar_top + 0.02, p_label, ha="center", va="bottom", fontsize=9)

    n_good = int(sig["n_good"].iloc[0])
    n_poor = int(sig["n_poor"].iloc[0])
    fig.text(0.01, 0.01,
             f"Good sleepers n = {n_good}; Poor sleepers n = {n_poor}. "
             f"Error bars = SE. * p_FDR < .05  ** p_FDR < .01  *** p_FDR < .001",
             fontsize=8, color="gray", transform=fig.transFigure)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def main() -> None:
    ensure_dirs()
    set_pub_style()

    print("=" * 60)
    print("02 — HCP Sleep × Cortical Thickness")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load clean data
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading {HCP_PARQUET.name} …")
    if not HCP_PARQUET.exists():
        raise FileNotFoundError(
            "Clean HCP parquet not found. Run 01_hcp_data_prep.py first.")
    df = pd.read_parquet(HCP_PARQUET)

    n_good = (df["PSQI_Score"] <= PSQI_THRESHOLD).sum()
    n_poor = (df["PSQI_Score"] >  PSQI_THRESHOLD).sum()
    print(f"      Good sleepers (PSQI ≤ {PSQI_THRESHOLD}): n = {n_good}")
    print(f"      Poor sleepers (PSQI > {PSQI_THRESHOLD}): n = {n_poor}")

    # ------------------------------------------------------------------
    # 2. Run t-tests with FDR correction
    # ------------------------------------------------------------------
    print("\n[2/4] Running Welch t-tests across 68 DK thickness ROIs …")
    results = run_ttests(df)
    n_sig = results["sig_fdr"].sum()
    print(f"      FDR-significant ROIs (q < {FDR_ALPHA}): {n_sig} / {len(results)}")

    # ------------------------------------------------------------------
    # 3. Save tables
    # ------------------------------------------------------------------
    print("\n[3/4] Saving tables …")

    # Full results (all 68 ROIs)
    apa_rows = [
        apa_ttest_row(
            roi=row["roi_label"],
            t=row["t"], df=row["df"], p=row["p"],
            d=row["d"], ci_lo=row["d_ci_lo"], ci_hi=row["d_ci_hi"],
            p_fdr=row["p_fdr"], sig_fdr=row["sig_fdr"]
        )
        for _, row in results.iterrows()
    ]
    apa_df = pd.DataFrame(apa_rows)
    all_path = TABLES_DIR / "hcp_sleep_ttest_all.csv"
    apa_df.to_csv(all_path, index=False)
    print(f"      Full table → {all_path.name}")

    # Significant only
    sig_df = apa_df[results["sig_fdr"].values].copy()
    sig_path = TABLES_DIR / "hcp_sleep_ttest_significant.csv"
    sig_df.to_csv(sig_path, index=False)
    print(f"      Significant table → {sig_path.name}")

    if not sig_df.empty:
        print("\n      FDR-significant ROIs:")
        print(sig_df[["ROI", "t", "p", "d", "p_FDR"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Figures
    # ------------------------------------------------------------------
    print("\n[4/4] Generating figures …")
    fig = plot_bar(results, df)
    if fig is not None:
        save_fig(fig, "sleep_bar_significant", subdir=SUBDIR)
        print(f"      Bar chart → results/figures/{SUBDIR}/sleep_bar_significant.pdf/.png")
    else:
        print("      No significant ROIs — bar chart skipped.")

    print("\n✓ Sleep analysis complete.\n")


if __name__ == "__main__":
    main()
