"""
06b_aabc_behavior.py
--------------------
Analysis 5 (AABC Behavioral):
  5a) Sleep quality (psqi_global) × cortical thickness — aging sample
  5b) Neuroticism (neo_n) × cortical thickness — aging sample

Both analyses mirror the HCP analyses (scripts 02 & 04) but in an older
population (36–89 yrs) using Glasser HCP-MMP1 atlas parcellation.

Statistics: Welch t-test (group comparisons) + Pearson correlation,
            FDR-BH correction, Cohen's d, APA-format output tables.

Outputs:
  results/tables/aabc_sleep_ttest_all.csv
  results/tables/aabc_sleep_ttest_significant.csv
  results/tables/aabc_neuroticism_thck_all.csv
  results/tables/aabc_neuroticism_thck_significant.csv
  results/figures/aabc/sleep_bar_significant.pdf/.png
  results/figures/aabc/neuroticism_violin_significant.pdf/.png

Run from project root:
    python scripts/06b_aabc_behavior.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    AABC_PARQUET, TABLES_DIR, ensure_dirs, set_pub_style, save_fig,
    cohens_d, apa_ttest_row, PALETTE_GROUP,
)

PSQI_THRESHOLD = 5     # ≤5 = good, >5 = poor (Buysse et al., 1989)
FDR_ALPHA      = 0.05
MAX_VIOLIN     = 12    # cap violin panels for readability
SUBDIR         = "aabc"


def _get_thick_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("thick_")]
    if not cols:
        raise ValueError("No 'thick_*' columns found. Run 05_aabc_data_prep.py first.")
    return cols


def _pretty_roi(col: str) -> str:
    """'thick_L_V1_ROI' → 'L_V1_ROI'"""
    return col.removeprefix("thick_")


def _run_ttest(df: pd.DataFrame, roi_cols: list[str],
               group_col: str, g1_mask, g2_mask) -> pd.DataFrame:
    """Generic Welch t-test loop over roi_cols; g1_mask / g2_mask are boolean Series."""
    rows = []
    g1 = df[g1_mask]
    g2 = df[g2_mask]
    for col in roi_cols:
        v1 = g1[col].dropna().values
        v2 = g2[col].dropna().values
        if len(v1) < 5 or len(v2) < 5:
            continue
        t, pval = stats.ttest_ind(v1, v2, equal_var=False)
        df_stat = len(v1) + len(v2) - 2
        d = cohens_d(v1, v2)
        se_d = np.sqrt((len(v1) + len(v2)) / (len(v1) * len(v2))
                       + d**2 / (2 * (len(v1) + len(v2))))
        rows.append({
            "roi_col":   col,
            "roi_label": _pretty_roi(col),
            "n_g1":      len(v1),
            "n_g2":      len(v2),
            "mean_g1":   np.mean(v1),
            "mean_g2":   np.mean(v2),
            "se_g1":     np.std(v1, ddof=1) / np.sqrt(len(v1)),
            "se_g2":     np.std(v2, ddof=1) / np.sqrt(len(v2)),
            "t":         t,
            "df":        df_stat,
            "p":         pval,
            "d":         d,
            "d_ci_lo":   d - 1.96 * se_d,
            "d_ci_hi":   d + 1.96 * se_d,
        })
    result = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(result["p"].values, alpha=FDR_ALPHA, method="fdr_bh")
    result["p_fdr"]   = p_fdr
    result["sig_fdr"] = reject
    return result


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


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_sleep_bar(results: pd.DataFrame, g1_label: str,
                   g2_label: str, n_g1: int, n_g2: int) -> plt.Figure | None:
    sig = results[results["sig_fdr"]].copy()
    if sig.empty:
        return None
    sig = sig.reindex(sig["d"].abs().sort_values(ascending=False).index)

    set_pub_style()
    n_rois = len(sig)
    fig, ax = plt.subplots(figsize=(max(8, n_rois * 0.6), 5))
    x, w = np.arange(n_rois), 0.35

    ax.bar(x - w/2, sig["mean_g1"], w, yerr=sig["se_g1"],
           capsize=3, color=PALETTE_GROUP[0], label=g1_label,
           error_kw=dict(elinewidth=0.8, ecolor="dimgray"))
    ax.bar(x + w/2, sig["mean_g2"], w, yerr=sig["se_g2"],
           capsize=3, color=PALETTE_GROUP[1], label=g2_label,
           error_kw=dict(elinewidth=0.8, ecolor="dimgray"))

    ax.set_xticks(x)
    ax.set_xticklabels(sig["roi_label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean cortical thickness (mm)")
    ax.set_title("AABC: Cortical thickness — good vs. poor sleepers\n"
                 "(FDR-significant ROIs, sorted by |Cohen's d|)", pad=10)
    ax.legend(frameon=False)

    for i, (_, row) in enumerate(sig.iterrows()):
        p_label = "***" if row["p_fdr"] < 0.001 else ("**" if row["p_fdr"] < 0.01 else "*")
        top = max(row["mean_g1"] + row["se_g1"], row["mean_g2"] + row["se_g2"])
        ax.text(i, top + 0.02, p_label, ha="center", va="bottom", fontsize=9)

    fig.text(0.01, 0.01,
             f"Good sleepers n={n_g1}; Poor sleepers n={n_g2}. Error bars=SE. "
             "* p_FDR<.05  ** p_FDR<.01  *** p_FDR<.001",
             fontsize=7, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def plot_neuroticism_violin(df: pd.DataFrame, results: pd.DataFrame,
                             low_mask, high_mask) -> plt.Figure | None:
    sig = results[results["sig_fdr"]].nlargest(MAX_VIOLIN, "d").reset_index(drop=True)
    if sig.empty:
        return None

    set_pub_style()
    n_rois = len(sig)
    fig, axes = plt.subplots(1, n_rois, figsize=(max(8, n_rois * 1.8), 5.5))
    if n_rois == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, sig.iterrows()):
        col = row["roi_col"]
        lo_data = df[low_mask][col].dropna().values
        hi_data = df[high_mask][col].dropna().values

        parts = ax.violinplot([lo_data, hi_data], positions=[0, 1],
                              showmedians=True, showextrema=False, widths=0.7)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)
        for body, colour in zip(parts["bodies"], PALETTE_GROUP):
            body.set_facecolor(colour)
            body.set_alpha(0.75)
            body.set_edgecolor("none")

        rng = np.random.default_rng(42)
        for i, (data, col_) in enumerate(zip([lo_data, hi_data], PALETTE_GROUP)):
            jitter = rng.uniform(-0.08, 0.08, len(data))
            ax.scatter(np.full(len(data), i) + jitter, data,
                       s=3, alpha=0.25, color=col_, linewidths=0, zorder=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low N", "High N"], fontsize=9)
        p_label = "***" if row["p_fdr"] < 0.001 else ("**" if row["p_fdr"] < 0.01 else "*")
        ax.set_title(f"{row['roi_label']}\nd={row['d']:.2f}{p_label}", fontsize=8)
        ax.set_ylabel("mm", fontsize=8)

    n_lo = int(sig["n_g1"].iloc[0])
    n_hi = int(sig["n_g2"].iloc[0])
    fig.suptitle("AABC: Neuroticism × Cortical Thickness\n"
                 "(FDR-significant ROIs; High N = top tertile; Low N = bottom tertile)",
                 fontsize=11, y=1.01)
    fig.text(0.01, 0.01,
             f"Low-N n={n_lo}; High-N n={n_hi}. White line=median. "
             "* p_FDR<.05  ** p_FDR<.01  *** p_FDR<.001",
             fontsize=7, color="gray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_dirs()
    set_pub_style()

    print("=" * 60)
    print("06b — AABC Behavioral × Cortical Thickness")
    print("=" * 60)

    print(f"\n[1/6] Loading {AABC_PARQUET.name} …")
    if not AABC_PARQUET.exists():
        raise FileNotFoundError("Run 05_aabc_data_prep.py first.")
    df = pd.read_parquet(AABC_PARQUET)
    print(f"      N = {len(df):,}")

    thick_cols = _get_thick_cols(df)
    print(f"      {len(thick_cols)} Glasser thickness ROIs")

    # ------------------------------------------------------------------
    # A. Sleep quality (PSQI) × cortical thickness
    # ------------------------------------------------------------------
    print("\n[2/6] PSQI grouping …")
    good_mask = df["psqi_global"] <= PSQI_THRESHOLD
    poor_mask = df["psqi_global"] >  PSQI_THRESHOLD
    n_good, n_poor = good_mask.sum(), poor_mask.sum()
    print(f"      Good sleepers (PSQI ≤{PSQI_THRESHOLD}): n={n_good}")
    print(f"      Poor sleepers (PSQI > {PSQI_THRESHOLD}): n={n_poor}")
    if n_good < 20 or n_poor < 20:
        raise RuntimeError("PSQI groups too small (< 20).")

    print("\n[3/6] PSQI t-tests (Glasser thickness) …")
    res_sleep = _run_ttest(df, thick_cols, "psqi_global", good_mask, poor_mask)
    n_sig_sleep = res_sleep["sig_fdr"].sum()
    print(f"      FDR-sig: {n_sig_sleep} / {len(res_sleep)}")

    _save_table(res_sleep,
                "aabc_sleep_ttest_all.csv",
                "aabc_sleep_ttest_significant.csv")

    fig_sleep = plot_sleep_bar(
        res_sleep,
        g1_label=f"Good sleepers (PSQI ≤{PSQI_THRESHOLD})",
        g2_label=f"Poor sleepers (PSQI >{PSQI_THRESHOLD})",
        n_g1=n_good, n_g2=n_poor
    )
    if fig_sleep is not None:
        save_fig(fig_sleep, "sleep_bar_significant", subdir=SUBDIR)
        print(f"      → results/figures/{SUBDIR}/sleep_bar_significant.pdf/.png")
    else:
        print("      No FDR-significant ROIs — bar plot skipped.")

    # ------------------------------------------------------------------
    # B. Neuroticism (neo_n) × cortical thickness
    # ------------------------------------------------------------------
    print("\n[4/6] NEO-N tertile grouping …")
    t33 = df["neo_n"].quantile(1/3)
    t67 = df["neo_n"].quantile(2/3)
    low_n_mask  = df["neo_n"] <= t33
    high_n_mask = df["neo_n"] >= t67
    n_lo, n_hi = low_n_mask.sum(), high_n_mask.sum()
    print(f"      Low-N ≤{t33:.0f}: n={n_lo}")
    print(f"      High-N ≥{t67:.0f}: n={n_hi}")
    if n_lo < 20 or n_hi < 20:
        raise RuntimeError("NEO-N tertile groups too small (< 20).")

    print("\n[5/6] NEO-N t-tests (Glasser thickness) …")
    res_neo = _run_ttest(df, thick_cols, "neo_n", low_n_mask, high_n_mask)
    n_sig_neo = res_neo["sig_fdr"].sum()
    print(f"      FDR-sig: {n_sig_neo} / {len(res_neo)}")

    _save_table(res_neo,
                "aabc_neuroticism_thck_all.csv",
                "aabc_neuroticism_thck_significant.csv")

    print("\n[6/6] Generating violin plots …")
    fig_neo = plot_neuroticism_violin(df, res_neo, low_n_mask, high_n_mask)
    if fig_neo is not None:
        save_fig(fig_neo, "neuroticism_violin_significant", subdir=SUBDIR)
        print(f"      → results/figures/{SUBDIR}/neuroticism_violin_significant.pdf/.png")
    else:
        print("      No FDR-significant neuroticism ROIs — violin plot skipped.")

    if not res_sleep[res_sleep["sig_fdr"]].empty:
        print("\n      PSQI top significant ROIs:")
        print(res_sleep[res_sleep["sig_fdr"]][["roi_label", "t", "d", "p_fdr"]]
              .nlargest(5, "d").to_string(index=False))
    if not res_neo[res_neo["sig_fdr"]].empty:
        print("\n      NEO-N top significant ROIs:")
        print(res_neo[res_neo["sig_fdr"]][["roi_label", "t", "d", "p_fdr"]]
              .nlargest(5, "d").to_string(index=False))

    print("\n✓ AABC behavioral analysis complete.\n")


if __name__ == "__main__":
    main()
