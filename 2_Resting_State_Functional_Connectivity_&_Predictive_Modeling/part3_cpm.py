"""
Part 3: Connectome-Based Predictive Modeling (CPM)
Predicts Fluid Intelligence (FluidIQ_Tr35_60y) from AABC connectivity matrices.
Protocol: Shen et al. 2017 (LOOCV, p<0.01 edge threshold).
Run from the assignment2/ directory:
    conda run -n sandbox python part3_cpm.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ── output directories ───────────────────────────────────────────────────────
os.makedirs("results/part3/data",    exist_ok=True)
os.makedirs("results/part3/figures", exist_ok=True)

DPI = 300
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 150})


def save_fig(fig, stem):
    fig.savefig(f"results/part3/figures/{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"results/part3/figures/{stem}.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


# ── 1. load connectivity array and subject list from Part 2 ──────────────────
print("Loading Part 2 connectivity data...")
conn_array = np.load("results/part2/data/connectivity_array.npy")  # (N, 379, 379)
subj_df    = pd.read_csv("results/part2/data/subjects_filtered.csv", index_col=0)
N_all, n_rois, _ = conn_array.shape
print(f"  connectivity array: {conn_array.shape}")

# ── 2. load behavioral variable ───────────────────────────────────────────────
print("Loading FluidIQ scores...")
demo = pd.read_csv("AABC2_subjects.csv",
                   usecols=["id_event", "FluidIQ_Tr35_60y"], low_memory=False)
demo = demo[demo["id_event"].str.endswith("_V1")].set_index("id_event")

# Align to subjects in connectivity array
valid_idx    = []
valid_scores = []

for i, sid in enumerate(subj_df.index):
    if sid in demo.index:
        score = demo.loc[sid, "FluidIQ_Tr35_60y"]
        if pd.notna(score):
            try:
                valid_idx.append(i)
                valid_scores.append(float(score))
            except (ValueError, TypeError):
                pass

valid_idx    = np.array(valid_idx, dtype=int)
valid_scores = np.array(valid_scores, dtype=np.float32)
N = len(valid_idx)
print(f"  subjects with valid FluidIQ: {N}")

# Subset connectivity array to subjects with valid scores
X_full = conn_array[valid_idx]  # (N, 379, 379)
y      = valid_scores           # (N,)

# ── 3. extract upper-triangle edges ──────────────────────────────────────────
triu_i, triu_j = np.triu_indices(n_rois, k=1)
n_edges = len(triu_i)
print(f"  upper-triangle edges: {n_edges}")

# Edge matrix: (N, n_edges)
edges_full = X_full[:, triu_i, triu_j].astype(np.float32)

# ── 4. LOOCV CPM (Shen et al. 2017) ──────────────────────────────────────────
print(f"\nRunning LOOCV CPM on {N} subjects (p<0.01 threshold)...")
print("  This may take several minutes...")

P_THRESH = 0.01

pred_pos      = np.zeros(N, dtype=np.float64)
pred_neg      = np.zeros(N, dtype=np.float64)
pred_combined = np.zeros(N, dtype=np.float64)

# Accumulate edge selection counts for visualization
edge_pos_count = np.zeros(n_edges, dtype=np.int32)
edge_neg_count = np.zeros(n_edges, dtype=np.int32)


def fast_pearsonr_matrix(X, y):
    """Compute Pearson r and p-value between each column of X and vector y.
    X: (n_subj, n_edges)  y: (n_subj,)
    Returns r: (n_edges,), p: (n_edges,)
    """
    n = X.shape[0]
    X_c = X - X.mean(axis=0, keepdims=True)
    y_c = y - y.mean()
    num   = (X_c * y_c[:, None]).sum(axis=0)
    denom = np.sqrt((X_c ** 2).sum(axis=0)) * np.sqrt((y_c ** 2).sum())
    denom = np.where(denom == 0, 1e-12, denom)
    r = num / denom
    r = np.clip(r, -1.0 + 1e-8, 1.0 - 1e-8)
    # t-statistic and two-tailed p
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    from scipy.stats import t as t_dist
    p = 2 * t_dist.sf(np.abs(t), df=n - 2)
    return r.astype(np.float32), p.astype(np.float32)


for i in range(N):
    if (i + 1) % 100 == 0:
        print(f"  fold {i+1}/{N}")

    # Train/test split
    train_mask = np.ones(N, dtype=bool)
    train_mask[i] = False

    X_train = edges_full[train_mask]   # (N-1, n_edges)
    y_train = y[train_mask].astype(np.float64)
    X_test  = edges_full[i]            # (n_edges,)

    # Feature selection: edge-behavior correlation on training set
    r_vals, p_vals = fast_pearsonr_matrix(X_train, y_train)

    pos_mask = (r_vals > 0) & (p_vals < P_THRESH)
    neg_mask = (r_vals < 0) & (p_vals < P_THRESH)

    edge_pos_count += pos_mask.astype(np.int32)
    edge_neg_count += neg_mask.astype(np.int32)

    # Network strength: sum of selected edges per subject
    if pos_mask.sum() > 0:
        pos_strength_train = X_train[:, pos_mask].sum(axis=1)
        pos_strength_test  = X_test[pos_mask].sum()
        coef_pos = np.polyfit(pos_strength_train, y_train, 1)
        pred_pos[i] = np.polyval(coef_pos, pos_strength_test)
    else:
        pred_pos[i] = np.nan

    if neg_mask.sum() > 0:
        neg_strength_train = X_train[:, neg_mask].sum(axis=1)
        neg_strength_test  = X_test[neg_mask].sum()
        coef_neg = np.polyfit(neg_strength_train, y_train, 1)
        pred_neg[i] = np.polyval(coef_neg, neg_strength_test)
    else:
        pred_neg[i] = np.nan

    # Combined: positive − negative network strength
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        comb_train = pos_strength_train - neg_strength_train
        comb_test  = pos_strength_test  - neg_strength_test
        coef_comb  = np.polyfit(comb_train, y_train, 1)
        pred_combined[i] = np.polyval(coef_comb, comb_test)
    elif pos_mask.sum() > 0:
        pred_combined[i] = pred_pos[i]
    elif neg_mask.sum() > 0:
        pred_combined[i] = pred_neg[i]
    else:
        pred_combined[i] = np.nan

print("  LOOCV complete.")

# ── 5. save predictions ───────────────────────────────────────────────────────
pred_df = pd.DataFrame({
    "subject_id":        subj_df.index[valid_idx],
    "actual_FluidIQ":    y,
    "predicted_pos":     pred_pos,
    "predicted_neg":     pred_neg,
    "predicted_combined": pred_combined,
})
pred_df.to_csv("results/part3/data/cpm_predictions.csv", index=False)
print("  saved cpm_predictions.csv")

# Save edge selection frequency maps
pos_freq_matrix = np.zeros((n_rois, n_rois), dtype=np.float32)
neg_freq_matrix = np.zeros((n_rois, n_rois), dtype=np.float32)
pos_freq_matrix[triu_i, triu_j] = edge_pos_count / N
pos_freq_matrix[triu_j, triu_i] = edge_pos_count / N
neg_freq_matrix[triu_i, triu_j] = edge_neg_count / N
neg_freq_matrix[triu_j, triu_i] = edge_neg_count / N
np.save("results/part3/data/cpm_selected_edges_pos.npy", pos_freq_matrix)
np.save("results/part3/data/cpm_selected_edges_neg.npy", neg_freq_matrix)


# ── 6. evaluation ─────────────────────────────────────────────────────────────
print("\nCPM Evaluation Results:")
results = {}
for name, preds in [("Positive Network", pred_pos),
                    ("Negative Network", pred_neg),
                    ("Combined (Pos-Neg)", pred_combined)]:
    valid = ~np.isnan(preds)
    if valid.sum() < 10:
        print(f"  {name}: insufficient predictions")
        continue
    r, p = pearsonr(y[valid], preds[valid])
    rho, p_rho = spearmanr(y[valid], preds[valid])
    mae = np.mean(np.abs(y[valid] - preds[valid]))
    results[name] = {"r": r, "p": p, "rho": rho, "p_rho": p_rho, "mae": mae,
                     "preds": preds, "valid": valid}
    print(f"  {name}: Pearson r={r:.3f} (p={p:.4f}), Spearman rho={rho:.3f}, MAE={mae:.2f}")


# ── 7. scatter plot: predicted vs. observed ───────────────────────────────────
print("\nPlotting predicted vs. observed scatter...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("CPM: Predicted vs. Observed Fluid Intelligence\n"
             "(LOOCV, Shen et al. 2017, p<0.01 threshold)",
             fontsize=13, fontweight="bold")

plot_items = [
    ("Positive Network", pred_pos,      "#2196F3"),
    ("Negative Network", pred_neg,      "#F44336"),
    ("Combined (Pos−Neg)", pred_combined, "#4CAF50"),
]

for ax, (name, preds, color) in zip(axes, plot_items):
    valid = ~np.isnan(preds)
    if valid.sum() < 10:
        ax.set_title(f"{name}\n(insufficient data)")
        continue

    y_v = y[valid]
    p_v = preds[valid]
    r, p_val = pearsonr(y_v, p_v)

    ax.scatter(y_v, p_v, alpha=0.4, s=18, color=color, edgecolors="none")

    # regression line
    m, b = np.polyfit(y_v, p_v, 1)
    x_line = np.linspace(y_v.min(), y_v.max(), 100)
    ax.plot(x_line, m * x_line + b, color="black", lw=1.5, ls="--")

    # identity reference
    lo = min(y_v.min(), p_v.min())
    hi = max(y_v.max(), p_v.max())
    ax.plot([lo, hi], [lo, hi], color="gray", lw=0.8, ls=":", alpha=0.7)

    ax.set_xlabel("Observed FluidIQ", fontsize=10)
    ax.set_ylabel("Predicted FluidIQ", fontsize=10)
    ax.set_title(f"{name}\nr = {r:.3f}, p = {p_val:.4f}", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # annotation box
    ax.text(0.05, 0.92, f"n = {valid.sum()}", transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.tight_layout()
save_fig(fig, "scatter_predicted_vs_observed")


# ── 8. edge selection frequency heatmaps ─────────────────────────────────────
print("Plotting edge selection frequency heatmaps...")

for freq_mat, label, color, stem in [
    (pos_freq_matrix, "Positive (pos corr. with FluidIQ)", "YlOrRd",
     "edge_selection_matrix_pos"),
    (neg_freq_matrix, "Negative (neg corr. with FluidIQ)", "YlGnBu",
     "edge_selection_matrix_neg"),
]:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(freq_mat, cmap=color, vmin=0, vmax=freq_mat.max(),
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Selection frequency (proportion of folds)")
    ax.set_title(f"CPM Edge Selection Frequency – {label}\n"
                 f"(LOOCV across {N} subjects, Glasser 379-ROI atlas)",
                 fontweight="bold")
    ax.set_xlabel("Target ROI index")
    ax.set_ylabel("Source ROI index")
    plt.tight_layout()
    save_fig(fig, stem)

print("\nPart 3 complete. Results in results/part3/")
