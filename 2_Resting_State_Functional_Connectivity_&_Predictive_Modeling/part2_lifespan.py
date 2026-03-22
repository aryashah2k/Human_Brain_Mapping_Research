"""
Part 2: Lifespan Connectivity Analysis
AABC dataset – Glasser HCP-MMPv1 atlas (379 ROIs).
Run from the assignment2/ directory:
    conda run -n sandbox python part2_lifespan.py
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

# ── output directories ───────────────────────────────────────────────────────
os.makedirs("results/part2/data",    exist_ok=True)
os.makedirs("results/part2/figures", exist_ok=True)

DPI = 300
CONN_DIR = "AABC_Release2_rfMRI_REST_FullCovarianceConnectivity"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "figure.dpi": 150})


def save_fig(fig, stem):
    fig.savefig(f"results/part2/figures/{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"results/part2/figures/{stem}.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


# ── 1. demographics: V1 subjects + age groups ────────────────────────────────
print("Loading demographics...")
demo = pd.read_csv("AABC2_subjects.csv", usecols=["id_event", "age_open", "sex"],
                   low_memory=False)
demo = demo[demo["id_event"].str.endswith("_V1")].copy()
demo["age"] = pd.to_numeric(demo["age_open"].astype(str)
                             .str.extract(r"(\d+)")[0], errors="coerce")
demo = demo.dropna(subset=["age"])
demo["age"] = demo["age"].astype(int)

bins   = [35, 50, 65, 80, 200]
labels = ["36-50", "51-65", "66-80", "81+"]
demo["age_group"] = pd.cut(demo["age"], bins=bins, labels=labels, right=True)
demo = demo.dropna(subset=["age_group"])
demo = demo.set_index("id_event")

print(f"  V1 subjects with valid age: {len(demo)}")
print("  Age group counts:\n", demo["age_group"].value_counts().sort_index())


# ── 2. discover ROI names and subject list from one reference file ─────────────
print("\nDiscovering ROI structure...")
csv_files = sorted([f for f in os.listdir(CONN_DIR) if f.endswith(".csv")])
print(f"  {len(csv_files)} CSV files found")

ref_file = os.path.join(CONN_DIR, csv_files[0])
ref_df   = pd.read_csv(ref_file, header=None, dtype={0: str}, low_memory=False)

# Row 0 is header: col0='x___', cols1..N = target ROI names
target_roi_names = ref_df.iloc[0, 1:].tolist()
n_rois = len(target_roi_names)

# Subject IDs present in the connectivity data (data rows, col 0)
conn_subjects_all = set(ref_df.iloc[1:, 0].astype(str).tolist())

# Intersect with demographics V1 subjects
valid_subjects = sorted(demo.index.intersection(conn_subjects_all))
print(f"  ROIs: {n_rois}, subjects in both datasets: {len(valid_subjects)}")

subj_to_idx = {s: i for i, s in enumerate(valid_subjects)}
N = len(valid_subjects)
del ref_df; gc.collect()


# ── 3. assemble (N, n_rois, n_rois) connectivity array ────────────────────────
# Each CSV corresponds to one source ROI; col0=subject_id, cols1..=target ROIs
print(f"\nAssembling {N} × {n_rois} × {n_rois} connectivity array...")
print("  (loading one CSV per source ROI – may take a few minutes)")

conn_array = np.zeros((N, n_rois, n_rois), dtype=np.float32)

# Extract source ROI name from filename:
# rfMRI_REST_FullCovarianceConnectivity_<ROI>.csv
def roi_name_from_file(fname):
    stem = fname.replace("rfMRI_REST_FullCovarianceConnectivity_", "").replace(".csv", "")
    return stem

source_roi_names = [roi_name_from_file(f) for f in csv_files]

for file_idx, fname in enumerate(csv_files):
    fpath = os.path.join(CONN_DIR, fname)
    df = pd.read_csv(fpath, header=None, dtype={0: str}, low_memory=False)

    # Skip header row; set subject id as index
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = ["subject_id"] + [f"t{i}" for i in range(n_rois)]
    df = df[df["subject_id"].isin(subj_to_idx)]

    if df.empty:
        continue

    source_name = source_roi_names[file_idx]
    # Find column index of this source ROI in target_roi_names
    if source_name in target_roi_names:
        src_col = target_roi_names.index(source_name)
    else:
        # Some source names may not exactly match target names (e.g. subcortical)
        src_col = file_idx if file_idx < n_rois else None

    # Fill row src_col in the matrix for each subject
    values = df.iloc[:, 1:].values.astype(np.float32)
    subject_ids = df["subject_id"].tolist()

    for local_i, sid in enumerate(subject_ids):
        if sid in subj_to_idx:
            row_i = subj_to_idx[sid]
            if src_col is not None:
                conn_array[row_i, src_col, :] = values[local_i]
            else:
                # fallback: fill by file order
                conn_array[row_i, file_idx % n_rois, :] = values[local_i]

    del df; gc.collect()

    if (file_idx + 1) % 50 == 0:
        print(f"  processed {file_idx + 1}/{len(csv_files)} files")

print(f"  done. Array shape: {conn_array.shape}")

# Symmetrize (covariance matrices should be symmetric; fill lower triangle)
conn_array = (conn_array + conn_array.transpose(0, 2, 1)) / 2.0

np.save("results/part2/data/connectivity_array.npy", conn_array)
print("  saved connectivity_array.npy")


# ── 4. save filtered subject list with age groups ─────────────────────────────
subj_df = demo.loc[valid_subjects, ["age", "age_group", "sex"]].copy()
subj_df.to_csv("results/part2/data/subjects_filtered.csv")
print("  saved subjects_filtered.csv")


# ── 5. compute per-group mean matrices ────────────────────────────────────────
print("\nComputing per-group mean connectivity matrices...")
group_data  = {}
group_means = {}

for grp in labels:
    mask_grp = subj_df["age_group"] == grp
    grp_subjects = subj_df[mask_grp].index.tolist()
    grp_indices  = [subj_to_idx[s] for s in grp_subjects if s in subj_to_idx]
    if len(grp_indices) == 0:
        print(f"  WARNING: no subjects for group {grp}")
        continue
    grp_arr = conn_array[grp_indices]
    group_data[grp]  = grp_arr
    group_means[grp] = grp_arr.mean(axis=0).astype(np.float32)
    fname = f"group_mean_{grp.replace('+', 'plus').replace('-', '_')}"
    np.save(f"results/part2/data/{fname}.npy", group_means[grp])
    print(f"  {grp}: n={len(grp_indices)}, saved {fname}.npy")


# ── 6. helper: heatmap plot ───────────────────────────────────────────────────
def plot_matrix_heatmap(mat, title, stem, vmin=None, vmax=None, cmap="RdBu_r",
                        cbar_label="Covariance"):
    fig, ax = plt.subplots(figsize=(9, 8))
    if vmin is None:
        lim = np.percentile(np.abs(mat), 98)
        vmin, vmax = -lim, lim
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Target ROI index")
    ax.set_ylabel("Source ROI index")
    plt.tight_layout()
    save_fig(fig, stem)


# ── 7. heatmaps per age group ─────────────────────────────────────────────────
print("\nPlotting per-group heatmaps...")
# Shared color limits across groups for fair comparison
all_means = np.stack(list(group_means.values()), axis=0)
lim = np.percentile(np.abs(all_means), 98)

for grp, mat in group_means.items():
    fstem = f"heatmap_group_{grp.replace('+', 'plus').replace('-', '_')}"
    n_subj = group_data[grp].shape[0]
    plot_matrix_heatmap(mat,
                        f"Mean Functional Connectivity – Age Group {grp} (n={n_subj})\n"
                        "(Glasser HCP-MMPv1, 379 ROIs)",
                        fstem, vmin=-lim, vmax=lim,
                        cbar_label="Mean Covariance")


# ── 8. one-way ANOVA across age groups at every edge ─────────────────────────
print("\nRunning one-way ANOVA at every edge...")
grp_list = [g for g in labels if g in group_data]
group_arrays = [group_data[g] for g in grp_list]  # list of (n_g, 379, 379)

F_matrix = np.zeros((n_rois, n_rois), dtype=np.float32)
p_matrix = np.ones( (n_rois, n_rois), dtype=np.float32)

# vectorised across subjects; loop over upper triangle
triu_i, triu_j = np.triu_indices(n_rois, k=1)
n_edges = len(triu_i)
print(f"  testing {n_edges} edges...")

# stack: (n_groups, n_edges) for each group
edge_groups = []
for ga in group_arrays:
    # ga shape: (n_subj, n_rois, n_rois)
    edges = ga[:, triu_i, triu_j]  # (n_subj, n_edges)
    edge_groups.append(edges)

# ANOVA edge-by-edge (batch in chunks to balance speed and memory)
chunk = 5000
F_vals = np.zeros(n_edges, dtype=np.float32)
p_vals = np.ones( n_edges, dtype=np.float32)

for start in range(0, n_edges, chunk):
    end = min(start + chunk, n_edges)
    samples = [g[:, start:end] for g in edge_groups]  # list of (n_subj, chunk)
    for k in range(end - start):
        col_data = [s[:, k] for s in samples]
        f, p = f_oneway(*col_data)
        F_vals[start + k] = f if np.isfinite(f) else 0.0
        p_vals[start + k] = p if np.isfinite(p) else 1.0

    if (start // chunk) % 10 == 0:
        print(f"  edges {start}–{end}/{n_edges}")

# Fill symmetric matrices
F_matrix[triu_i, triu_j] = F_vals
F_matrix[triu_j, triu_i] = F_vals
p_matrix[triu_i, triu_j] = p_vals
p_matrix[triu_j, triu_i] = p_vals

# FDR correction
_, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")
p_fdr_matrix = np.ones((n_rois, n_rois), dtype=np.float32)
p_fdr_matrix[triu_i, triu_j] = p_fdr
p_fdr_matrix[triu_j, triu_i] = p_fdr

sig_count = (p_fdr < 0.05).sum()
print(f"  significant edges (FDR q<0.05): {sig_count}/{n_edges}")

np.save("results/part2/data/anova_F_matrix.npy", F_matrix)
np.save("results/part2/data/anova_p_matrix.npy", p_matrix)
np.save("results/part2/data/anova_p_fdr_matrix.npy", p_fdr_matrix)


# ── 9. heatmap: F-statistic matrix ───────────────────────────────────────────
print("\nPlotting ANOVA F-stat matrix...")
fig, ax = plt.subplots(figsize=(9, 8))
vlim = np.percentile(F_matrix[F_matrix > 0], 99)
im = ax.imshow(F_matrix, cmap="hot_r", vmin=0, vmax=vlim,
               aspect="auto", interpolation="nearest")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="F-statistic")
ax.set_title("One-Way ANOVA F-Statistic Matrix Across Age Groups\n"
             "(Glasser HCP-MMPv1, 379 ROIs)", fontweight="bold")
ax.set_xlabel("Target ROI index")
ax.set_ylabel("Source ROI index")
plt.tight_layout()
save_fig(fig, "anova_fstat_matrix")


# ── 10. heatmap: significant edges only (FDR q<0.05) ─────────────────────────
print("Plotting significant edges heatmap...")
sig_F = np.where(p_fdr_matrix < 0.05, F_matrix, np.nan)
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(sig_F, cmap="hot_r", vmin=0, vmax=vlim,
               aspect="auto", interpolation="nearest")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="F-statistic")
ax.set_title("Significant Age-Related Connectivity Differences\n"
             "(FDR q<0.05, Glasser HCP-MMPv1)", fontweight="bold")
ax.set_xlabel("Target ROI index")
ax.set_ylabel("Source ROI index")
plt.tight_layout()
save_fig(fig, "anova_significant_edges")


# ── 11. network-level contrast bar plot ───────────────────────────────────────
print("Plotting network contrast barplot...")

# Glasser networks (broad grouping from file names / known HCP community IDs)
# We use the filename suffixes as proxy network labels
# Map each of the 379 source ROIs to a broad network using known HCP community structure
# Network labels come from the CSV filenames and the known HCP groupings
NETWORK_MAP = {
    "Default":            "Default",
    "Frontoparietal":     "Frontoparietal",
    "Cingulo-Opercular":  "Cingulo-Opercular",
    "Dorsal-attention":   "Dorsal Attention",
    "Language":           "Language",
    "Auditory":           "Auditory",
    "Somatomotor":        "Somatomotor",
    "Visual1":            "Visual 1",
    "Visual2":            "Visual 2",
    "Posterior-Multimodal": "Post. Multimodal",
    "Ventral-Multimodal": "Ventral Multimodal",
    "Orbito-Affective":   "Orbito-Affective",
    "TaskPositive":       "Task Positive",
}

# Build ROI → network mapping from CSV filenames that correspond to network-level files
# (e.g. rfMRI_REST_FullCovarianceConnectivity_Default.csv)
roi_network = {}
network_csv = [f for f in csv_files if not any(x in f for x in ["_L_", "_R_",
               "ACCUMBENS", "AMYGDALA", "CAUDATE", "CEREBELLUM", "DIENCEPHALON",
               "HIPPOCAMPUS", "PALLIDUM", "PUTAMEN", "THALAMUS", "BRAIN_STEM"])]
for f in network_csv:
    net_key = f.replace("rfMRI_REST_FullCovarianceConnectivity_", "").replace(".csv", "")
    if net_key in NETWORK_MAP:
        roi_network[roi_name_from_file(f)] = NETWORK_MAP[net_key]

# For ROIs not in roi_network, assign "Cortical (other)" or "Subcortical"
subcortical_keys = ["ACCUMBENS", "AMYGDALA", "CAUDATE", "CEREBELLUM",
                    "DIENCEPHALON", "HIPPOCAMPUS", "PALLIDUM", "PUTAMEN",
                    "THALAMUS", "BRAIN_STEM"]

for src_name in source_roi_names:
    if src_name not in roi_network:
        if any(k in src_name for k in subcortical_keys):
            roi_network[src_name] = "Subcortical"
        else:
            roi_network[src_name] = "Cortical (Other)"

# Mean connectivity per subject per network (mean over ROIs in that network)
network_list = sorted(set(roi_network.values()))
group_net_mean = {grp: {} for grp in grp_list}

for net in network_list:
    src_indices = [i for i, sn in enumerate(source_roi_names) if roi_network.get(sn) == net
                   and i < n_rois]
    if not src_indices:
        continue
    for grp in grp_list:
        ga = group_data[grp]  # (n_subj, n_rois, n_rois)
        net_mean_per_subj = ga[:, src_indices, :].mean(axis=(1, 2))
        group_net_mean[grp][net] = net_mean_per_subj

# Plot: grouped bar chart
valid_nets = [n for n in network_list if all(n in group_net_mean[g] for g in grp_list)]
x = np.arange(len(valid_nets))
width = 0.2
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(14, 6))
for gi, (grp, color) in enumerate(zip(grp_list, colors)):
    means = [group_net_mean[grp][n].mean() for n in valid_nets]
    sems  = [group_net_mean[grp][n].std() / np.sqrt(len(group_net_mean[grp][n]))
             for n in valid_nets]
    ax.bar(x + gi * width, means, width, yerr=sems, label=f"Age {grp}",
           color=color, alpha=0.85, capsize=3)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(valid_nets, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Mean Connectivity (Covariance)")
ax.set_title("Mean Functional Connectivity by Network and Age Group\n"
             "(Glasser HCP-MMPv1 Atlas)", fontweight="bold")
ax.legend(title="Age Group", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
save_fig(fig, "network_contrast_barplot")


# ── 12. summary report ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("ANOVA SUMMARY: Networks with greatest age-related differences")
print("="*60)

# Mean F per network
net_F = {}
for net in network_list:
    src_indices = [i for i, sn in enumerate(source_roi_names) if roi_network.get(sn) == net
                   and i < n_rois]
    if not src_indices:
        continue
    net_F[net] = F_matrix[np.ix_(src_indices, list(range(n_rois)))].mean()

for net, f_val in sorted(net_F.items(), key=lambda x: -x[1]):
    print(f"  {net:<30s}  mean F = {f_val:.3f}")

print(f"\nTotal significant edges (FDR q<0.05): {sig_count}/{n_edges} "
      f"({100*sig_count/n_edges:.1f}%)")
print("\nPart 2 complete. Results in results/part2/")
