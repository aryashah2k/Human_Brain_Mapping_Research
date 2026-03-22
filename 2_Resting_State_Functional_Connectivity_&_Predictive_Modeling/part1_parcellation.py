"""
Part 1: Parcellation & Connectivity Matrix Construction
Schaefer 400-parcel, 17-network atlas applied to HCP resting-state fMRI.
Run from the assignment2/ directory:
    conda run -n sandbox python part1_parcellation.py
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

# ── output directories ───────────────────────────────────────────────────────
os.makedirs("results/part1/data",    exist_ok=True)
os.makedirs("results/part1/figures", exist_ok=True)

# ── paths ────────────────────────────────────────────────────────────────────
DTSERIES = "rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii"
DLABEL   = "Schaefer2018_400Parcels_17Networks_order.dlabel.nii"
INFO_TXT = "Schaefer2018_400Parcels_17Networks_order_info.txt"

# ── figure style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})
DPI = 300


def save_fig(fig, stem):
    fig.savefig(f"results/part1/figures/{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(f"results/part1/figures/{stem}.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.png/.pdf")


# ── 1. load atlas label info ──────────────────────────────────────────────────
print("Loading atlas label info...")
parcel_names = []
with open(INFO_TXT) as f:
    lines = [l.strip() for l in f if l.strip()]

# Alternating: name line, then "index R G B A" line
for i in range(0, len(lines), 2):
    parcel_names.append(lines[i])

assert len(parcel_names) == 400, f"Expected 400 parcels, got {len(parcel_names)}"

# Extract 17-network label per parcel (e.g. "17Networks_LH_VisCent_ExStr_1" → "VisCent")
network_names = []
for name in parcel_names:
    parts = name.split("_")
    # format: 17Networks_<hemi>_<network>_<subregion>_<idx>
    if len(parts) >= 3:
        network_names.append(parts[2])
    else:
        network_names.append("Unknown")

unique_networks = list(dict.fromkeys(network_names))  # ordered unique
print(f"  {len(parcel_names)} parcels, {len(unique_networks)} unique networks: {unique_networks}")


# ── 2. build atlas label array mapped onto dtseries grayordinates ─────────────
print("Building parcel-to-grayordinate mapping...")

ts_img    = nib.load(DTSERIES)
atlas_img = nib.load(DLABEL)

ts_bm    = ts_img.header.get_axis(1)    # BrainModelAxis for dtseries
atlas_bm = atlas_img.header.get_axis(1) # BrainModelAxis for atlas (cortex only)
atlas_data = atlas_img.get_fdata(dtype=np.float32)[0]  # (64984,)

# dtseries: CORTEX_LEFT = [0, 29696), CORTEX_RIGHT = [29696, 59412)
# atlas:    CORTEX_LEFT = [0, 32492), CORTEX_RIGHT = [32492, 64984)
# Both use vertex indices; we need to map atlas vertex → dtseries position

# Get vertex indices used in the dtseries for cortex
ts_structs = {s[0]: s[1] for s in ts_bm.iter_structures()}
atlas_structs = {s[0]: s[1] for s in atlas_bm.iter_structures()}

n_ts = ts_img.shape[1]  # 91282
parcel_labels = np.zeros(n_ts, dtype=np.int32)  # 0 = unlabelled

for hemi, ts_sl, atlas_sl in [
    ("CIFTI_STRUCTURE_CORTEX_LEFT",  ts_structs["CIFTI_STRUCTURE_CORTEX_LEFT"],
     atlas_structs["CIFTI_STRUCTURE_CORTEX_LEFT"]),
    ("CIFTI_STRUCTURE_CORTEX_RIGHT", ts_structs["CIFTI_STRUCTURE_CORTEX_RIGHT"],
     atlas_structs["CIFTI_STRUCTURE_CORTEX_RIGHT"]),
]:
    # vertex indices present in dtseries for this hemisphere
    ts_verts  = ts_bm.vertex[ts_sl]
    # vertex indices present in atlas for this hemisphere
    atlas_verts = atlas_bm.vertex[atlas_sl]
    atlas_labels_hemi = atlas_data[atlas_sl]  # label at each atlas vertex

    # build lookup: vertex → atlas label
    atlas_vertex_to_label = dict(zip(atlas_verts.tolist(), atlas_labels_hemi.tolist()))

    ts_positions = np.arange(ts_sl.start, ts_sl.stop)
    for pos, vert in zip(ts_positions, ts_verts):
        label = atlas_vertex_to_label.get(int(vert), 0)
        parcel_labels[pos] = int(label)

labelled = (parcel_labels > 0).sum()
print(f"  {labelled}/{n_ts} grayordinates mapped to a parcel")


# ── 3. extract mean time series per parcel ────────────────────────────────────
print("Extracting parcel time series (this may take ~30s)...")
ts_data = ts_img.get_fdata(dtype=np.float32)  # (1200, 91282)
T = ts_data.shape[0]
timeseries = np.zeros((400, T), dtype=np.float32)

for p in range(1, 401):
    mask = parcel_labels == p
    if mask.sum() == 0:
        print(f"  WARNING: parcel {p} has 0 grayordinates")
        continue
    timeseries[p - 1] = ts_data[:, mask].mean(axis=1)

np.save("results/part1/data/timeseries_schaefer400.npy", timeseries)
print(f"  time series shape: {timeseries.shape}  → saved")


# ── 4. plot 20 ROI time series ────────────────────────────────────────────────
print("Plotting 20 ROI time series...")

# select 20 ROIs spanning different networks
selected_idx = []
seen_nets = set()
for i, net in enumerate(network_names):
    if net not in seen_nets:
        selected_idx.append(i)
        seen_nets.add(net)
    if len(selected_idx) == 20:
        break
# pad if < 20 unique networks
while len(selected_idx) < 20:
    selected_idx.append(selected_idx[-1] + 1)
selected_idx = selected_idx[:20]

fig, axes = plt.subplots(20, 1, figsize=(14, 18), sharex=True)
fig.suptitle("Resting-State BOLD Time Series – 20 Representative Parcels\n"
             "(Schaefer 400, 17-Network Atlas)", fontsize=13, fontweight="bold", y=1.01)

# assign a color per network
cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(unique_networks))
net_color = {n: cmap(i) for i, n in enumerate(unique_networks)}
t_axis = np.arange(T) * 0.72  # TR = 0.72 s for HCP

for row, idx in enumerate(selected_idx):
    ts = timeseries[idx]
    ts_z = (ts - ts.mean()) / (ts.std() + 1e-8)
    net  = network_names[idx]
    color = net_color[net]
    axes[row].plot(t_axis, ts_z, lw=0.7, color=color)
    axes[row].set_ylabel(parcel_names[idx].replace("17Networks_", ""), fontsize=6,
                         rotation=0, labelpad=2, ha="right", va="center")
    axes[row].yaxis.set_tick_params(labelsize=6)
    axes[row].spines[["top", "right"]].set_visible(False)
    axes[row].axhline(0, color="gray", lw=0.3, ls="--")

axes[-1].set_xlabel("Time (s)", fontsize=9)
fig.text(0.01, 0.5, "Z-scored BOLD signal", va="center", rotation="vertical", fontsize=9)
plt.tight_layout(rect=[0.08, 0, 1, 1])
save_fig(fig, "timeseries_20rois")


# ── 5. Pearson correlation matrix ─────────────────────────────────────────────
print("Computing 400×400 Pearson correlation matrix...")
# z-score each time series first (faster corrcoef)
ts_z = timeseries - timeseries.mean(axis=1, keepdims=True)
ts_z /= (timeseries.std(axis=1, keepdims=True) + 1e-8)
r_matrix = (ts_z @ ts_z.T) / T  # (400, 400)
np.fill_diagonal(r_matrix, 1.0)
r_matrix = np.clip(r_matrix, -1.0, 1.0)
np.save("results/part1/data/connectivity_matrix_pearson.npy", r_matrix)
print(f"  r_matrix range: [{r_matrix.min():.3f}, {r_matrix.max():.3f}]")


# ── 6. Fisher r-to-z transform ────────────────────────────────────────────────
print("Applying Fisher r-to-z transform...")
r_offdiag = r_matrix.copy()
np.fill_diagonal(r_offdiag, 0.0)
z_matrix = np.arctanh(r_offdiag)
np.save("results/part1/data/connectivity_matrix_fisherz.npy", z_matrix)
print(f"  z_matrix range: [{np.nanmin(z_matrix):.3f}, {np.nanmax(z_matrix):.3f}]")


# ── helper: compute network boundary tick positions ───────────────────────────
def network_boundaries(net_list):
    """Return (tick_positions, tick_labels, boundary_lines) for heatmap."""
    ticks, labels, bounds = [], [], []
    prev = None
    start = 0
    for i, n in enumerate(net_list):
        if n != prev:
            if prev is not None:
                bounds.append(i)
                ticks.append((start + i) / 2)
                labels.append(prev)
            start = i
            prev = n
    ticks.append((start + len(net_list)) / 2)
    labels.append(prev)
    return ticks, labels, bounds


ticks, tick_labels, bounds = network_boundaries(network_names)


# ── 7. heatmap: Pearson r matrix ──────────────────────────────────────────────
print("Plotting Pearson r heatmap...")
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(r_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto",
               interpolation="nearest")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
for b in bounds:
    ax.axhline(b - 0.5, color="black", lw=0.5, alpha=0.6)
    ax.axvline(b - 0.5, color="black", lw=0.5, alpha=0.6)
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
ax.set_yticks(ticks)
ax.set_yticklabels(tick_labels, fontsize=5)
ax.set_title("Functional Connectivity Matrix – Pearson r\n"
             "(Schaefer 400-Parcel, 17-Network Atlas)", fontweight="bold")
ax.set_xlabel("Parcels (ordered by network)")
ax.set_ylabel("Parcels (ordered by network)")
plt.tight_layout()
save_fig(fig, "heatmap_pearson")


# ── 8. heatmap: Fisher z matrix ───────────────────────────────────────────────
print("Plotting Fisher z heatmap...")
vlim = np.percentile(np.abs(z_matrix[~np.isinf(z_matrix)]), 99)
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(z_matrix, vmin=-vlim, vmax=vlim, cmap="RdBu_r", aspect="auto",
               interpolation="nearest")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fisher z")
for b in bounds:
    ax.axhline(b - 0.5, color="black", lw=0.5, alpha=0.6)
    ax.axvline(b - 0.5, color="black", lw=0.5, alpha=0.6)
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
ax.set_yticks(ticks)
ax.set_yticklabels(tick_labels, fontsize=5)
ax.set_title("Functional Connectivity Matrix – Fisher z-transformed\n"
             "(Schaefer 400-Parcel, 17-Network Atlas)", fontweight="bold")
ax.set_xlabel("Parcels (ordered by network)")
ax.set_ylabel("Parcels (ordered by network)")
plt.tight_layout()
save_fig(fig, "heatmap_fisherz")

print("\nPart 1 complete. Results in results/part1/")
