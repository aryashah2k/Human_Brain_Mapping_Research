"""
Part 1: EEG Preprocessing for Spontaneous Activity
====================================================
Resting-State Oscillatory Dynamics & Spectral Analysis
Subject: sub-NORB00001, Session 1 (~5-month-old infant)

Pipeline:
    1. Data loading & montage setup
    2. Bad channel identification
    3. Bandpass filtering (1-30 Hz)
    4. Bad channel interpolation
    5. ICA artifact correction
    6. Re-referencing to average
    7. Visualization (raw vs preprocessed)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

# ---------------------------------------------------------------------------
# Paths (all relative)
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(
    "sub-NORB00001", "ses-1", "eeg", "sub-NORB00001_ses-1_task-EEG_eeg.edf"
)
ELECTRODES_PATH = os.path.join(
    "sub-NORB00001", "ses-1", "eeg", "sub-NORB00001_ses-1_electrodes.tsv"
)
EVENTS_PATH = os.path.join(
    "sub-NORB00001", "ses-1", "eeg", "sub-NORB00001_ses-1_task-EEG_events.tsv"
)
PREPROCESSED_DIR = os.path.join("results", "preprocessed")
FIGURES_DIR = os.path.join("results", "part1_figures")

os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

raw = mne.io.read_raw_edf(DATA_PATH, preload=True, verbose=True)
print(f"Loaded {raw.info['nchan']} channels, sfreq={raw.info['sfreq']} Hz, "
      f"duration={raw.times[-1]:.1f} s")

# Rename channels that use older 10-20 naming to standard 10-10 names
RENAME_MAP = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "FZ": "Fz", "CZ": "Cz", "PZ": "Pz",
}
rename_actual = {k: v for k, v in RENAME_MAP.items() if k in raw.ch_names}
if rename_actual:
    raw.rename_channels(rename_actual)
    print(f"Renamed channels: {rename_actual}")

# Set channel types (all EEG)
raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

# Set montage — use standard 10-10 for channels that have positions
# Pg1/Pg2 won't match the standard montage and will be handled as bads
montage = mne.channels.make_standard_montage("standard_1020")
# Only set montage for channels present in both the data and the montage
matched = [ch for ch in raw.ch_names if ch in montage.ch_names]
unmatched = [ch for ch in raw.ch_names if ch not in montage.ch_names]
print(f"Channels matched to montage: {matched}")
print(f"Channels NOT in montage (will be marked bad): {unmatched}")
raw.set_montage(montage, on_missing="warn")

# Read events to find eyes-closed segment boundary
import csv
with open(EVENTS_PATH, "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    events_list = list(reader)

eyes_open_onset = None
for ev in events_list:
    if ev["trial_type"].strip() == "eyes_open":
        eyes_open_onset = float(ev["onset"])
        break

if eyes_open_onset is not None:
    print(f"Cropping to eyes-closed segment: 0 – {eyes_open_onset:.1f} s")
    raw.crop(tmin=0, tmax=eyes_open_onset)
else:
    print("No eyes_open event found; using full recording.")

# Keep a copy of raw data for comparison plots later
raw_original = raw.copy()

# ---------------------------------------------------------------------------
# 2. Bad Channel Management
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Bad channel identification")
print("=" * 60)

# Drop channels with no electrode coordinates (cannot be interpolated)
no_coord_chs = list(unmatched)  # Pg1, Pg2
if no_coord_chs:
    print(f"Dropping channels without coordinates: {no_coord_chs}")
    raw.drop_channels(no_coord_chs)
    raw_original.drop_channels(no_coord_chs)

# Statistical detection: flag channels with extreme variance or near-zero variance
data = raw.get_data()  # (n_channels, n_times)
ch_vars = np.var(data, axis=1)
median_var = np.median(ch_vars)
bads = []

for idx, ch in enumerate(raw.ch_names):
    # Flag if variance > 5× median or < 0.1× median (flat)
    if ch_vars[idx] > 5 * median_var or ch_vars[idx] < 0.1 * median_var:
        bads.append(ch)
        print(f"  Flagged '{ch}' — variance={ch_vars[idx]:.4e} "
              f"(median={median_var:.4e})")

# Check for bridged channels (very high correlation between neighbors)
from itertools import combinations
corr_matrix = np.corrcoef(data)
good_indices = [i for i, ch in enumerate(raw.ch_names) if ch not in bads]
for i, j in combinations(good_indices, 2):
    if abs(corr_matrix[i, j]) > 0.99:
        ch_i, ch_j = raw.ch_names[i], raw.ch_names[j]
        if ch_i not in bads:
            bads.append(ch_i)
            print(f"  Flagged '{ch_i}' — bridged with '{ch_j}' "
                  f"(r={corr_matrix[i, j]:.4f})")

raw.info["bads"] = bads
if bads:
    print(f"Bad channels to interpolate ({len(bads)}): {bads}")
else:
    print("No additional bad channels detected.")

# ---------------------------------------------------------------------------
# 3. Filtering
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Bandpass filtering (1–30 Hz)")
print("=" * 60)

# Hardware already applied 0.5–30 Hz; we tighten the high-pass to 1 Hz
# to further suppress slow drifts.  No notch filter needed (60 Hz line
# noise is already above the 30 Hz lowpass).
raw.filter(l_freq=1.0, h_freq=30.0, fir_design="firwin", verbose=True)
print("Bandpass filter applied: 1–30 Hz (FIR, firwin)")

# ---------------------------------------------------------------------------
# 4. Artifact Rejection & Re-referencing
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4a: Interpolating bad channels")
print("=" * 60)

if raw.info["bads"]:
    raw.interpolate_bads(reset_bads=True, verbose=True)
    print("Bad channels interpolated via spherical spline.")
else:
    print("No bad channels to interpolate.")

print("\n" + "=" * 60)
print("STEP 4b: ICA for artifact correction")
print("=" * 60)

# Use all but 1 component for ICA
n_good = len(raw.ch_names) - len(raw.info["bads"])
n_components = min(n_good - 1, 15)  # cap at 15 for 21-ch infant data
print(f"Fitting ICA with {n_components} components …")

ica = ICA(n_components=n_components, method="fastica", random_state=42,
          max_iter=1000)
ica.fit(raw, verbose=True)
explained_var = ica.get_explained_variance_ratio(raw)
total_var = sum(explained_var.values())
print(f"ICA fit complete — explained variance ratio: {total_var:.2%}")

# Identify eye-blink components by correlating with frontal channels
frontal_channels = [ch for ch in ["Fp1", "Fp2"] if ch in raw.ch_names]
eog_indices = []
for ch in frontal_channels:
    inds, scores = ica.find_bads_eog(
        raw, ch_name=ch, threshold=2.5, verbose=False
    )
    eog_indices.extend(inds)
eog_indices = list(set(eog_indices))

if eog_indices:
    print(f"Excluding ICA components (eye artifacts): {eog_indices}")
else:
    # Fallback: exclude component most correlated with frontal channels
    if frontal_channels:
        fp_data = raw.copy().pick(frontal_channels).get_data().mean(axis=0)
        sources = ica.get_sources(raw).get_data()
        corrs = np.array([abs(np.corrcoef(s, fp_data)[0, 1])
                          for s in sources])
        best = int(np.argmax(corrs))
        if corrs[best] > 0.3:
            eog_indices = [best]
            print(f"Heuristic: excluding IC{best} (r={corrs[best]:.3f} "
                  f"with frontal avg)")
        else:
            print("No clearly artifactual components detected.")
    else:
        print("No frontal channels available; skipping EOG rejection.")

ica.exclude = eog_indices
raw = ica.apply(raw, verbose=True)
print("ICA artifact correction applied.")

print("\n" + "=" * 60)
print("STEP 4c: Re-referencing to average")
print("=" * 60)

raw.set_eeg_reference("average", projection=False, verbose=True)
print("Re-referenced to average.")

# ---------------------------------------------------------------------------
# 5. Visualization — Raw vs Preprocessed
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Generating comparison figures")
print("=" * 60)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
})

# --- Figure 1: PSD comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Raw PSD
psd_raw = raw_original.compute_psd(method="welch", fmin=1, fmax=30,
                                    n_fft=512, n_overlap=256, verbose=False)
psd_raw.plot(axes=axes[0], show=False, spatial_colors=True)
axes[0].set_title("Raw — Power Spectral Density")
axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Power Spectral Density (dB)")

# Preprocessed PSD
psd_clean = raw.compute_psd(method="welch", fmin=1, fmax=30,
                             n_fft=512, n_overlap=256, verbose=False)
psd_clean.plot(axes=axes[1], show=False, spatial_colors=True)
axes[1].set_title("Preprocessed — Power Spectral Density")
axes[1].set_xlabel("Frequency (Hz)")

fig.suptitle("PSD Comparison: Raw vs Preprocessed", fontsize=14, y=1.02)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGURES_DIR, f"fig1_psd_comparison.{ext}"),
                bbox_inches="tight")
print("Saved fig1_psd_comparison (.png, .pdf)")
plt.close(fig)

# --- Figure 2: Time-series comparison (10 s snippet) ---
t_start, t_end = 10.0, 20.0  # seconds
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Pick a subset of channels for readability
display_chs = ["Fp1", "F3", "C3", "P3", "O1", "Fz", "Cz", "Pz"]
display_chs = [ch for ch in display_chs if ch in raw.ch_names
               and ch in raw_original.ch_names]

# Raw time series
raw_snippet = raw_original.copy().pick(display_chs).crop(tmin=t_start,
                                                          tmax=t_end)
data_raw_snip = raw_snippet.get_data() * 1e6  # to µV
times = raw_snippet.times + t_start
offsets = np.arange(len(display_chs)) * 80  # µV spacing

for i, ch in enumerate(display_chs):
    axes[0].plot(times, data_raw_snip[i] + offsets[i], linewidth=0.5)
axes[0].set_yticks(offsets)
axes[0].set_yticklabels(display_chs)
axes[0].set_title("Raw — Time Series (10–20 s)")
axes[0].set_ylabel("Channel")

# Preprocessed time series
clean_snippet = raw.copy().pick(display_chs).crop(tmin=t_start, tmax=t_end)
data_clean_snip = clean_snippet.get_data() * 1e6
times_c = clean_snippet.times + t_start

for i, ch in enumerate(display_chs):
    axes[1].plot(times_c, data_clean_snip[i] + offsets[i], linewidth=0.5)
axes[1].set_yticks(offsets)
axes[1].set_yticklabels(display_chs)
axes[1].set_title("Preprocessed — Time Series (10–20 s)")
axes[1].set_ylabel("Channel")
axes[1].set_xlabel("Time (s)")

fig.suptitle("Time-Series Comparison: Raw vs Preprocessed",
             fontsize=14, y=1.01)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGURES_DIR, f"fig2_timeseries_comparison.{ext}"),
                bbox_inches="tight")
print("Saved fig2_timeseries_comparison (.png, .pdf)")
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Save Preprocessed Data
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: Saving preprocessed data")
print("=" * 60)

out_fname = os.path.join(PREPROCESSED_DIR,
                         "sub-NORB00001_ses-1_preprocessed-raw.fif")
raw.save(out_fname, overwrite=True, verbose=True)
print(f"Preprocessed data saved to: {out_fname}")

print("\n✓ Part 1 complete. Figures in:", FIGURES_DIR)
print("  Preprocessed data in:", PREPROCESSED_DIR)
