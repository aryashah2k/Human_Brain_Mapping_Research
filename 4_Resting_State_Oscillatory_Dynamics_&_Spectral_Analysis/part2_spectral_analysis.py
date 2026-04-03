"""
Part 2: Spectral Power and Time-Frequency Visualization
=========================================================
Resting-State Oscillatory Dynamics & Spectral Analysis
Subject: sub-NORB00001, Session 1 (~5-month-old infant)

Pipeline:
    1. Power Spectral Density (Welch's method)
    2. Time-Frequency Representation (Morlet wavelets)
    3. Topographic mapping of band power
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet

# ---------------------------------------------------------------------------
# Paths (all relative)
# ---------------------------------------------------------------------------
PREPROCESSED_FILE = os.path.join(
    "results", "preprocessed", "sub-NORB00001_ses-1_preprocessed-raw.fif"
)
FIGURES_DIR = os.path.join("results", "part2_figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Frequency band definitions (Hz)
BANDS = {
    "Delta (1–4 Hz)": (1, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–13 Hz)": (8, 13),
    "Beta (13–30 Hz)": (13, 30),
}

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------------------
# Load preprocessed data
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading preprocessed data")
print("=" * 60)

raw = mne.io.read_raw_fif(PREPROCESSED_FILE, preload=True, verbose=True)
print(f"Channels: {raw.ch_names}")
print(f"Duration: {raw.times[-1]:.1f} s, sfreq: {raw.info['sfreq']} Hz")

# ---------------------------------------------------------------------------
# 1. Power Spectral Density — Welch's Method
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 1: Power Spectral Density (Welch's method)")
print("=" * 60)

psd = raw.compute_psd(method="welch", fmin=1, fmax=30,
                       n_fft=400, n_overlap=200, verbose=False)
freqs = psd.freqs
psd_data = psd.get_data()  # (n_channels, n_freqs) in V²/Hz
psd_db = 10 * np.log10(psd_data)  # convert to dB

# Define channel groups for regional averages
REGIONS = {
    "Frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "Central": ["C3", "C4", "Cz"],
    "Parietal": ["P3", "P4", "Pz"],
    "Occipital": ["O1", "O2"],
    "Temporal": ["T7", "T8", "P7", "P8"],
}

# Filter regions to only include channels that exist in the data
regions_filtered = {}
for region, chs in REGIONS.items():
    present = [ch for ch in chs if ch in raw.ch_names]
    if present:
        regions_filtered[region] = present

# --- Figure 3: PSD with band shading + regional averages ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: All channels with band shading
band_colors = ["#d4e6f1", "#d5f5e3", "#fdebd0", "#fadbd8"]
for idx, (band_name, (fmin, fmax)) in enumerate(BANDS.items()):
    axes[0].axvspan(fmin, fmax, alpha=0.3, color=band_colors[idx],
                    label=band_name)

for i, ch in enumerate(raw.ch_names):
    axes[0].plot(freqs, psd_db[i], linewidth=0.6, alpha=0.7)

axes[0].set_xlabel("Frequency (Hz)")
axes[0].set_ylabel("Power Spectral Density (dB)")
axes[0].set_title("PSD — All Channels (Welch's Method)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_xlim(1, 30)

# Panel B: Regional averages
region_colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#c0392b"]
for idx, (region, chs) in enumerate(regions_filtered.items()):
    ch_indices = [raw.ch_names.index(ch) for ch in chs]
    region_avg = psd_db[ch_indices].mean(axis=0)
    axes[1].plot(freqs, region_avg, linewidth=1.8,
                 color=region_colors[idx], label=region)

for idx, (band_name, (fmin, fmax)) in enumerate(BANDS.items()):
    axes[1].axvspan(fmin, fmax, alpha=0.15, color=band_colors[idx])

axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Power Spectral Density (dB)")
axes[1].set_title("PSD — Regional Averages")
axes[1].legend(loc="upper right", fontsize=9)
axes[1].set_xlim(1, 30)

fig.suptitle("Power Spectral Density (Welch's Method)", fontsize=14, y=1.02)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGURES_DIR, f"fig3_psd_welch.{ext}"),
                bbox_inches="tight")
print("Saved fig3_psd_welch (.png, .pdf)")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Time-Frequency Representation — Morlet Wavelets
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Time-Frequency Representation (Morlet wavelets)")
print("=" * 60)

# Create fixed-length epochs (2 s, no overlap) for TFR
events = mne.make_fixed_length_events(raw, duration=2.0)
epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0 - 1.0 / raw.info["sfreq"],
                    baseline=None, preload=True, verbose=False)
print(f"Created {len(epochs)} epochs of 2.0 s each")

# Define frequencies and wavelet cycles
tfr_freqs = np.arange(1, 31, 0.5)  # 1 to 30 Hz in 0.5 Hz steps
n_cycles = tfr_freqs / 2.0  # adaptive: more cycles at higher freq

# Compute TFR (average across epochs)
tfr = tfr_morlet(epochs, freqs=tfr_freqs, n_cycles=n_cycles,
                 return_itc=False, average=True, verbose=True)

# --- Figure 4: TFR for representative channels ---
repr_channels = ["Cz", "O1"]
repr_channels = [ch for ch in repr_channels if ch in raw.ch_names]

fig, axes = plt.subplots(1, len(repr_channels), figsize=(8 * len(repr_channels), 5))
if len(repr_channels) == 1:
    axes = [axes]

for ax, ch in zip(axes, repr_channels):
    ch_idx = tfr.ch_names.index(ch)
    tfr_ch = tfr.data[ch_idx]  # (n_freqs, n_times)
    tfr_ch_db = 10 * np.log10(tfr_ch + 1e-30)  # dB, avoid log(0)

    im = ax.pcolormesh(tfr.times, tfr_freqs, tfr_ch_db,
                       shading="gouraud", cmap="RdBu_r")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Time-Frequency Representation — {ch}")
    cbar = fig.colorbar(im, ax=ax, label="Power (dB)")

fig.suptitle("Morlet Wavelet TFR (averaged across 2-s epochs)",
             fontsize=14, y=1.02)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGURES_DIR, f"fig4_tfr_morlet.{ext}"),
                bbox_inches="tight")
print("Saved fig4_tfr_morlet (.png, .pdf)")
plt.close(fig)

# --- Figure 4b: Spectrogram (single-trial concatenated view) ---
# Show power over the entire recording for a single representative channel
repr_ch = "Cz" if "Cz" in raw.ch_names else raw.ch_names[0]

fig, ax = plt.subplots(figsize=(16, 5))

# Build epoch-concatenated TFR matrix
ch_idx_in_tfr = tfr.ch_names.index(repr_ch)
n_epochs = len(epochs)
n_freqs = len(tfr_freqs)

# Recompute per-epoch TFR (not averaged)
tfr_single = tfr_morlet(epochs, freqs=tfr_freqs, n_cycles=n_cycles,
                        return_itc=False, average=False, verbose=False)
# tfr_single.data shape: (n_epochs, n_channels, n_freqs, n_times)
ch_idx_st = tfr_single.ch_names.index(repr_ch)
single_data = tfr_single.data[:, ch_idx_st, :, :]  # (n_epochs, n_freqs, n_times)

# Concatenate epochs along time
concat = np.concatenate([single_data[e] for e in range(n_epochs)], axis=1)
concat_db = 10 * np.log10(concat + 1e-30)

# Build concatenated time axis
epoch_dur = epochs.times[-1] - epochs.times[0] + 1.0 / raw.info["sfreq"]
concat_times = np.linspace(0, n_epochs * epoch_dur, concat.shape[1])

im = ax.pcolormesh(concat_times, tfr_freqs, concat_db,
                   shading="gouraud", cmap="RdBu_r")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title(f"Spectrogram — {repr_ch} (concatenated 2-s epochs)")
fig.colorbar(im, ax=ax, label="Power (dB)")
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGURES_DIR, f"fig4b_spectrogram_{repr_ch}.{ext}"),
                bbox_inches="tight")
print(f"Saved fig4b_spectrogram_{repr_ch} (.png, .pdf)")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Topographic Mapping — Band Power
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Topographic mapping of absolute band power")
print("=" * 60)

# Compute absolute band power from PSD (Welch) for each channel
band_powers = {}
for band_name, (fmin, fmax) in BANDS.items():
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    # Integrate power (mean across frequency bins, still in V²/Hz)
    band_powers[band_name] = psd_data[:, freq_mask].mean(axis=1)

# --- Figure 5: Topographic maps (4-panel, one per band) ---
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for ax, (band_name, power) in zip(axes, band_powers.items()):
    im, _ = mne.viz.plot_topomap(
        power, raw.info, axes=ax, show=False,
        cmap="YlOrRd", contours=4,
    )
    ax.set_title(band_name, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Power (V²/Hz)")

fig.suptitle(
    "Topographic Distribution of Absolute Band Power\n"
    "(sub-NORB00001, ~5-month-old infant, eyes-closed resting state)",
    fontsize=13, y=1.06,
)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGURES_DIR, f"fig5_topomap_bandpower.{ext}"),
                bbox_inches="tight")
print("Saved fig5_topomap_bandpower (.png, .pdf)")
plt.close(fig)

print("\n✓ Part 2 complete. All figures saved in:", FIGURES_DIR)
