"""
Part 1: Data Cleaning & Preprocessing
P300 Auditory Oddball EEG Dataset (ds003061, sub-001)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EEG_DIR  = os.path.join(BASE_DIR, "eeg")
RES_DIR  = os.path.join(BASE_DIR, "results")
FIG_DIR  = os.path.join(RES_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(RES_DIR, "stats"), exist_ok=True)

RAW_FILE  = os.path.join(EEG_DIR, "sub-001_task-P300_run-1_eeg.set")
CLEAN_FIF = os.path.join(RES_DIR, "sub-001_preprocessed-raw.fif")
ICA_FIF   = os.path.join(RES_DIR, "sub-001_ica-cleaned-raw.fif")

# ── Figure helpers ─────────────────────────────────────────────────────────────
def save_fig(name):
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

# ── 1. Load raw data ───────────────────────────────────────────────────────────
print("Loading raw data...")
raw = mne.io.read_raw_eeglab(RAW_FILE, preload=True, verbose=False)

# Drop non-EEG auxiliary channels
aux_types = ["misc", "gsr", "resp", "temp"]
aux_ch = [ch for ch in raw.ch_names
          if raw.get_channel_types([ch])[0].lower() in aux_types
          or ch.startswith("EXG") or ch in ("GSR1","GSR2","Erg1","Erg2","Resp","Plet","Temp")]
raw.drop_channels([c for c in aux_ch if c in raw.ch_names])

# Set channel type for remaining channels and montage
raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
montage = mne.channels.make_standard_montage("biosemi64")
raw.set_montage(montage, on_missing="ignore", verbose=False)
print(f"  Channels retained: {len(raw.ch_names)}")

# ── 2. Bad channel detection ───────────────────────────────────────────────────
print("Detecting bad channels...")
raw_for_bads = raw.copy().filter(1.0, 40.0, verbose=False)
data = raw_for_bads.get_data()
ch_var  = np.var(data, axis=1)
ch_mean = np.mean(data, axis=1)

var_z   = np.abs((ch_var  - np.median(ch_var))  / (np.std(ch_var)  + 1e-12))
mean_z  = np.abs((ch_mean - np.median(ch_mean)) / (np.std(ch_mean) + 1e-12))

bad_by_var   = [raw.ch_names[i] for i in np.where(var_z  > 3.5)[0]]
bad_by_flat  = [raw.ch_names[i] for i in np.where(ch_var < 1e-14)[0]]
bad_channels = list(set(bad_by_var + bad_by_flat))
raw.info["bads"] = bad_channels
print(f"  Bad channels detected: {bad_channels}")

# ── 3. Bandpass + Notch filtering ─────────────────────────────────────────────
print("Filtering: bandpass 1-40 Hz + notch 50 Hz...")
raw_raw_copy = raw.copy()   # keep unfiltered copy for comparison plots
raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)
raw.notch_filter(50.0, verbose=False)

# ── 4. Interpolate bad channels ───────────────────────────────────────────────
if bad_channels:
    print(f"  Interpolating {len(bad_channels)} bad channel(s)...")
    raw.interpolate_bads(reset_bads=True, verbose=False)

# ── 5. ICA artifact correction ────────────────────────────────────────────────
print("Running ICA (infomax, 20 components)...")
ica = ICA(n_components=20, method="infomax", fit_params=dict(extended=True),
          random_state=42, max_iter=800, verbose=False)
ica.fit(raw, verbose=False)

# Auto-detect ocular components via correlation with frontal channels
frontal_chs = [ch for ch in ["Fp1", "Fp2", "AFz"] if ch in raw.ch_names]
eog_indices, eog_scores = ica.find_bads_eog(
    raw, ch_name=frontal_chs, threshold=3.0, verbose=False
)

# Auto-detect muscle components via high-frequency power ratio
muscle_indices, muscle_scores = ica.find_bads_muscle(raw, threshold=0.5, verbose=False)

exclude = list(set(eog_indices + muscle_indices))
ica.exclude = exclude
print(f"  ICA components excluded (ocular+muscle): {exclude}")

# Save ICA component plot for documentation
fig_ica = ica.plot_components(picks=range(min(20, ica.n_components_)),
                               show=False, res=64)
if isinstance(fig_ica, list):
    fig_ica = fig_ica[0]
fig_ica.savefig(os.path.join(FIG_DIR, "part1_ica_components.png"), dpi=150, bbox_inches="tight")
fig_ica.savefig(os.path.join(FIG_DIR, "part1_ica_components.pdf"), dpi=150, bbox_inches="tight")
plt.close("all")

# Apply ICA
raw_ica = raw.copy()
ica.apply(raw_ica, verbose=False)

# Save ICA-cleaned (pre-rereference) for Part 3
raw_ica.save(ICA_FIF, overwrite=True, verbose=False)
print(f"  ICA-cleaned (pre-rereference) saved → {ICA_FIF}")

# ── 6. Average re-reference ───────────────────────────────────────────────────
print("Applying average reference...")
raw_clean, _ = mne.set_eeg_reference(raw_ica, ref_channels="average",
                                      projection=False, verbose=False)

# ── 7. Save preprocessed data ─────────────────────────────────────────────────
raw_clean.save(CLEAN_FIF, overwrite=True, verbose=False)
print(f"Preprocessed data saved → {CLEAN_FIF}")

# ── 8. Visualization: PSD comparison ──────────────────────────────────────────
print("Generating PSD comparison plot...")

# Compute PSDs via Welch
fmin, fmax = 1.0, 50.0
raw_raw_copy.filter(0.5, 90.0, verbose=False)  # light filter just to remove DC for PSD display

psd_raw  = raw_raw_copy.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                                     n_fft=512, n_overlap=128, verbose=False)
psd_clean = raw_clean.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                                   n_fft=512, n_overlap=128, verbose=False)

freqs_r, psds_r = psd_raw.freqs,  psd_raw.get_data()
freqs_c, psds_c = psd_clean.freqs, psd_clean.get_data()

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(freqs_r, np.mean(psds_r, axis=0) * 1e12,
            color="steelblue", alpha=0.8, linewidth=1.5, label="Raw")
ax.semilogy(freqs_c, np.mean(psds_c, axis=0) * 1e12,
            color="tomato",    alpha=0.9, linewidth=1.5, label="Preprocessed")
ax.axvline(50, color="gray", linestyle="--", linewidth=1, label="50 Hz notch")
ax.set_xlabel("Frequency (Hz)", fontsize=12)
ax.set_ylabel("Power Spectral Density (µV²/Hz)", fontsize=12)
ax.set_title("PSD Comparison: Raw vs Preprocessed (Mean across EEG channels)", fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(fmin, fmax)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
save_fig("part1_psd_comparison")
print("  Saved: part1_psd_comparison.png/.pdf")

# ── 9. Visualization: Time-series comparison ──────────────────────────────────
print("Generating time-series comparison plot...")

rep_chs = [ch for ch in ["Fp1", "Fz", "Cz", "Pz", "Oz"] if ch in raw.ch_names]
t_start, t_stop = 30.0, 40.0   # 10-second segment

# Use the pre-filtered copy already in memory (aux-dropped, montage-set)
# Apply 0.5 Hz high-pass only to remove Biosemi DC offset for display purposes
raw_display = raw_raw_copy.copy().pick_channels(rep_chs, ordered=False)
raw_display.filter(0.5, None, fir_design="firwin", verbose=False)
raw_clean_plot = raw_clean.copy().pick_channels(rep_chs, ordered=False)

sfreq = raw_clean.info["sfreq"]
s0, s1 = int(t_start * sfreq), int(t_stop * sfreq)
times  = np.arange(s1 - s0) / sfreq + t_start

data_r = raw_display.get_data(start=s0, stop=s1) * 1e6
data_c = raw_clean_plot.get_data(start=s0, stop=s1) * 1e6

# Channel order may differ after pick; reindex by name
raw_disp_chs = raw_display.ch_names
clean_chs    = raw_clean_plot.ch_names

n_ch = len(rep_chs)
fig, axes = plt.subplots(n_ch, 2, figsize=(16, 3 * n_ch), sharex=True)

# Use a percentile-based symmetric ylim so both panels share the same scale
combined = np.concatenate([data_r, data_c], axis=1)
ylim_val = np.percentile(np.abs(combined), 99) * 1.1
ylim_val = max(ylim_val, 5.0)  # at least ±5 µV

for i, ch in enumerate(rep_chs):
    ri = raw_disp_chs.index(ch) if ch in raw_disp_chs else i
    ci = clean_chs.index(ch)    if ch in clean_chs    else i

    axes[i, 0].plot(times, data_r[ri], color="steelblue", linewidth=0.8)
    axes[i, 0].set_ylabel(f"{ch}\n(µV)", fontsize=9)
    axes[i, 0].set_ylim(-ylim_val, ylim_val)
    axes[i, 0].grid(True, alpha=0.3)

    axes[i, 1].plot(times, data_c[ci], color="tomato", linewidth=0.8)
    axes[i, 1].set_ylim(-ylim_val, ylim_val)
    axes[i, 1].grid(True, alpha=0.3)

axes[0, 0].set_title("Raw EEG", fontsize=12, fontweight="bold")
axes[0, 1].set_title("Preprocessed EEG", fontsize=12, fontweight="bold")
axes[-1, 0].set_xlabel("Time (s)", fontsize=11)
axes[-1, 1].set_xlabel("Time (s)", fontsize=11)
fig.suptitle("EEG Time-Series Comparison (30–40 s segment)", fontsize=13, y=1.01)
fig.tight_layout()
save_fig("part1_timeseries_comparison")
print("  Saved: part1_timeseries_comparison.png/.pdf")

print("\nPart 1 complete.")
print(f"  Bad channels: {bad_channels}")
print(f"  ICA components excluded: {exclude}")
print(f"  Preprocessed file: {CLEAN_FIF}")
