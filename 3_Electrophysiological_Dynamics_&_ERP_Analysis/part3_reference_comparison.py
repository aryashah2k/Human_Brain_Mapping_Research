"""
Part 3: Reference Comparison & Topographic Analysis
P300 Auditory Oddball EEG Dataset (ds003061, sub-001)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
EEG_DIR   = os.path.join(BASE_DIR, "eeg")
RES_DIR   = os.path.join(BASE_DIR, "results")
FIG_DIR   = os.path.join(RES_DIR, "figures")
STATS_DIR = os.path.join(RES_DIR, "stats")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

ICA_FIF    = os.path.join(RES_DIR, "sub-001_ica-cleaned-raw.fif")
EVENTS_TSV = os.path.join(EEG_DIR, "sub-001_task-P300_run-1_events.tsv")

# ── Reference schemes ─────────────────────────────────────────────────────────
# Each entry: (label, ref_channels or "average" or None)
REFERENCES = [
    ("Average Reference",        "average"),
    ("Cz Reference",             ["Cz"]),
    ("Linked Mastoids (TP7+TP8)","linked_mastoids"),
    ("Fpz (Nasion Proxy)",       ["Fpz"]),
    ("No Re-reference (CMS/DRL)","none"),
]

# ── Event parsing helper ───────────────────────────────────────────────────────
def build_events(raw):
    ev_df = pd.read_csv(EVENTS_TSV, sep="\t")
    sfreq = raw.info["sfreq"]
    EVENT_ID = {"standard": 1, "oddball": 2, "noise": 3}

    def map_event(val):
        val = str(val).strip().lower()
        if "oddball" in val:
            return 2
        if "noise" in val:
            return 3
        if "standard" in val:
            return 1
        return None

    rows = []
    for _, row in ev_df.iterrows():
        eid = map_event(row["value"])
        if eid is None:
            continue
        sample = int(round(row["onset"] * sfreq))
        rows.append([sample, 0, eid])
    return np.array(rows, dtype=np.int32), EVENT_ID


# ── Epoch & average helper ────────────────────────────────────────────────────
def get_oddball_evoked(raw_ref):
    events, EVENT_ID = build_events(raw_ref)
    reject = dict(eeg=150e-6)
    epochs = mne.Epochs(
        raw_ref, events, event_id=EVENT_ID,
        tmin=-0.2, tmax=0.8,
        baseline=(-0.2, 0.0),
        reject=reject,
        preload=True,
        verbose=False
    )
    return epochs["oddball"].average()


# ── Apply reference schemes ───────────────────────────────────────────────────
print("Loading ICA-cleaned (pre-reference) data...")
raw_base = mne.io.read_raw_fif(ICA_FIF, preload=True, verbose=False)

# Determine P300 peak latency from average-reference oddball evoked at Pz
print("Computing average-reference evoked to determine P300 peak latency...")
raw_avg = raw_base.copy()
raw_avg, _ = mne.set_eeg_reference(raw_avg, ref_channels="average",
                                    projection=False, verbose=False)
ev_avg = get_oddball_evoked(raw_avg)
pz_idx = ev_avg.ch_names.index("Pz") if "Pz" in ev_avg.ch_names else 0
t_mask = (ev_avg.times >= 0.25) & (ev_avg.times <= 0.55)
pz_data_win = ev_avg.data[pz_idx, t_mask]
peak_lat = ev_avg.times[t_mask][np.argmax(np.abs(pz_data_win))]
print(f"  P300 peak latency at Pz (average ref): {peak_lat*1000:.1f} ms")

# ── Compute evoked per reference scheme ───────────────────────────────────────
evokeds = {}
print("\nApplying reference schemes and computing oddball evoked...")
for label, ref in REFERENCES:
    print(f"  {label} ...")
    raw_r = raw_base.copy()

    if ref == "average":
        raw_r, _ = mne.set_eeg_reference(raw_r, ref_channels="average",
                                          projection=False, verbose=False)
    elif ref == "none":
        pass  # no rereferencing — use as-is (CMS/DRL)
    elif ref == "linked_mastoids":
        mastoids = [ch for ch in ["TP7", "TP8"] if ch in raw_r.ch_names]
        if len(mastoids) == 2:
            raw_r, _ = mne.set_eeg_reference(raw_r, ref_channels=mastoids,
                                              projection=False, verbose=False)
        elif len(mastoids) == 1:
            raw_r, _ = mne.set_eeg_reference(raw_r, ref_channels=mastoids,
                                              projection=False, verbose=False)
        else:
            print(f"    WARNING: mastoid channels not found, skipping rereferencing")
    else:
        # Single electrode reference
        valid = [ch for ch in ref if ch in raw_r.ch_names]
        if valid:
            raw_r, _ = mne.set_eeg_reference(raw_r, ref_channels=valid,
                                              projection=False, verbose=False)
        else:
            print(f"    WARNING: reference channel(s) {ref} not found")

    ev = get_oddball_evoked(raw_r)
    evokeds[label] = ev

# ── Scalp topography maps at P300 peak latency ───────────────────────────────
print(f"\nGenerating topography comparison at {peak_lat*1000:.1f} ms...")

n_refs = len(REFERENCES)
fig = plt.figure(figsize=(4 * n_refs, 4.5))
gs  = gridspec.GridSpec(1, n_refs, figure=fig, wspace=0.35)

peak_idx_map = {}
axes_list = []
data_at_peak = []

for i, (label, _) in enumerate(REFERENCES):
    ev = evokeds[label]
    t_idx = ev.time_as_index(peak_lat)[0]
    topo_data = ev.data[:, t_idx]
    data_at_peak.append(topo_data)

# Symmetric color scale across all refs for fair comparison
all_vals = np.concatenate(data_at_peak)
vmax = np.percentile(np.abs(all_vals), 97) 
vlim = (-vmax, vmax)

for i, (label, _) in enumerate(REFERENCES):
    ax = fig.add_subplot(gs[i])
    ev = evokeds[label]
    t_idx = ev.time_as_index(peak_lat)[0]
    topo_data = ev.data[:, t_idx]

    im, _ = mne.viz.plot_topomap(
        topo_data, ev.info,
        axes=ax, show=False,
        contours=6, cmap="RdBu_r",
        vlim=vlim,
        sensors=True,
    )
    ax.set_title(label, fontsize=9, fontweight="bold", pad=6)
    axes_list.append((ax, im))

# Shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                            norm=plt.Normalize(vmin=vlim[0]*1e6, vmax=vlim[1]*1e6))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("Amplitude (µV)", fontsize=10)

fig.suptitle(
    f"Scalp Topography at P300 Peak ({peak_lat*1000:.0f} ms) — Oddball Condition\n"
    "Reference Scheme Comparison",
    fontsize=12, fontweight="bold", y=1.02
)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"part3_topo_reference_comparison.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: part3_topo_reference_comparison.png/.pdf")

# ── ERP waveform comparison at Pz ────────────────────────────────────────────
print("Plotting ERP waveform comparison at Pz across references...")

ref_colors = ["crimson", "steelblue", "forestgreen", "darkorchid", "darkorange"]

fig, ax = plt.subplots(figsize=(11, 5))
for i, (label, _) in enumerate(REFERENCES):
    ev = evokeds[label]
    if "Pz" in ev.ch_names:
        pz_i = ev.ch_names.index("Pz")
        ax.plot(ev.times * 1000, ev.data[pz_i] * 1e6,
                color=ref_colors[i], linewidth=1.8, label=label)

ax.axvspan(300, 500, alpha=0.12, color="gold", label="P300 window")
ax.axvline(peak_lat * 1000, color="gray", linestyle="--", linewidth=1,
           label=f"Peak ({peak_lat*1000:.0f} ms)")
ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Time (ms)", fontsize=12)
ax.set_ylabel("Amplitude (µV)", fontsize=12)
ax.set_title("P300 ERP at Pz — Oddball Condition Across Reference Schemes", fontsize=13)
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(-200, 800)
ax.grid(True, alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIG_DIR, f"part3_erp_Pz_reference_comparison.{ext}"),
                dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: part3_erp_Pz_reference_comparison.png/.pdf")

# ── Quantitative summary: P300 amplitude & peak latency per reference ─────────
print("\nP300 summary per reference scheme:")
summary_lines = ["P300 Mean Amplitude (300–500 ms) and Peak Latency at Pz",
                 "=" * 60, ""]
for label, _ in REFERENCES:
    ev = evokeds[label]
    if "Pz" not in ev.ch_names:
        continue
    pz_i = ev.ch_names.index("Pz")
    win_mask = (ev.times >= 0.30) & (ev.times <= 0.50)
    mean_amp  = ev.data[pz_i, win_mask].mean() * 1e6
    peak_mask = (ev.times >= 0.25) & (ev.times <= 0.55)
    pk_t = ev.times[peak_mask][np.argmax(np.abs(ev.data[pz_i, peak_mask]))] * 1000
    line = f"  {label:<30s}: mean amp = {mean_amp:+.3f} µV, peak latency = {pk_t:.1f} ms"
    summary_lines.append(line)
    print(line)

# ── Written discussion ────────────────────────────────────────────────────────
discussion = """

Discussion: How Reference Choice Affects ERP Topography
========================================================

Background
----------
In EEG, all voltage measurements are inherently relative — the recorded signal
at any electrode reflects the potential difference between that electrode and a
chosen reference. This means the reference is never "neutral"; it is always an
active contributor to the scalp topography observed.

Effects of Each Reference Scheme on P300 Topography
----------------------------------------------------

1. Average Reference
   The average reference computes the mean across all scalp electrodes and
   subtracts it from each channel. This approximates a "zero-potential" reference
   if the electrode coverage is dense and approximately uniform across the head.
   For the P300 (maximally positive at centro-parietal sites), the average
   reference distributes the return current across all sensors, yielding a
   broadly accurate centro-parietal positivity with complementary frontal
   negativity. This is the de-facto standard for ERP topography analysis
   (Luck, 2005; Nunez & Srinivasan, 2006).

2. Cz Reference
   Cz sits at the vertex — near the center of P300 activity. Referencing to Cz
   subtracts the P300 signal itself from all other channels, artificially
   suppressing the apparent amplitude of centro-parietal electrodes and shifting
   the apparent topographic center outward. The P300 positivity will appear
   weaker or absent at parietal sites and may create spurious polarities at
   frontal or temporal channels (Donchin & Hagans, 1977).

3. Linked Mastoids (TP7 + TP8)
   Mastoid references are common in clinical EEG and auditory ERP research.
   For auditory P300, the mastoids are relatively inactive (low voltage at
   latencies of 300–500 ms), making them a reasonable choice. However, mastoid
   referencing can introduce asymmetries if one mastoid is noisier than the
   other, and it tends to inflate fronto-central amplitudes compared to average
   reference (Hagemann et al., 2001). The occipito-temporal activity can be
   slightly suppressed relative to average reference due to proximity of TP7/TP8.

4. Fpz / Nasion Reference
   The Fpz electrode (frontal midline, closest available to nasion) is highly
   susceptible to eye-blink and frontal muscle artefacts. Using it as a reference
   injects these artefacts into every channel. For P300 analysis this is a
   particularly poor choice: the blink-contaminated Fpz reference inverts frontal
   voltages and distorts the centro-parietal P300 positivity, often generating
   artifactual anterior-posterior gradients. It is generally not recommended
   for cognitive ERP research (Nunez et al., 1997).

5. No Re-reference (CMS/DRL, Biosemi)
   The Biosemi CMS/DRL system uses a Common Mode Sense (CMS) electrode and a
   Driven Right Leg (DRL) electrode rather than a true ground/reference. Without
   re-referencing, each channel's offset is determined by the CMS location
   (typically near POz). This can introduce a common DC offset and the true
   scalp distribution is obscured, making topographic interpretation unreliable
   (Biosemi, 2020). It is never recommended for publication-quality topographic
   analysis.

Why Reference Shifts the 'Center' of Electrical Activity
---------------------------------------------------------
The apparent "center" of activation in a topographic map is determined by where
the most positive (or most negative) voltages are observed relative to the
reference. When the reference carries residual signal at the P300 latency
(e.g., Cz, Fpz), that signal is subtracted from all electrodes, rigidly shifting
the entire topographic landscape. Electrodes physically close to the reference
site will appear suppressed, while distant electrodes may paradoxically appear
more activated — an entirely artifactual redistribution.

Best Reference for Occipito-Temporal Mapping
---------------------------------------------
For the P300 and occipito-temporal ERP components (e.g., P1, N170 for face
stimuli; N200/P300 for auditory deviants), the **average reference** provides
the most accurate spatial representation because:
  (a) It does not preferentially suppress any scalp region.
  (b) It preserves the bilateral occipito-temporal distribution of sensory
      components (P1, N1) while maintaining the centro-parietal P300 signature.
  (c) It is recommended by consensus guidelines for high-density EEG (>32 ch)
      topographic analysis (Luck, 2005; Hari & Puce, 2017).
The linked mastoids offer a reasonable alternative for auditory ERPs when
occipito-temporal activity is of secondary interest, but average reference
remains superior for whole-scalp comparisons.

References
----------
Biosemi (2020). ActiveTwo system description.
  https://www.biosemi.com/faq/cms&drl.htm

Donchin, E., & Hagans, C. L. (1977). A re-examination of the endogenous
  components of the event-related potential. In J. E. Desmedt (Ed.),
  Progress in Clinical Neurophysiology (Vol. 1, pp. 349–411). Karger.

Hagemann, D., Naumann, E., & Thayer, J. F. (2001). The quest for the EEG
  reference revisited: A glance from brain asymmetry research.
  Psychophysiology, 38(5), 847–857. https://doi.org/10.1111/1469-8986.3850847

Hari, R., & Puce, A. (2017). MEG-EEG Primer. Oxford University Press.

Luck, S. J. (2005). An Introduction to the Event-Related Potential Technique.
  MIT Press.

Nunez, P. L., Silberstein, R. B., Shi, Z., Carpenter, M. R., Srinivasan, R.,
  Tucker, D. M., ... & Cadusch, P. J. (1999). EEG coherency II: experimental
  comparisons of multiple measures. Clinical Neurophysiology, 110(3), 469–486.
  https://doi.org/10.1016/S1388-2457(98)00043-1

Nunez, P. L., & Srinivasan, R. (2006). Electric Fields of the Brain: The
  Neurophysics of EEG (2nd ed.). Oxford University Press.

Yao, D. (2001). A method to standardize a reference of scalp EEG recordings to
  a point at infinity. Physiological Measurement, 22(4), 693–711.
  https://doi.org/10.1088/0967-3334/22/4/305
"""

full_report = "\n".join(summary_lines) + "\n" + discussion
stats_path  = os.path.join(STATS_DIR, "part3_reference_discussion.txt")
with open(stats_path, "w", encoding="utf-8") as f:
    f.write(full_report)
print(f"\n  Discussion saved → {stats_path}")

print("\nPart 3 complete.")
