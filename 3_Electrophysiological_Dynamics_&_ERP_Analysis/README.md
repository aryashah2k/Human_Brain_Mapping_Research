# EEG Analysis Report: Electrophysiological Dynamics & ERP Analysis
### P300 Auditory Oddball Paradigm — Single-Subject Dataset (ds003061, sub-001)

---

## Dataset Overview

| Parameter | Value |
|---|---|
| System | Biosemi Active Two |
| Channels | 64 EEG (10-20 standard biosemi64 cap) |
| Sampling rate | 256 Hz |
| Power line | 50 Hz (Rishikesh, India) |
| Task | Auditory P300 — eyes closed |
| Stimuli | Standard (500 Hz, 70%), Oddball (1000 Hz, 15%), Noise (white noise, 15%) |
| Instructions | Press button on oddball; ignore others |
| Recording duration | ~758 s (~12.6 min) |

---

## Part 1: Data Cleaning & Preprocessing

### 1.1 Pipeline Summary

The raw `.set` file was loaded using MNE-Python. All 16 auxiliary channels (EXG1–8, GSR1–2, Erg1–2, Resp, Plet, Temp) were dropped, retaining 64 EEG channels. A standard biosemi64 montage was applied.

| Step | Details |
|---|---|
| Bad channel detection | Variance z-score threshold (> 3.5 SD from median) |
| Bad channels identified | **T7** |
| Bandpass filter | 1–40 Hz (FIR, firwin) |
| Notch filter | 50 Hz (power line) |
| Interpolation | Spherical spline interpolation of T7 |
| ICA | Infomax extended, 20 components |
| Components excluded | **IC2, IC10, IC13, IC17, IC18** (ocular + muscle) |
| Re-reference | Average reference |

### 1.2 Bad Channel: T7

T7 (left temporal) was flagged as bad via automated variance z-score analysis. Left temporal channels are common sites for electrode bridging, impedance issues, and jaw/muscle movement artifacts, particularly in participants wearing headphones or sitting asymmetrically. T7 was successfully interpolated using spherical splines from its neighbours.

### 1.3 ICA Artifact Correction

Five independent components were removed:

- **IC2**: Likely eye-blink artifact — identified via high correlation with Fp1/Fp2 (frontal pole electrodes). Eye blinks generate large, stereotyped deflections (~100–200 µV) with a characteristic frontal topography and slow time course.
- **IC10, IC13**: Likely lateral eye movement (saccade) artifacts — these typically project to frontal-lateral channels (AF7/AF8, F7/F8) and show step-like or sharp transient waveforms.
- **IC17, IC18**: Likely muscle/EMG artifacts — muscle components characteristically show broadband high-frequency (>20 Hz) power spectra with focal topographies at temporal or peripheral scalp regions.

Removing these 5 components substantially cleaned the signal without discarding trial data.

### 1.4 PSD Comparison: Raw vs. Preprocessed

**Key observations from `part1_psd_comparison.png`:**

- **Low-frequency power (1–4 Hz, delta)**: The preprocessed signal shows reduced slow-drift power compared to raw, confirming that the 1 Hz high-pass cutoff effectively removed DC drift and movement-related slow fluctuations.
- **50 Hz notch**: A visible reduction at 50 Hz in the preprocessed trace confirms successful removal of power-line interference, which is particularly important for recordings in India (50 Hz grid).
- **Alpha peak (~10 Hz)**: Both traces show a prominent alpha peak around 9–11 Hz, consistent with the eyes-closed resting/task state. This is preserved after preprocessing, confirming the filter did not distort physiologically relevant oscillations.
- **High-frequency power (>20 Hz)**: The preprocessed signal shows attenuated power above ~20 Hz compared to raw, reflecting the combined effect of the 40 Hz low-pass and ICA muscle component removal.
- **Overall power reduction**: The preprocessed trace sits systematically below raw across most frequencies, reflecting artifact removal — not signal loss.

### 1.5 Time-Series Comparison: Raw vs. Preprocessed

**Key observations from `part1_timeseries_comparison.png`:**

- **Fp1 (frontal pole)**: The raw trace shows large, intermittent transient spikes (eye blinks) that are entirely absent in the preprocessed trace. ICA successfully isolated and removed the blink topography.
- **Fz (frontal midline)**: Slow drifts visible in raw are removed; the preprocessed trace shows cleaner oscillatory activity.
- **Cz (vertex)**: Both traces look relatively clean, as Cz is distant from artifact sources. The preprocessed trace shows slightly reduced variance from muscle artifact removal.
- **Pz (parietal midline)**: The channel of primary interest for P300 shows minimal artifact contamination in raw, and the preprocessed trace is nearly identical — confirming that preprocessing did not distort the target ERP region.
- **Oz (occipital)**: Residual slow drifts are removed; alpha oscillations remain clearly visible, consistent with the eyes-closed paradigm.

---

## Part 2: Epoching, ERP Analysis & Statistical Results

### 2.1 Epoching Summary

| Parameter | Value |
|---|---|
| Epoch window | −200 to +800 ms |
| Baseline | −200 to 0 ms |
| Rejection threshold | Peak-to-peak amplitude > 150 µV |
| Total events parsed | 747 |
| Epochs retained | 741 (99.2% retention) |
| Standard retained | 518 / 522 (99.2%) |
| Oddball retained | 112 / 113 (99.1%) |
| Noise retained | 111 / 112 (99.1%) |

The extremely high epoch retention rate (99.2%) indicates that the preprocessing pipeline effectively removed artifacts from the continuous data, leaving clean epochs. Only 6 trials were rejected due to residual amplitude transients exceeding the 150 µV threshold.

### 2.2 ERP Waveforms by Condition

**From `part2_erp_butterfly_conditions.png` and `part2_erp_joint_*.png`:**

#### Standard Condition (500 Hz, frequent)
The standard condition shows a characteristic auditory ERP morphology:
- **N1 (~100 ms)**: A negative deflection over fronto-central scalp reflecting early auditory cortex response to the sound onset. Present but modest in amplitude, consistent with the frequent, unattended nature of standard stimuli.
- **P2 (~180–200 ms)**: A small positive component over central sites.
- **No prominent P300**: The standard (non-target) condition does not engage the target-detection mechanism, producing a flat or slightly positive waveform at parietal sites after 250 ms (M = +0.231 µV, SD = 2.86).

#### Oddball Condition (1000 Hz, target)
- **N1/P2**: Present and slightly larger than standard, reflecting the physical deviance (different pitch) and attentional capture.
- **P300 (300–500 ms)**: A large, robust positivity peaking around **406 ms** at centro-parietal sites (Pz, CPz), reaching **M = +4.658 µV** (SD = 3.17). This is the canonical P3b component — a hallmark of conscious target detection, context updating, and working memory engagement.
- **Topography**: The topomaps at 300–500 ms show a clear centro-parietal maximum with a broad posterior distribution, consistent with the P3b generator in the posterior parietal cortex and hippocampal/temporal sources.

#### Noise Condition (white noise, distractor)
- **N1**: Larger than standard but smaller than oddball, reflecting deviance detection for the unexpected white noise stimulus.
- **P300 window**: A small but positive deflection (M = +1.081 µV, SD = 3.07) at parietal sites. This may reflect a partial P3a (novelty P300) — an earlier, more frontally distributed response to novel, task-irrelevant stimuli — rather than the target-specific P3b.
- **Importantly**: The noise P300 amplitude is intermediate between standard and oddball, consistent with the noise being surprising (like oddball) but not task-relevant (unlike oddball).

### 2.3 ERP at Pz — P300 Component

**From `part2_erp_Pz_P300.png`:**

The three conditions diverge clearly after ~200 ms at Pz:
- Oddball rises sharply from ~200 ms, peaks at **406.2 ms** at +4.66 µV
- Noise shows a modest rise peaking ~300–350 ms (consistent with frontal P3a)
- Standard remains near baseline throughout the 300–500 ms window

The 300–500 ms window (shaded gold) captures the P3b maximum for the oddball, validating the chosen analysis window.

### 2.4 Statistical Analysis

**Channels**: Pz, CPz, P3, P4, Cz (mean amplitude, 300–500 ms window)

#### Descriptive Statistics

| Condition | n | M (µV) | SD | SEM |
|---|---|---|---|---|
| Standard | 518 | +0.231 | 2.863 | 0.126 |
| Oddball | 112 | +4.658 | 3.170 | 0.300 |
| Noise | 111 | +1.081 | 3.067 | 0.291 |

#### One-Way ANOVA

A one-way independent ANOVA revealed a highly significant main effect of stimulus type on P300 mean amplitude, **F(2, 738) = 103.86, p < .001**. This indicates that the three stimulus conditions produced reliably different neural responses in the 300–500 ms window.

#### Post-Hoc Comparisons (Bonferroni-corrected α = .017)

| Comparison | t | df | p (corrected) | Cohen's d |
|---|---|---|---|---|
| Oddball vs. Standard | 14.53 | 628 | < .001 | **1.46** (large) |
| Oddball vs. Noise | 8.53 | 221 | < .001 | **1.14** (large) |
| Standard vs. Noise | −2.80 | 627 | .016 | — |

#### Interpretation

The oddball condition elicited a **significantly and substantially larger P300 amplitude** compared to both standard (d = 1.46) and noise (d = 1.14) conditions. These are large effect sizes by Cohen's (1988) conventional thresholds (d > 0.8 = large), indicating a robust and neurophysiologically meaningful difference.

The Standard vs. Noise comparison also reached significance after Bonferroni correction (p = .016 < .017), consistent with the noise condition eliciting a small but reliable novelty response (P3a) absent for the standard.

These results are fully consistent with the **P3b** interpretation: the parietal P300 is selectively enlarged for task-relevant target stimuli (oddballs requiring a button press) relative to non-target standards and distractors. This reflects the neural substrates of **attentional resource allocation, context updating, and working memory engagement** in response to target stimuli (Polich, 2007; Picton, 1992).

---

## Part 3: Reference Scheme Comparison

### 3.1 P300 Amplitude & Peak Latency by Reference

| Reference Scheme | Mean Amplitude at Pz (300–500 ms) | Peak Latency at Pz |
|---|---|---|
| Average Reference | +4.384 µV | 406.2 ms |
| Cz Reference | +1.280 µV | 476.6 ms |
| Linked Mastoids (TP7+TP8) | +5.279 µV | 398.4 ms |
| Fpz (Nasion Proxy) | +9.471 µV | 406.2 ms |
| No Re-reference (CMS/DRL) | −1.013 µV | 250.0 ms |

### 3.2 Interpretation of Each Reference Scheme

#### Average Reference (+4.384 µV, 406.2 ms)
The average reference subtracts the mean of all 64 electrodes from each channel. For a dense 64-channel array with roughly symmetric coverage, this approximates a reference-free (zero-potential) solution. The centro-parietal P300 positivity is clearly expressed (+4.384 µV at Pz), and the topography shows a clean posterior maximum with complementary frontal negativity — the expected polarity inversion under average reference. This is the recommended reference for topographic analysis of high-density EEG (Luck, 2005; Nunez & Srinivasan, 2006).

#### Cz Reference (+1.280 µV, 476.6 ms)
Cz sits near the vertex — directly within the P300 topographic maximum. Referencing to Cz subtracts the P300 signal itself from all other channels, artificially **suppressing the P300 amplitude** (from +4.384 µV to +1.280 µV — a 71% reduction) and **delaying the apparent peak latency** by ~70 ms (406 → 477 ms). The topographic map will show a distorted centro-parietal distribution because the reference itself carries substantial P300 signal. Cz is a poor choice for P300 analysis as it confounds amplitude and latency estimation.

#### Linked Mastoids (TP7+TP8) (+5.279 µV, 398.4 ms)
The mastoid electrodes (TP7/TP8) are relatively inactive during the P300 window (~300–500 ms), making them a reasonable reference for auditory ERP research. The amplitude (+5.279 µV) is slightly inflated compared to average reference because the mastoids carry slightly negative potential at this latency (subtracting a negative value increases the apparent positive amplitude at Pz). The peak latency (398.4 ms) is comparable to average reference. The mastoid reference provides a physiologically sensible result for central and parietal channels but may distort temporal and occipito-temporal topography due to proximity of TP7/TP8 to those regions.

#### Fpz / Nasion Reference (+9.471 µV, 406.2 ms)
The Fpz amplitude is dramatically inflated (+9.471 µV — more than double the average reference estimate). This is because Fpz is highly susceptible to **eye-blink and frontal muscle residuals** — even after ICA, frontal electrodes carry more noise. Using Fpz as a reference injects this frontal noise into every channel subtractively, creating an artifactual large-amplitude posterior positivity. The peak latency coincides with average reference (406.2 ms) only because the Fpz is relatively flat at this latency, but the overall ERP morphology and topographic distribution will be distorted. Nasion/Fpz reference is generally **not recommended** for cognitive ERP research (Nunez et al., 1999).

#### No Re-reference / CMS-DRL (−1.013 µV, 250.0 ms)
Without re-referencing, the data retains the Biosemi CMS/DRL offset. The P300 appears **negative** (−1.013 µV) at Pz — a polarity inversion — and the apparent peak latency is implausibly early (250 ms, likely corresponding to the N2 negativity rather than P300). This demonstrates that without re-referencing, the topographic interpretation is entirely unreliable: the CMS electrode (typically near POz) acts as an active reference that inverts the polarity of posterior channels relative to the true scalp distribution. Biosemi explicitly recommends offline re-referencing for all analysis (Biosemi, 2020).

### 3.3 How Reference Choice Shifts the 'Center' of Electrical Activity

The fundamental issue is that **EEG voltage is always a difference measure**, not an absolute potential. Every scalp map is the spatial distribution of (electrode − reference). When the reference electrode itself is located within an active field:

1. **The reference site appears forced to zero** in the topographic map, creating an artificial "neutral" region where there is actually signal.
2. **All other electrodes' values are rigidly shifted** by the reference's true potential, globally rotating the topographic landscape.
3. **The apparent 'center' of activity** shifts toward regions distant from the reference, not toward the true neural generator.

For the P300:
- **Cz reference** suppresses the centro-parietal maximum (the reference is inside the P300 field) → apparent center shifts peripherally
- **Fpz reference** makes all channels appear more positive relative to Fpz → inflates posterior positivity, the apparent center shifts posteriorly and the amplitude is distorted
- **Average reference** distributes the reference signal symmetrically → preserves the true centro-parietal topographic maximum
- **Mastoid reference** is near-zero at P300 latency → produces a topography close to average reference but with mild anterior inflation

### 3.4 Best Reference for Occipito-Temporal Sensor Mapping

For mapping **occipito-temporal ERP components** (e.g., P1, N170, N2, P300 temporal contributors at PO7/PO8), the **average reference** is optimal because:

1. It does not preferentially suppress any scalp region, including occipital and temporal sites.
2. With 64 electrodes providing dense coverage, the average approximates a reference-free estimate well.
3. It preserves the bilateral occipito-temporal distribution of early sensory components (P1 at ~100 ms, N1 at ~100 ms bilateral) as well as the centro-parietal P300 maximum.
4. Mastoid reference partially suppresses TP7/TP8 and adjacent temporal channels, making it suboptimal for occipito-temporal analysis specifically.

The average reference is the consensus recommendation for high-density (>32 channel) EEG topographic analysis in cognitive neuroscience (Luck, 2005; Hari & Puce, 2017).

---

## Summary of Key Findings

| Finding | Value/Result |
|---|---|
| Bad channel | T7 (left temporal, variance outlier) |
| ICA components removed | 5 (IC2, IC10, IC13, IC17, IC18 — ocular + muscle) |
| Epoch retention | 741/747 (99.2%) |
| P300 peak latency (oddball, Pz) | **406.2 ms** |
| P300 amplitude: Oddball | **+4.658 µV** (M, 300–500 ms, Pz/CPz/P3/P4/Cz) |
| P300 amplitude: Standard | +0.231 µV |
| P300 amplitude: Noise | +1.081 µV |
| ANOVA main effect | F(2, 738) = 103.86, **p < .001** |
| Oddball vs. Standard effect size | Cohen's d = **1.46** (large) |
| Oddball vs. Noise effect size | Cohen's d = **1.14** (large) |
| Best reference for topography | **Average reference** |

---

## References

Biosemi (2020). *ActiveTwo system description — CMS/DRL*. https://www.biosemi.com/faq/cms&drl.htm

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

Donchin, E., & Hagans, C. L. (1977). A re-examination of the endogenous components of the event-related potential. In J. E. Desmedt (Ed.), *Progress in Clinical Neurophysiology* (Vol. 1, pp. 349–411). Karger.

Hagemann, D., Naumann, E., & Thayer, J. F. (2001). The quest for the EEG reference revisited: A glance from brain asymmetry research. *Psychophysiology, 38*(5), 847–857. https://doi.org/10.1111/1469-8986.3850847

Hari, R., & Puce, A. (2017). *MEG-EEG Primer*. Oxford University Press.

Luck, S. J. (2005). *An Introduction to the Event-Related Potential Technique*. MIT Press.

Nunez, P. L., Silberstein, R. B., Shi, Z., Carpenter, M. R., Srinivasan, R., Tucker, D. M., ... & Cadusch, P. J. (1999). EEG coherency II: experimental comparisons of multiple measures. *Clinical Neurophysiology, 110*(3), 469–486. https://doi.org/10.1016/S1388-2457(98)00043-1

Nunez, P. L., & Srinivasan, R. (2006). *Electric Fields of the Brain: The Neurophysics of EEG* (2nd ed.). Oxford University Press.

Picton, T. W. (1992). The P300 wave of the human event-related potential. *Journal of Clinical Neurophysiology, 9*(4), 456–479.

Polich, J. (2007). Updating P300: An integrative theory of P3a and P3b. *Clinical Neurophysiology, 118*(10), 2128–2148. https://doi.org/10.1016/j.clinph.2007.04.019

Yao, D. (2001). A method to standardize a reference of scalp EEG recordings to a point at infinity. *Physiological Measurement, 22*(4), 693–711. https://doi.org/10.1088/0967-3334/22/4/305
