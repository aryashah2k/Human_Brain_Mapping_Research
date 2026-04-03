"""Quick statistical analysis: Frontal vs Posterior delta power (for LaTeX report)."""
import os
import numpy as np
from scipy import stats
import mne

raw = mne.io.read_raw_fif(
    os.path.join("results", "preprocessed", "sub-NORB00001_ses-1_preprocessed-raw.fif"),
    preload=True, verbose=False
)

psd = raw.compute_psd(method="welch", fmin=1, fmax=30, n_fft=400,
                       n_overlap=200, verbose=False)
freqs = psd.freqs
psd_data = psd.get_data()

# Frequency band masks
delta_mask = (freqs >= 1) & (freqs <= 4)
theta_mask = (freqs >= 4) & (freqs <= 8)
alpha_mask = (freqs >= 8) & (freqs <= 13)
beta_mask  = (freqs >= 13) & (freqs <= 30)

# Channel groups
frontal = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"]
posterior = ["P3", "P4", "Pz", "O1", "O2"]
central = ["C3", "C4", "Cz"]
temporal = ["T7", "T8", "P7", "P8"]

def band_power(ch_list, band_mask):
    idxs = [raw.ch_names.index(c) for c in ch_list if c in raw.ch_names]
    return np.array([psd_data[i, band_mask].mean() for i in idxs])

# Delta power
frontal_delta = band_power(frontal, delta_mask)
posterior_delta = band_power(posterior, delta_mask)

t_stat, p_val = stats.ttest_ind(frontal_delta, posterior_delta)
print("=== Frontal vs Posterior DELTA Power ===")
print(f"Frontal  (n={len(frontal_delta)}): M={frontal_delta.mean():.4e}, SD={frontal_delta.std():.4e}")
print(f"Posterior(n={len(posterior_delta)}): M={posterior_delta.mean():.4e}, SD={posterior_delta.std():.4e}")
print(f"t({len(frontal_delta)+len(posterior_delta)-2}) = {t_stat:.3f}, p = {p_val:.4f}")
print(f"Cohen's d = {(frontal_delta.mean()-posterior_delta.mean())/np.sqrt((frontal_delta.var()+posterior_delta.var())/2):.3f}")

# Theta power
frontal_theta = band_power(frontal, theta_mask)
posterior_theta = band_power(posterior, theta_mask)
t2, p2 = stats.ttest_ind(frontal_theta, posterior_theta)
print("\n=== Frontal vs Posterior THETA Power ===")
print(f"Frontal  (n={len(frontal_theta)}): M={frontal_theta.mean():.4e}, SD={frontal_theta.std():.4e}")
print(f"Posterior(n={len(posterior_theta)}): M={posterior_theta.mean():.4e}, SD={posterior_theta.std():.4e}")
print(f"t({len(frontal_theta)+len(posterior_theta)-2}) = {t2:.3f}, p = {p2:.4f}")
print(f"Cohen's d = {(frontal_theta.mean()-posterior_theta.mean())/np.sqrt((frontal_theta.var()+posterior_theta.var())/2):.3f}")

# All bands by region
print("\n=== Band Power by Region (mean V^2/Hz) ===")
for name, chs in [("Frontal", frontal), ("Central", central), ("Posterior", posterior), ("Temporal", temporal)]:
    d = band_power(chs, delta_mask).mean()
    t = band_power(chs, theta_mask).mean()
    a = band_power(chs, alpha_mask).mean()
    b = band_power(chs, beta_mask).mean()
    print(f"{name:10s}: delta={d:.3e}  theta={t:.3e}  alpha={a:.3e}  beta={b:.3e}")
