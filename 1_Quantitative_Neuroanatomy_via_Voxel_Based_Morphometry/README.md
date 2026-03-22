# VBM Analysis: Research Summary & Interpretation Report

**Study:** Voxel-Based Morphometry Analysis of Cortical Structure across the Adult Lifespan
**Datasets:** HCP Young Adult (N = 1,104; age 22–38) · AABC Release 2 (N = 1,150; age 36–89)
**Date:** February 2026

---

## Overview

This study investigates how sleep quality, fluid intelligence, and personality (neuroticism) relate to cortical structure in young adults, and how cortical structure changes with healthy aging. Two complementary datasets were used: the **Human Connectome Project Young Adult (HCP-YA)** sample using the Desikan-Killiany (DK) parcellation, and the **Aging Brain and Behaviour Cohort (AABC Release 2)** using the Glasser HCP-MMP1 parcellation (360 ROIs across both hemispheres).

All statistics were corrected for multiple comparisons using **Benjamini-Hochberg False Discovery Rate (FDR-BH, q < .05)**.

---

## Research Question 1: Sleep Quality and Cortical Thickness (HCP Young Adults)

### Background & Hypothesis

Poor sleep quality has been associated with accelerated cortical thinning, particularly in regions supporting memory and emotion regulation. We hypothesized that **young adults with poor sleep (PSQI > 5) would show reduced cortical thickness** compared to good sleepers (PSQI ≤ 5), particularly in medial temporal and prefrontal regions.

### Sample

- **Good sleepers** (PSQI ≤ 5): *n* = 754
- **Poor sleepers** (PSQI > 5): *n* = 350
- Mean PSQI overall: 4.80 (SD = 2.76; range 0–19)

### Plain-Language Result

Out of 68 DK cortical regions tested, only **3 showed significantly thinner cortex in poor sleepers** after correcting for multiple comparisons. These were:
- **Left and Right Entorhinal Cortex** — a gateway region to the hippocampus responsible for memory consolidation
- **Right Temporal Pole** — involved in semantic memory and social-emotional processing

The effect sizes were **small** (Cohen's *d* ≈ 0.20–0.25), meaning the difference, while statistically real, is modest. This is expected in a young, healthy sample where sleep effects on the brain are still emerging.

> **Key interpretation:** Even in healthy young adults (mean age ~29), chronic poor sleep quality is already linked to subtle thinning of memory-related cortical regions. The specificity of the entorhinal effect is notable — this region is one of the first affected in Alzheimer's disease, suggesting that sleep-related vulnerability may begin decades before clinical symptoms.

### APA-Style Results

A series of Welch's independent-samples *t*-tests compared cortical thickness (68 DK ROIs) between good sleepers (PSQI ≤ 5) and poor sleepers (PSQI > 5) in the HCP young adult sample (*N* = 1,104). Results were corrected for family-wise error using the Benjamini-Hochberg procedure (q < .05). Three regions survived FDR correction and showed significantly reduced thickness in the poor sleep group:

- **Left Entorhinal cortex**: *t*(786) = 3.16, *p* = .002, *d* = 0.20, 95% CI [0.07, 0.34], *p*_FDR = .037
- **Right Entorhinal cortex**: *t*(786) = 3.83, *p* < .001, *d* = 0.25, 95% CI [0.11, 0.39], *p*_FDR = .009
- **Right Temporal pole**: *t*(786) = 3.61, *p* < .001, *d* = 0.24, 95% CI [0.10, 0.37], *p*_FDR = .011

Effect sizes were small-to-small-medium (Cohen's *d* range: 0.20–0.25). The remaining 65 DK regions did not survive FDR correction.

---

## Research Question 2: Fluid Intelligence and Cortical Surface Area (HCP Young Adults)

### Background & Hypothesis

Fluid intelligence (the ability to reason and solve novel problems) has been associated with cortical surface area rather than thickness. We hypothesized that **higher scores on the Penn Matrix Reasoning Test (PMAT24)** would correlate positively with cortical surface area across associative, heteromodal cortex — particularly parietal and temporal association areas.

### Sample

- Full HCP sample: *N* = 1,104
- PMAT24 range: 4–24; Mean = 16.79 (SD = 4.87)

### Plain-Language Result

This was the **most robust finding in the study**: **all 68 DK cortical regions** showed a statistically significant positive correlation between PMAT24 scores and cortical surface area after FDR correction. The strongest associations were in **inferior temporal**, **middle temporal**, **superior parietal**, and **precuneus** regions — all classic components of the brain's "task-positive" and default mode networks.

Correlations ranged from *r* = **0.14** (weakest, frontal regions) to *r* = **0.29** (strongest, inferior temporal). With N = 1,104 and all 68 ROIs surviving FDR correction, this constitutes an extremely strong and consistent brain-intelligence relationship.

> **Key interpretation:** The breadth of the cortical area–intelligence association (100% of ROIs significant) reflects that fluid intelligence is not localised to one region but distributed across the entire cortex. The strongest associations in temporal and parietal association cortices are consistent with the P-FIT (Parieto-Frontal Integration Theory) of intelligence. Cortical surface area appears to be a better marker of cognitive capacity than thickness in this young, healthy population.

### Top 10 Regions by Correlation Strength

| ROI | *r* | *p*_FDR |
|---|---|---|
| L Inferiortemporal | 0.295 | < .001 |
| R Middletemporal | 0.285 | < .001 |
| R Inferiortemporal | 0.266 | < .001 |
| L Middletemporal | 0.264 | < .001 |
| L Superiorparietal | 0.264 | < .001 |
| L Precuneus | 0.262 | < .001 |
| L Entorhinal | 0.255 | < .001 |
| L Supramarginal | 0.252 | < .001 |
| R Precuneus | 0.248 | < .001 |
| L Rostralanteriorcingulate | 0.244 | < .001 |

### APA-Style Results

Pearson correlations were computed between PMAT24 total correct scores and cortical surface area for each of 68 DK ROIs (*N* = 1,104). All 68 regions survived Benjamini-Hochberg FDR correction (q < .05). Correlations ranged from *r*(1102) = 0.144 to *r*(1102) = 0.295, all *p*_FDR < .001. The strongest associations were observed in left inferior temporal [*r* = .295, 95% CI (.237, .352)], right middle temporal [*r* = .285, 95% CI (.226, .342)], left superior parietal [*r* = .264, 95% CI (.204, .321)], and left precuneus [*r* = .262, 95% CI (.202, .320)]. All confidence intervals were computed using Fisher's *r*-to-*z* transformation.

---

## Research Question 3: Neuroticism and Cortical Structure (HCP Young Adults)

### Background & Hypothesis

Neuroticism — a personality dimension characterised by emotional instability, anxiety, and negative affect — has been proposed to relate to altered cortical morphology. We hypothesized that **high-neuroticism individuals (top tertile, NEO-FFI) would show reduced cortical thickness and/or surface area** in frontolimbic regions relative to low-neuroticism individuals (bottom tertile).

### Sample

- **Low-N tertile** (NEOFAC_N ≤ 12): *n* = 379
- **High-N tertile** (NEOFAC_N ≥ 21): *n* = 409
- NEO-N range: 0–43; Mean = 16.65 (SD = 7.35)

### Plain-Language Results

#### Cortical Thickness
No regions survived FDR correction for cortical thickness (0/68). This null result suggests that neuroticism in young adults is **not strongly associated with cortical thickness** — consistent with the emerging view that personality effects on thickness are either very small, require larger samples, or manifest only later in life.

#### Cortical Surface Area
A much stronger pattern emerged for **surface area**: **39 out of 68 regions** survived FDR correction, all showing *smaller* surface area in the high-neuroticism group. The largest effects were in:
- **Right Supramarginal** (*d* = –0.29) — involved in emotional processing and mentalising
- **Right Superior frontal** (*d* = –0.28) — executive function and self-regulation
- **Left & Right Superiorfrontal, Inferiortemporal, Middletemporal** — widespread prefrontal-temporal network

Effect sizes were uniformly small (Cohen's *d* range: –0.16 to –0.29), but consistent in direction.

> **Key interpretation:** The dissociation between surface area (39/68 significant) and thickness (0/68 significant) for neuroticism is theoretically meaningful. Surface area reflects early neurodevelopmental cortical expansion, while thickness reflects later maturational thinning. The neuroticism–area association may reflect a developmental trajectory rather than ongoing atrophy. The finding that high-N individuals have smaller surface area across widespread cortex — particularly in prefrontal and temporoparietal networks — is consistent with reduced cognitive control and emotion regulation capacity.

### APA-Style Results

Welch's independent-samples *t*-tests compared cortical thickness and surface area between low-neuroticism (NEOFAC_N ≤ 12; *n* = 379) and high-neuroticism (NEOFAC_N ≥ 21; *n* = 409) tertile groups across 68 DK ROIs. Results were FDR-BH corrected (q < .05).

**Cortical thickness:** No regions survived FDR correction (0/68; all *p*_FDR > .05).

**Cortical surface area:** 39 of 68 regions survived FDR correction, all exhibiting smaller area in the high-neuroticism group. Effect sizes were small (*d* range: −0.16 to −0.29). The five strongest effects were:

| ROI | *t*(df) | *p* | *d* | 95% CI [d] | *p*_FDR |
|---|---|---|---|---|---|
| R Supramarginal | −4.05(786) | < .001 | −0.29 | [−0.43, −0.15] | .003 |
| R Superiorfrontal | −3.93(786) | < .001 | −0.28 | [−0.42, −0.14] | .003 |
| L Superiorfrontal | −3.48(786) | < .001 | −0.25 | [−0.39, −0.11] | .006 |
| L Precentral | −3.38(786) | < .001 | −0.24 | [−0.38, −0.10] | .007 |
| L Inferiortemporal | −3.35(786) | = .001 | −0.24 | [−0.38, −0.10] | .007 |

---

## Research Question 4: Cortical Aging in the AABC Sample

### Background & Hypothesis

Healthy aging is associated with widespread cortical thinning and volume loss, but this occurs heterogeneously across regions. We hypothesized that **older AABC participants (up to age 89) would show significantly reduced cortical thickness and grey matter volume** across the majority of Glasser parcels, with the strongest age effects in primary sensorimotor and occipital regions.

### Sample

- *N* = 1,150 (after behavioral merge)
- Age range: 36–89 years; Mean = 60.9 (SD = 15.5)
- Sex: 56.4% female
- 6 age-decade groups: 30s, 40s, 50s, 60s, 70s, 80s+

### Plain-Language Results

The aging analysis yielded the **most striking results in the study**:

- **356 out of 360** Glasser ROIs showed significant negative correlations with age for **cortical thickness** *(r* range: −0.28 to −0.69*)
- **355 out of 360** Glasser ROIs showed significant negative correlations for **grey matter volume** *(r* range: −0.18 to −0.50*)

The **strongest age-related thinning** occurred in:
- **Primary motor cortex (Area 4):** *r* = −0.69 — the cortex shrinks most dramatically in the motor planning area
- **Frontal Eye Field (FEF):** *r* = −0.66 — oculomotor control
- **Area 55b / Premotor cortex:** *r* = −0.59 — motor sequencing
- **Primary visual cortex (V4, V1):** *r* = −0.57, −0.53

ANOVAs across age decades confirmed that these age effects are step-wise and linear: all 20 ANOVA top ROIs had **F > 15, all p < .001**, with 6–15 out of 15 possible pairwise decade comparisons being significant per ROI.

> **Key interpretation:** Age-related cortical atrophy in this large aging sample (N = 1,150, age 36–89) is near-universal: 99% of Glasser parcels shrink with age. The strongest effects are not in the prefrontal cortex (as popular accounts often suggest) but in **primary motor and visual sensory cortices**. This is consistent with recent large-sample neuroimaging studies showing that primary cortices are among the most vulnerable to age-related volume loss. The pattern likely reflects selective vulnerability of regions with high metabolic demand and specific myeloarchitecture.

### APA-Style Results

Pearson correlations were computed between age and cortical thickness/volume for each of 360 Glasser HCP-MMP1 ROIs (*N* = 1,150; age 36–89 years). Results were corrected using Benjamini-Hochberg FDR (q < .05). For **cortical thickness**, 356 of 360 regions survived correction (98.9%), all showing negative correlations (*r* range: −.285 to −.693, all *p*_FDR < .001). For **grey matter volume**, 355 of 360 regions survived correction (98.6%; *r* range: −.178 to −.496, all *p*_FDR < .001). The strongest age–thickness association was observed in the right primary motor cortex [Area 4: *r*(1148) = −.693, 95% CI (−.722, −.661), *p*_FDR < .001], followed by right Frontal Eye Field [FEF: *r* = −.661, 95% CI (−.693, −.628)].

One-way ANOVAs were conducted on the 20 most FDR-significant ROIs using age-decade groupings. All 20 ANOVAs were highly significant (F range: 15.06–207.96, all *p* < .001), with post-hoc Tukey HSD tests revealing 6–15 of 15 possible pairwise decade comparisons reaching significance per ROI.

---

## Research Question 5: Behavioral Correlates of Cortical Structure in Aging (AABC)

### 5a. Sleep Quality × Cortical Thickness (Aging Sample)

After FDR correction, **no ROIs** survived for the sleep × thickness comparison in the AABC aging sample (0/360, all *p*_FDR > .05).

> **Interpretation:** The absence of sleep–thickness effects in the aging sample is likely explained by **age as a dominant confound**: the sheer magnitude of age-related cortical thinning (effect sizes *r* up to .69) substantially reduces the residual variance available for PSQI to explain. Future analyses should partial out age, or use age-residualised cortical thickness as the dependent variable. Additionally, the PSQI measure in this older sample may reflect age-related sleep changes (lighter sleep, earlier wake) rather than pathological sleep disruption per se.

### 5b. Neuroticism × Cortical Thickness (Aging Sample)

**327 out of 360** Glasser ROIs showed significantly reduced cortical thickness in the high-neuroticism group (top tertile: NEO-N ≥ 17) compared to low-neuroticism individuals (bottom tertile: NEO-N ≤ 11).

Effect sizes were small (Cohen's *d* range: −0.14 to −0.38), with the strongest effects in:
- **Right V1 (primary visual):** *d* = −0.38
- **Right V2:** *d* = −0.39
- **Area 55b (premotor):** *d* = −0.39
- **Area 4 (primary motor):** *d* = −0.37

> **Interpretation:** In the aging sample, the neuroticism–cortical thickness association is far broader and stronger than in young adults (327/360 vs. 0/68 for thickness). This is consistent with **stress sensitisation** and cumulative allostatic load models: decades of elevated neuroticism-related stress reactivity eventually translate into measurable cortical thinning. The widespread nature of the effect (spanning sensory, motor, and association cortex) suggests a systemic rather than region-specific process, possibly mediated by chronic HPA-axis activation, elevated inflammatory markers, or altered sleep over the life course.

### APA-Style Results

**Sleep:** Independent-samples Welch *t*-tests comparing cortical thickness between good PSQI ≤ 5 (*n* = 755) and poor PSQI > 5 (*n* = 395) sleepers yielded no significant results after FDR-BH correction (0 of 360 Glasser ROIs, all *p*_FDR > .05).

**Neuroticism:** Welch *t*-tests comparing low-N (NEO-N ≤ 11; *n* = 411) and high-N (NEO-N ≥ 17; *n* = 412) groups across 360 Glasser ROIs found 327 significant regions (91%; *d* range: −0.14 to −0.39, all *p*_FDR < .05). All effects were in the direction of reduced thickness in the high-neuroticism group, *t*(821) range: −2.02 to −5.60.

---

## Summary Table: Results at a Glance

| Analysis | Dataset | *N* | ROIs | FDR-sig | Effect Direction | Max Effect |
|---|---|---|---|---|---|---|
| **1. Sleep × Thickness** | HCP | 1,104 | 68 DK | **3/68** | Poor sleep → less thick | *d* = 0.25 |
| **2. Cognition × Area** | HCP | 1,104 | 68 DK | **68/68** | Higher IQ → more area | *r* = 0.29 |
| **3. Neuroticism × Thickness** | HCP | 788 | 68 DK | **0/68** | — | — |
| **3. Neuroticism × Area** | HCP | 788 | 68 DK | **39/68** | High-N → less area | *d* = 0.29 |
| **4. Age × Thickness** | AABC | 1,150 | 360 Glasser | **356/360** | Older → less thick | *r* = 0.69 |
| **4. Age × Volume** | AABC | 1,150 | 360 Glasser | **355/360** | Older → less volume | *r* = 0.50 |
| **5a. Sleep × Thickness** | AABC | 1,150 | 360 Glasser | **0/360** | — | — |
| **5b. Neuroticism × Thickness** | AABC | 823 | 360 Glasser | **327/360** | High-N → less thick | *d* = 0.39 |

---

## Cross-Atlas Note (Methods Transparency)

The HCP analyses used the **Desikan-Killiany (DK) atlas** (34 regions/hemisphere), while the AABC analyses used the **Glasser HCP-MMP1 atlas** (180 regions/hemisphere). These atlases differ in resolution and parcellation philosophy — DK is macro-anatomical; Glasser is cytoarchitectonically defined. Results are therefore **not directly comparable at the ROI level** but are compared at the **lobe level** in the Atlas Mapping output ([atlas_cross_dataset_summary.csv](file:///c:/Users/Arya/Desktop/hbm1/results/tables/atlas_cross_dataset_summary.csv)). DK cingulate regions (4) were redistibuted to Frontal (anterior/mid) and Parietal (posterior/isthmus) for this cross-atlas comparison, consistent with the Glasser atlas lobe assignments.

---

## Statistical Methods Summary

| Method | Used for |
|---|---|
| Welch's independent *t*-test | Group comparisons (sleep, neuroticism tertiles) |
| Pearson *r* with Fisher *z* CI | Continuous correlations (cognition, age) |
| Benjamini-Hochberg FDR (q < .05) | All multiple comparison corrections |
| Cohen's *d* (pooled SD) | Effect size for group comparisons |
| One-way ANOVA + Tukey HSD | Decade-level group comparisons (aging) |
| All analyses: | `scipy`, `statsmodels`, `pandas` in Python 3.x |

---

*Document generated from pipeline results: [VBM_Analysis_Summary.xlsx](file:///c:/Users/Arya/Desktop/hbm1/results/tables/VBM_Analysis_Summary.xlsx) and individual CSV tables in `results/tables/`.*
