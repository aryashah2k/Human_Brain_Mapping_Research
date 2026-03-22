# Resting-State Functional Connectivity & Predictive Modeling
## Analysis Report — Assignment 2

**Dataset:** Human Connectome Project (HCP) + AABC Release 2  
**Atlases:** Schaefer 400-Parcel 17-Network (Part 1), Glasser HCP-MMPv1 379-ROI (Parts 2–3)  
**Analysis Date:** March 2026

---

## Part 1: Parcellation & Connectivity Matrix Construction

### 1.1 Time-Series Extraction

The preprocessed HCP resting-state CIFTI file (`rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii`) contains **1200 timepoints × 91,282 grayordinates** at a TR of 0.72 s (~14.4 min of resting-state data). The Schaefer 400-parcel atlas was applied by mapping each grayordinate's vertex index to its assigned network label, yielding **58,606 of 91,282 grayordinates** assigned to a parcel. The remaining ~36% are subcortical grayordinates not covered by this cortical-only atlas, which is expected behavior. Mean BOLD signal per parcel was extracted to produce a **(400 parcels × 1200 timepoints)** matrix.

### 1.2 Time-Series Characteristics (Figure: `timeseries_20rois.png`)

The 20 representative parcels, one per 17-network community, reveal several key observations:

**Similarities across ROIs:**
- All time series exhibit broadband, low-frequency fluctuations (0.01–0.1 Hz) characteristic of resting-state BOLD, superimposed on higher-frequency noise.
- Signal amplitude (z-scored) is uniformly bounded within approximately ±2–3 standard deviations across all networks, indicating consistent preprocessing (HPF at 2000 s cutoff, ICA-FIX denoising).
- Slow-wave oscillatory structure (~30–60 s periods) is visible in nearly all parcels, reflecting the dominant infra-slow frequency of intrinsic neural activity.

**Differences across ROIs:**
- **Visual networks (VisCent, VisPeri):** The top two traces show notably lower signal variance and smoother time courses compared to association cortex, consistent with primary sensory areas having stronger, more constrained connectivity.
- **Somatomotor (SomMotA/B):** Slightly higher-amplitude fluctuations, reflecting the well-known high signal-to-noise of motor cortex in fMRI.
- **Default Mode (DefaultA/B/C) and Control (ContA/B):** Display prominent slow oscillations and more variable amplitude, consistent with these networks supporting internally-directed cognition with less trial-locked structure.
- **Limbic (LimbicA/B):** Notably more irregular, low-amplitude traces — reflecting the known lower BOLD signal quality in orbitofrontal/temporal pole regions due to susceptibility artifacts.
- **TempPar (Temporoparietal):** Shows high variability and occasional large deflections, consistent with this region's role at the intersection of multiple large-scale networks.

The diversity of temporal signatures across networks — from the smooth, constrained visual cortex traces to the variable, high-amplitude default mode traces — directly reflects the functional specialization of these regions during rest.

### 1.3 Pearson Correlation Matrix (Figure: `heatmap_pearson.png`)

The 400×400 Pearson correlation matrix, with parcels ordered by their 17-network assignment, reveals a rich modular structure:

**r-value range: −0.450 to +1.000**

**Key visible structures:**

- **Diagonal blocks (within-network correlation):** Bright red squares along the diagonal indicate strong positive within-network correlations. The largest and most prominent blocks are:
  - *Visual (VisCent/VisPeri):* Two distinct but strongly inter-correlated blocks at the top-left — central and peripheral visual areas, reflecting retinotopic organization.
  - *Somatomotor (SomMotA/B):* Adjacent red blocks, with SomMotA (primary motor/somatosensory) and SomMotB (premotor/SMA) showing both strong within-block and moderate between-block coupling.
  - *Default Mode (DefaultA/B/C):* Large red blocks in the lower-right quadrant, with DefaultA showing the strongest internal coherence.
  - *Control/Frontoparietal (ContA/B/C):* Prominent blocks reflecting lateral prefrontal–parietal coupling.

- **Off-diagonal positive coupling (between-network integration):**
  - Strong positive cross-network coupling between **Default and Control** networks — consistent with decades of literature showing these networks co-activate during internally-directed cognition and are jointly anti-correlated with task-positive networks.
  - **DorsAttnA/B ↔ SomMot** cross-correlation: visible positive off-diagonal tiles, reflecting the dorsal attention–sensorimotor coupling for spatial orienting.

- **Negative correlations (blue tiles, r ~ −0.3 to −0.45):**
  - Visible between **Visual/SomMot** and **Default** networks — the classic task-positive vs. task-negative anticorrelation described by Fox et al. (2005). Though global signal regression was not explicitly applied here (HCP uses ICA-FIX), the anticorrelation is preserved at a modest magnitude.
  - **Limbic ↔ Dorsal Attention:** Negative coupling consistent with Limbic regions (orbitofrontal, parahippocampal) being anti-correlated with goal-directed attentional networks.

- **The SalVentAttnA/B blocks** show intermediate positive coupling with Default, consistent with the ventral attention (salience) network acting as a switch between default and dorsal attention networks.

### 1.4 Fisher z-Transformed Matrix (Figure: `heatmap_fisherz.png`)

**z-value range: −0.484 to +1.376** (diagonal set to 0)

The Fisher r-to-z transformation (`z = arctanh(r)`) has two main effects visible in the heatmap:

1. **Enhanced contrast for strong correlations:** The high-r diagonal blocks (r > 0.7) are stretched in z-space (z > 0.87), making within-network coherence appear more intense (darker red) compared to the raw r matrix. This is mathematically expected as arctanh is superlinear near ±1.

2. **Variance stabilization:** The z-matrix is more appropriate for statistical inference (e.g., group comparisons, t-tests) because z-scores are approximately normally distributed whereas r-values are bounded and skewed near ±1.

The network block structure is qualitatively identical to the r-matrix, but the visual contrast between strong within-network and weak between-network connections is heightened. The Default Mode network blocks in the z-matrix appear particularly prominent, with z > 0.8 within-DefaultA, reflecting very high temporal coherence among medial prefrontal, posterior cingulate, and angular gyrus parcels.

---

## Part 2: Lifespan Connectivity Analysis

### 2.1 Dataset Overview

| Age Group | N subjects | Age Range |
|-----------|-----------|-----------|
| 36–50     | 415       | Young middle age |
| 51–65     | 412       | Older middle age |
| 66–80     | 334       | Young-old |
| 81+       | 229       | Old-old |

**Total subjects:** 1,390 (V1 visit only, intersecting demographics with connectivity data)  
**Connectivity measure:** Full covariance (not normalized correlation), as provided by the HCP AABC pipeline using the Glasser HCP-MMPv1 atlas (379 ROIs).

### 2.2 Age Group Mean Connectivity Matrices (Figures: `heatmap_group_*.png`)

Comparing the 36–50 vs. 81+ mean connectivity matrices reveals a dramatic and visually striking age-related change:

**36–50 group (young middle age):**
- Rich, high-amplitude block structure with covariance values ranging up to ~700 (strong red) and down to ~−700 (strong blue).
- Dense off-diagonal positive coupling across many ROI pairs, with well-defined network clusters identifiable as concentrated red squares.
- The matrix is dense and richly structured — many ROI pairs show strong positive or negative covariance, indicating high global integration.

**81+ group (old-old):**
- Markedly reduced amplitude across the entire matrix — color saturation is substantially lower with fewer extreme values (the same color scale is used for fair comparison).
- The modular block structure is visually less sharp: within-network clusters are smaller in magnitude and less clearly delineated.
- Off-diagonal connectivity is noticeably reduced, with fewer strongly positive or negative pairs, reflecting a loss of both within-network cohesion and between-network integration.
- A visible increase in diffuse, near-zero connectivity patches, suggesting a general trend toward network dedifferentiation.

This pattern — reduced overall functional connectivity with preserved but weakened modular structure — is consistent with the network integration-to-segregation shift reported in the aging neuroscience literature (Damoiseaux, 2017; Ferreira & Busatto, 2013).

### 2.3 ANOVA Results: Which Networks Change Most with Age?

**57,358 of 71,631 edges (80.1%)** showed significant age-related differences after FDR correction (q < 0.05), indicating that aging exerts a **broad, near-global effect** on functional connectivity rather than being confined to a few specific circuits.

**F-statistic matrix (Figure: `anova_fstat_matrix.png`):**
The F-statistic map shows clusters of very high F-values (dark red/black in the `hot_r` colormap) distributed across the matrix in a non-random pattern. The highest F-values concentrate near specific ROI clusters, particularly in the top-left corner (early visual / auditory) and scattered diagonal blocks corresponding to higher-order association areas.

**Network ranking by mean F-statistic:**

| Rank | Network | Mean F | Interpretation |
|------|---------|--------|----------------|
| 1 | **Posterior Multimodal** | 33.71 | Highest age sensitivity — regions at the convergence of visual, auditory, and somatosensory processing (lateral temporal, posterior parietal) |
| 2 | **Language** | 31.05 | Bilateral perisylvian language network shows pronounced age-related reorganization |
| 3 | **Frontoparietal** | 29.23 | Lateral prefrontal–inferior parietal coupling, core of fluid intelligence, highly age-sensitive |
| 4 | **Cingulo-Opercular** | 28.04 | Task-control network; anterior insula and ACC show strong age effects |
| 5 | **Subcortical** | 26.52 | Basal ganglia, thalamus, hippocampus — expected due to structural volume loss with age |
| 6 | **Dorsal Attention** | 24.99 | Intraparietal sulcus and FEF; spatial attention circuits decline significantly |
| 7 | **Auditory** | 19.65 | Superior temporal / Heschl's gyrus connectivity declines with age |
| 8 | **Orbito-Affective** | 19.63 | Orbitofrontal and medial prefrontal emotion regulation circuits |
| 9 | **Default** | 16.36 | Lowest among named networks — relative preservation of DMN in aging |

**Key finding:** The **Posterior Multimodal, Language, and Frontoparietal networks** show the greatest age-related connectivity differences. Notably, the **Default Mode Network (DMN)** shows the *lowest* mean F-statistic among the named networks, consistent with the well-documented relative preservation of DMN connectivity into late life compared to other networks.

**Network Contrast Bar Plot (Figure: `network_contrast_barplot.png`):**
The bar plot shows an **orderly monotonic decrease in connectivity** from the youngest (36–50, blue) to the oldest (81+, red) group across *every* network without exception. The effect is particularly dramatic for:
- **Frontoparietal:** ~360 (36–50) → ~110 (81+): a **~70% reduction** in mean covariance
- **Language and Subcortical:** Similar steep gradients (~65–70% reduction)
- **Default:** Substantially lower absolute covariance at all ages compared to Frontoparietal, but the relative drop is proportionally similar

This universal reduction in covariance magnitude likely reflects a combination of true neurobiological changes (reduced synaptic density, white matter integrity loss, neurovascular coupling changes) and a methodological note: the covariance metric (not normalized correlation) is sensitive to signal variance, which may also decrease with age due to reduced neural signal power.

### 2.4 Interpretation: Mechanistic Framework

The results align with the **network dedifferentiation hypothesis of aging** (Park & Reuter-Lorenz, 2009):

1. **Reduced within-network cohesion:** The loss of strong positive covariance within network clusters reflects degraded structural connectivity (white matter tract deterioration, as confirmed by age-related DTI changes) and reduced coordination within specialized processing systems.

2. **Reduced between-network segregation:** In younger subjects, between-network negative connectivity (anticorrelation) is robust; in older subjects this segregation weakens, consistent with the "noisy brain" hypothesis where age-related signal-to-noise reduction blurs network boundaries.

3. **Association cortex > primary cortex:** While the bar plot shows decline across all networks, multimodal association areas (Frontoparietal, Posterior Multimodal) show larger absolute declines than primary sensory areas — consistent with the "last in, first out" principle of cortical aging (Raz & Rodrigue, 2006).

4. **DMN relative preservation:** The relatively lower F-statistic for DMN is consistent with multiple studies showing that DMN connectivity declines more gradually than task-positive networks, possibly because it is supported by robust structural hubs (posterior cingulate, mPFC) with extensive long-range white matter connections that have some resilience to early aging.

---

## Part 3: Connectome-Based Predictive Modeling (CPM)

### 3.1 Setup

- **Target variable:** `FluidIQ_Tr35_60y` — NIH Toolbox Fluid Intelligence composite score (age-normed T-score, mean=100, SD=15 in normative sample; appears z-scored or regressed here given the observed range of ~±5)
- **N subjects:** 1,290 (V1, with valid FluidIQ and connectivity data)
- **Feature space:** 71,631 upper-triangle edges from the 379×379 Glasser connectivity matrix
- **Protocol:** LOOCV (1,290 folds), p < 0.01 Pearson r threshold for edge selection per fold

### 3.2 CPM Performance Results

| Network | Pearson r | p-value | Spearman ρ | MAE |
|---------|-----------|---------|------------|-----|
| **Positive Network** | 0.240 | < 0.0001 | 0.343 | 0.83 |
| **Negative Network** | 0.333 | < 0.0001 | 0.360 | 0.80 |
| **Combined (Pos−Neg)** | 0.283 | < 0.0001 | 0.368 | 0.81 |

All three models achieve **highly significant prediction** (p < 0.0001 across 1,290 subjects), confirming that resting-state functional connectivity contains reliable information about individual fluid intelligence.

### 3.3 Scatter Plot Interpretation (Figure: `scatter_predicted_vs_observed.png`)

All three scatter plots show a clear **positive linear trend** (dashed regression line) with substantial scatter around it — typical for neuroimaging-based behavioral prediction. Several observations:

**Positive Network (r = 0.240):**
- Edges positively correlated with FluidIQ: when a subject has stronger connectivity in these edges, the model predicts higher IQ.
- The weakest of the three models; the relationship is real but modest.
- The regression line shows a flatter slope than ideal, indicating the model underestimates variance at the extremes.

**Negative Network (r = 0.333) — Best single-network predictor:**
- Edges where *lower* connectivity predicts *higher* FluidIQ. This is neurobiologically meaningful — it likely captures disengagement of task-irrelevant circuits (e.g., limbic-DMN connectivity) during high fluid intelligence, reflecting efficient neural resource allocation.
- The stronger prediction from the negative network is consistent with prior CPM literature (Finn et al., 2015; Rosenberg et al., 2016), where the negative network often captures noise-suppression mechanisms that distinguish high from low performers.

**Combined Network (r = 0.283):**
- Intermediate performance — the combination of pos − neg strength does not always outperform the best individual network, which can occur when the positive and negative networks capture partially redundant variance or when noise in one network degrades the combined signal.

**Overall scatter characteristics:**
- Observed FluidIQ ranges approximately −5 to +2 (z-score scale); predictions are compressed (~−1 to +0.5), the classic "regression to the mean" shrinkage in LOO cross-validation.
- The data cloud is wider than the regression line, reflecting that brain connectivity accounts for only a portion of IQ variance (~6–11% based on r²: r=0.24 → r²=5.8%; r=0.33 → r²=11.1%).

### 3.4 Effect Size and Contextual Benchmarking

| Model | r² (variance explained) | Benchmark |
|-------|--------------------------|-----------|
| Positive Network | 5.8% | Consistent with Finn et al. 2015 (r~0.26) |
| Negative Network | 11.1% | Above average for FC→IQ CPM (literature: r~0.2–0.4) |
| Combined | 8.0% | Typical for combined CPM models |

These effect sizes are **modest but scientifically meaningful** and align with published CPM studies predicting fluid intelligence from resting-state connectivity. The AABC sample (36–90+ years) is more heterogeneous in age than typical CPM studies, which likely introduces additional variability that reduces predictive accuracy compared to single-age cohorts.

### 3.5 Edge Selection Patterns (Figures: `edge_selection_matrix_pos.png`, `edge_selection_matrix_neg.png`)

The edge selection frequency maps show which edges were *consistently* selected across LOOCV folds (proportion of 1,290 folds in which each edge passed the p < 0.01 threshold):

**Positive network edges (positively correlated with FluidIQ):**
- High-frequency selection in specific ROI clusters corresponding to prefrontal–parietal connections — consistent with the role of the frontoparietal control network in fluid intelligence (Conway et al., 2003).
- Edges between frontal and cingulate regions are frequently selected, reflecting executive control circuitry relevant to problem-solving and reasoning.

**Negative network edges (negatively correlated with FluidIQ):**
- Higher overall selection frequency (denser map) than the positive network, explaining the negative network's superior predictive performance.
- Concentrated in connections between limbic/default and control/attention regions — higher connectivity in these "cross-network" edges is associated with *lower* FluidIQ, consistent with interference or noise in circuits that should be segregated during cognitive demands.

### 3.6 Neurobiological Interpretation

The CPM findings support a **functional segregation model of intelligence**:
- Higher fluid intelligence is associated with **stronger positive connectivity within frontoparietal and executive control circuits** (captured by the positive network).
- Simultaneously, higher FluidIQ individuals show **weaker connectivity between default mode and task-positive circuits** — suggesting better task-relevant network segregation (negative network).

This "efficient segregation" framework is supported by graph-theoretic studies showing that higher-IQ individuals have more modular brain network organization during rest (Hilger et al., 2017; Sporns, 2013).

---

## Summary & Integrated Conclusions

### Cross-Part Synthesis

| Finding | Parts 1 & 2 | Part 3 |
|---------|-------------|--------|
| **Network modularity** | Clear blocks in heatmap; reduced with age | Frontoparietal edges drive positive IQ prediction |
| **Default Mode** | Relatively preserved in aging (lowest ANOVA F) | Negative network includes DMN–control cross-connections |
| **Frontoparietal** | Largest absolute age-related connectivity decline | Core positive network for fluid intelligence prediction |
| **Aging & cognition** | ~70% covariance reduction in frontoparietal (36→81+) | Age heterogeneity in AABC likely limits CPM r to ~0.33 |

### Key Conclusions

1. **Resting-state functional connectivity is organized into reproducible, identifiable networks** visible as block structure in the Schaefer 400-parcel matrix. The dominant patterns — within-network positive coupling and cross-network anticorrelation between Default and task-positive systems — are consistent with the canonical resting-state network literature.

2. **Aging produces a broad, near-global reduction in functional connectivity** (80.1% of edges significantly differ across age groups, FDR q < 0.05). This is not network-specific: every identified network shows monotonic decline from age 36 to 81+. However, the magnitude of decline is largest for multimodal association networks (Posterior Multimodal, Language, Frontoparietal) and smallest for the Default Mode Network.

3. **Brain connectivity significantly predicts fluid intelligence** even in a large, age-diverse sample (r = 0.24–0.33, all p < 0.0001, LOOCV). The negative network (edges anti-correlated with IQ) outperforms the positive network, suggesting that the capacity to *suppress* task-irrelevant connectivity is as important for fluid intelligence as the strength of executive circuits.

4. **Methodological note:** The covariance-based connectivity metric (rather than normalized correlation) from the AABC pipeline is sensitive to absolute signal variance, which may amplify apparent age effects in Part 2. Future analyses should consider normalizing by signal standard deviation to disentangle connectivity changes from signal power changes.

---

## Technical Notes

- **Grayordinate mapping:** 58,606/91,282 (64.2%) mapped; unmapped grayordinates are subcortical regions not covered by the cortical-only Schaefer atlas — no data loss for the cortical analysis.
- **Warning `pixdim[1,2,3] should be non-zero`:** Benign nibabel warning from the dtseries header; no impact on data integrity.
- **Fisher z diagonal:** Diagonal set to 0 before arctanh to prevent Inf values (r=1 on diagonal); reflected in the z-matrix range (−0.484 to +1.376, no Inf values).
- **AABC covariance data:** Values are full (unnormalized) covariance — substantially larger absolute values than Pearson r, hence the ±600+ scale in Part 2 heatmaps.

---

*Report generated from outputs of `part1_parcellation.py`, `part2_lifespan.py`, and `part3_cpm.py`.*  
*All figures in `results/part1/figures/`, `results/part2/figures/`, `results/part3/figures/`.*
