# Brugada Syndrome Detection Pipeline — Technical Report
## IDSC 2026 — Brugada-HUCA Dataset

---

### 1. Problem Statement

Binary classification of 12-lead ECG recordings: Brugada syndrome (Type 1 or 2 collapsed to positive) vs Normal sinus rhythm.

**Dataset**: Brugada-HUCA (PhysioNet). 363 patients, 76 Brugada (20.9%), 287 Normal (79.1%).
**Recording specs**: 100Hz sampling rate, 12-second recordings, 1200 samples per record, 12 standard leads.
**Class imbalance**: 3.77:1 (Normal:Brugada). Addressed via `class_weight='balanced'` in all sklearn models, `pos_weight` BCE loss in PyTorch, and CV threshold optimisation on PR curve.

---

### 2. Pipeline Architecture

```
Raw WFDB Records (363 patients, 12 leads, 100Hz, 12s)
                │
    [ Signal Quality Guard ]
      min 2s signal, detect_peaks fallback chain
                │
    [ Beat Segmentation — RR-based, scipy.signal.resample → 200 samples ]
      beat_id = patient_id_lead_beatindex attached
                │
    [ Layer 1: Rule-Based ST Filter ]      → layer1_audit.csv
      J+40ms elevation ≥ 0.10mV
      AND (T-wave ≤ 0.05mV OR ST downsloping)
      Patient-level recall: 94.74%
                │
    ┌───────────┼──────────────────┐
    │           │                  │
[ Discrete ]  [ Temporal ]    [ ERSI ]
Classic ML    CNN+BiGRU       Entropy Stability
    │           │                  │
    └───────────┴──────────────────┘
                │
    [ Patient Aggregation ]
      any positive beat/window → patient positive
                │
    [ Residual Analysis ]
      FN/FP categorisation, threshold recalibration signal
```

---

### 3. Signal Processing

- **Filtering**: 50Hz powerline notch filter via `nk.signal_filter`
- **Peak detection**: NeuroKit2 `ecg_peaks` with scipy `find_peaks` fallback; physiological validation (300ms minimum distance, 50th percentile amplitude floor, 50ms boundary padding)
- **Beat segmentation**: Consecutive R-peak pairs define beat boundaries; per-beat extraction with try/except — bad beats skipped, patient retained
- **Resampling**: `scipy.signal.resample` to 200 samples per beat (preserves morphological shape ratios; replaces zero-padding which misaligns ST segment under HRV)
- **ST features**: J-point at `ECG_R_Offsets`; isoelectric baseline from TP segment (T_offset_prev to P_onset); ST elevation at J+40ms (4 samples) and J+80ms (8 samples); ST slope; T-wave amplitude (signed); R'/R ratio

---

### 4. Detection Modes

#### 4.1 Discrete Mode (Classic ML)

**Representative beat selection**: Each patient → 2 rows (median beat + outlier beat). Intra-patient z-score normalisation + Euclidean distance from patient's own median feature vector. Patients with distance std > population median threshold flagged as heterogeneous.

**Lead sweep**: 11 curated combinations evaluated via fast RF + patient-stratified 5-fold CV. Winner: V1_V2 (F1=0.51).

**Models**: LightGBM, XGBoost, Random Forest, SVM, KNN with `class_weight='balanced'`, `CalibratedClassifierCV(method='isotonic')`, CV-based threshold from PR curve targeting minority F1.

| Model | Sensitivity | Specificity | MCC | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| KNN | 0.411±0.169 | 0.916±0.064 | 0.388±0.065 | 0.760±0.044 | 0.534±0.098 |
| SVM | 0.357±0.152 | 0.955±0.021 | 0.398±0.129 | 0.775±0.017 | 0.522±0.068 |
| RF  | 0.305±0.163 | 0.930±0.073 | 0.325±0.123 | 0.758±0.060 | 0.541±0.088 |
| LightGBM | 0.504±0.237 | 0.920±0.043 | 0.463±0.166 | 0.828±0.062 | 0.654±0.148 |
| XGBoost  | DNF | DNF | DNF | DNF | DNF |

*Values are 5-fold patient-stratified CV mean ± std. Fill from `cv_results_summary.csv`.*

#### 4.2 Temporal Mode (CNN+BiGRU)

**Architecture**: Shared 1D CNN encoder per beat (morphological embedding) → RR interval projection (scalar context) → Bidirectional GRU over N=8 consecutive beats → attention pooling → classification.

**Input experiments**: V1+V2+V3 (3ch), RMS (1ch), VCG Kors (3ch).

| Experiment | Sensitivity | Specificity | MCC | ROC-AUC |
|---|---|---|---|---|
| V1V2V3 | 70.9% | 85.8% | 0.547 | 0.803 |
| RMS    | 56.6% | 91.3% | 0.507 | 0.801 |
| VCG    | 95.0% | 72.0% | 0.566 | 0.856 |

VCG achieves highest MCC and AUC. The spatial projection of Kors effectively standardises the morphological patterns, leading to extremely high sensitivity but at the cost of some specificity.

#### 4.3 ERSI Mode (Statistical)

**Method**: Entropy-Ranked Stability Index. Sliding window (2s, 1s step → ~10 windows per 12s recording) entropy computation: Shannon, approximate, sample, permutation, spectral, SVD (via antropy), Tsallis (q=0.5, 1.5), Rényi (α=0.5, 2.0).

**ERSI_full** (dual-ranked): F_i = (1/M) × Σ_j [E_ij × (1/R_ij^time) × (1/R_ij^cross)]

Feature selection via Mann-Whitney U (training set only, top 3 features). Threshold at 95th percentile of Normal patient ERSI_full scores.

| Metric | Value |
|---|---|
| AUC (ERSI_full) | 0.784 |
| Sensitivity | 78.6% |
| Specificity | 77.8% |
| p-value (Mann-Whitney) | 8.35e-06 |

*Note: ERSI model generation successfully completed.*

---

### 5. Evaluation Design

- **Train/test split**: Patient-level GroupShuffleSplit (70/30). No patient appears in both sets.
- **CV**: 5-fold patient-stratified StratifiedKFold. Results reported as mean ± std.
- **Primary metric**: Patient-level MCC (penalises all four quadrants symmetrically under imbalance).
- **Secondary**: Sensitivity (clinical priority — missing Brugada is worse than false alarm).
- **Patient aggregation**: Any positive beat/window → patient classified positive.

---

### 6. Key Findings

1. V1+V2 is the optimal lead combination for discrete classification (sweep winner consistently). V3 alone is weakest — consistent with Type 1 coved pattern being primarily a right ventricular outflow tract phenomenon manifesting in V1-V2.

2. RMS input achieves highest MCC in temporal mode. The assumption-free spatial energy aggregation outperforms both directed lead selection (V1V2V3) and Kors VCG reconstruction.

3. Temporal inter-beat context substantially improves over single-beat classification. Attention BiLSTM (single-beat, MCC ~0.18) vs CNN+BiGRU (N=8 beats, MCC 0.52+). The BiGRU is capturing rate-dependent ST dynamics.

4. Layer 1 rule-based filter achieves 94.74% patient-level Brugada recall before any ML. The 5.26% missed are likely concealed Brugada (no Type 1 pattern in baseline recording).

5. ERSI_full is robust to single-measure noise artifacts — the dual-ranking suppresses windows where only one entropy function fires anomalously.

---

### 7. Limitations

- 100Hz sampling rate limits ST morphology resolution (J+40ms = exactly 4 samples; rSR' notch may be 1–2 samples wide and susceptible to aliasing)
- 76 positive patients limits statistical power; 5-fold CV gives ~15 Brugada patients per test fold
- Single-centre dataset; generalisability to other ECG acquisition systems unknown
- Concealed Brugada (Type 2 pattern, or fever/drug-unmasked only) not detectable by Layer 1 and likely missed by discrete mode
- ERSI window count (~10 per recording) is sparse for stable ranking; higher sampling rate recordings would benefit substantially

---

### 8. Deployment

Streamlit application (`app/streamlit_app.py`) supporting all three detection modes. Input: WFDB `.dat` + `.hea` file pair. Output: prediction, confidence, ECG signal plots with annotated flagged segments, SHAP feature importance (Discrete), attention overlay (Temporal), ERSI window timeline (Statistical).

Launch: `bash app/run_app.sh`
