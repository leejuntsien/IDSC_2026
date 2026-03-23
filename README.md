# IDSC_2026 — Brugada Syndrome ECG Detection Pipeline

## Overview

End-to-end pipeline for Brugada syndrome detection from 12-lead ECG recordings using three complementary detection modes.

**Dataset**: Brugada-HUCA (PhysioNet). 363 patients, 76 Brugada (20.9%), 287 Normal. 100Hz, 12s, 12-lead.
> The `brugada-huca/` folder is gitignored. Download from PhysioNet and place records under `brugada-huca/files/{patient_id}/{patient_id}.dat`.

## Quick Start

```bash
# Install dependencies
pip install -r app/requirements.txt

# Step 1: Train classic ML models, fit ERSI, save all model files
python run_brugada_classic_ml.py

# Step 2: Train temporal CNN+BiGRU, save model
python run_brugada_temporal_dl.py

# Step 3: Launch Streamlit app
bash app/run_app.sh
```

## Repository Structure

```
IDSC_2026/
├── ml_pipeline/
│   ├── ecg_pipeline_features.py   # Signal processing, ST extraction, Layer 1 rule
│   ├── data_loader.py             # WFDB loader, feature extraction
│   ├── classic_pipeline.py        # Sklearn pipeline, CV, threshold calibration
│   ├── dl_pipeline.py             # ECGTemporalCNN, ECGBeatSequenceDataset
│   ├── beat_selector.py           # Intra-patient representative beat selection
│   ├── layer1_filter.py           # Rule-based ST filter
│   ├── patient_aggregator.py      # Beat→patient aggregation, residual analysis
│   ├── ersi_detector.py           # BrugadaERSIDetector adapter
│   ├── ersi.py                    # ERSI class (computation, timeseries, full)
│   ├── entropy_measures.py        # Shannon, Tsallis, Rényi, antropy integration
│   ├── ersi_pipeline.py           # process_patient, benchmark_ersi
│   └── ersi_val_pipeline.py       # ERSIPipelineValidator, ERSIDataPrep
├── app/
│   ├── streamlit_app.py           # Three-mode Streamlit detection interface
│   ├── requirements.txt
│   └── run_app.sh
├── models/                        # Saved model files (gitignored)
│   ├── best_classic_model.pkl
│   ├── best_temporal_model.pt
│   └── ersi_detector.pkl
├── figures/                       # Generated plots
├── run_brugada_classic_ml.py      # Classic ML runner
├── run_brugada_temporal_dl.py     # Temporal DL runner
├── run_brugada_explainability.py  # Attention BiLSTM explainability
├── TECHNICAL_REPORT.md
└── README.md
```

## Detection Modes

| Mode | Approach | Input | Primary Metric |
|---|---|---|---|
| **Discrete** | Classic ML on tabular ST features | Beat-level representative features (V1+V2) | MCC, Sensitivity |
| **Temporal** | CNN+BiGRU sliding window (N=8 beats) | Raw beat sequences V1V2V3 or RMS | MCC, ROC-AUC |
| **ERSI** | Entropy-Ranked Stability Index (ERSI_full) | Raw V1 continuous signal | AUC, p-value |

## Key Results

| Model | Sensitivity | Specificity | MCC | ROC-AUC |
|---|---|---|---|---|
| KNN (Discrete) | 0.411 | 0.916 | 0.388 | 0.760 |
| LightGBM (Discrete) | 0.504 | 0.920 | 0.463 | 0.828 |
| VCG CNN+BiGRU (Temporal) | 0.950 | 0.720 | 0.566 | 0.856 |
| ERSI_full (Statistical) | 0.786 | 0.778 | N/A | 0.784 |

*Fill from `cv_results_summary.csv`, `dl_lead_experiment_results.csv`, `ersi_evaluation.csv`.*

## Architecture

See `TECHNICAL_REPORT.md` for full pipeline description, mathematical formulations, and evaluation methodology.

## Citation

Brugada-HUCA dataset: PhysioNet. [Add citation when submitting.]