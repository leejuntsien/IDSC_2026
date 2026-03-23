# Brugada Syndrome ECG Detection Pipeline - Walkthrough

This document demonstrates how to use each part of the Brugada detection pipeline and the interactive Streamlit application.

## 1. Environment Setup

First, ensure your dependencies are installed. We use a virtual environment or conda.

```bash
# We assume you are at the repository root
pip install -r app/requirements.txt
```

> **Note**: The pipeline requires the `brugada-huca/` dataset from PhysioNet. It should be structured linearly where the patient ID maps to their recording, e.g., `brugada-huca/files/001/001.dat` and `001.hea`.

---

## 2. Training the Models

Before running the app, we need to generate the binary model files (`.pkl` and `.pt`) stored in the `models/` directory.

### Step 2.1: Classic Machine Learning & ERSI Pipeline
This script runs a complete patient-stratified 5-fold cross-validation on `LightGBM`, `XGBoost`, `Random Forest`, `SVM`, and `KNN`. It identifies the best lead combination (V1+V2) and saves the optimal model. Finally, it fits the Statistical Entropy ERSI model to the dataset.

```bash
python run_brugada_classic_ml.py
```
**Output**:
- [models/best_classic_model.pkl](file:///c:/Users/User/OneDrive/Documents/idsc_2026/models/best_classic_model.pkl) (Tabular ML Model)
- [models/ersi_detector.pkl](file:///c:/Users/User/OneDrive/Documents/idsc_2026/models/ersi_detector.pkl) (ERSI Model configured with empirical thresholds)
- [cv_results_summary.csv](file:///c:/Users/User/OneDrive/Documents/idsc_2026/cv_results_summary.csv) and [ersi_evaluation.csv](file:///c:/Users/User/OneDrive/Documents/idsc_2026/ersi_evaluation.csv)
- Explanatory plots in `figures/`.

### Step 2.2: Temporal Deep Learning Pipeline
This script trains a 1D CNN + BiGRU encoder over 8-beat sequences to capture rate-dependent ST morphology changes. It iterates through experiments on V1-V3, right precordial, and Vectorcardiogram (VCG) inputs.

```bash
python run_brugada_temporal_dl.py
```
**Output**:
- [models/best_temporal_model.pt](file:///c:/Users/User/OneDrive/Documents/idsc_2026/models/best_temporal_model.pt) (PyTorch Weights)
- [dl_lead_experiment_results.csv](file:///c:/Users/User/OneDrive/Documents/idsc_2026/dl_lead_experiment_results.csv)

---

## 3. Running the Streamlit App

With the models trained and saved, you can launch the interactive web interface to predict on new or existing patient records.

```bash
# Launch via bash script (handles directory traversal automatically)
bash app/run_app.sh

# Or directly via streamlit:
python -m streamlit run app/streamlit_app.py --server.maxUploadSize 50
```

Once launched, navigate to `http://localhost:8501` in your browser.

---

## 4. App Interface Modes

The Streamlit UI has a sidebar that allows you to toggle between three diagnostic analysis modes:

### Mode A: Discrete Mode (Classic ML)
1. **Upload**: Select a `.hea` and `.dat` file (e.g. `001.hea`, `001.dat`).
2. **Analysis**: The app segments the ECG down to individual beats, dropping noisy signals. It builds normalized features focusing on J+40/J+80 ST segments, filtering out normal sequences via the "Layer 1 ST Filter" rule.
3. **Classification**: The best LightGBM (or RF) checks representative beats. A global classification determines if the patient exhibits Brugada signatures.
4. **Visuals**: A SHAP beeswarm plot evaluates which morphological traits pushed the algorithm to say Yes/No.

### Mode B: Temporal Mode (Deep Learning)
1. **Selection**: Like before, provide the 12-lead files.
2. **Analysis**: Bypasses beat selection. Evaluates continuous 8-beat sliding windows through a pre-trained PyTorch 1D CNN+BiGRU model.
3. **Visuals**: The model highlights sequences (highlighted in red) matching positive windows alongside the prediction confidence trajectory plot.

### Mode C: ERSI Mode (Statistical Entropy)
1. **Analysis**: Analyzes the continuous, raw 12-second trace on V1 focusing purely on thermodynamic stability metrics (Shannon, Sample, Tsallis entropies) rather than waveform shape.
2. **Classification**: Ranks stability across sliding windows. A patient falling below a normal statistical threshold is flagged.
3. **Visuals**: Plots dynamic window variance across time highlighting entropy breakdown.
