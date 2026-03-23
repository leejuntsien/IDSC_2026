"""
run_brugada_classic_ml.py — Classic ML runner for Brugada detection.
V2 overhaul: full dataset, feature caching, Layer 1 filter,
representative beat selection, lead combination sweep,
threshold calibration, patient-level evaluation.
"""
import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

# Pipeline imports
from ml_pipeline.data_loader import load_wfdb_record, extract_discrete_features, extract_sequence_features
from ml_pipeline.classic_pipeline import train_and_evaluate, patient_level_evaluate
from ml_pipeline.layer1_filter import run_layer1_on_patient, build_layer1_audit
from ml_pipeline.beat_selector import build_representative_dataset
from ml_pipeline.patient_aggregator import aggregate_to_patient_level, run_residual_analysis

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('pipeline_run.log', mode='w'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Metadata columns to exclude from ML features ────────────────────────────
METADATA_COLS = [
    'beat_index', 'beat_id', 'patient_id', 'period_s',
    'label', 'layer1_suspected', 'layer1_evidence',
    'st_extraction_quality', 'beat_type', 'intra_patient_distance',
    'is_homogeneous',
]


def main():
    log.info("=" * 60)
    log.info("  Brugada Classic ML Pipeline v2 (Full Dataset)")
    log.info("=" * 60)

    # ── 1. Load Metadata ─────────────────────────────────────────────────────
    metadata_path = "brugada-huca/metadata.csv"
    if not os.path.exists(metadata_path):
        log.error(f"{metadata_path} not found.")
        return
    metadata = pd.read_csv(metadata_path)
    # Collapse Brugada Type 1 and Type 2 into single positive class
    # 0 = Normal, 1 = Brugada Type 1, 2 = Brugada Type 2  →  0 = Normal, 1 = Brugada
    metadata['brugada'] = (metadata['brugada'] > 0).astype(int)
    log.info(f"Metadata loaded: {len(metadata)} patients "
             f"({metadata['brugada'].sum()} Brugada, "
             f"{(~metadata['brugada'].astype(bool)).sum()} Normal)")

    # ── 2. Extract Features (with caching) ───────────────────────────────────
    CACHE_DISCRETE = 'extracted_features_all_leads.csv'
    CACHE_LABELS = 'patient_labels.csv'

    if os.path.exists(CACHE_DISCRETE) and os.path.exists(CACHE_LABELS):
        log.info(f"[Cache] Loading cached features from {CACHE_DISCRETE}")
        df_all_beats = pd.read_csv(CACHE_DISCRETE)
        patient_labels_df = pd.read_csv(CACHE_LABELS)
    else:
        log.info("[Cache] Extracting features for all records (this takes a while)...")
        target_leads = ['I', 'II', 'III', 'aVR', 'AVR', 'aVL', 'AVL',
                        'aVF', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        all_features_dfs = []
        patient_labels = []

        success_count = 0
        skip_reasons = {}

        for idx, row in metadata.iterrows():
            patient_id = str(row['patient_id'])
            label = int(row['brugada'])
            record_path = f"brugada-huca/files/{patient_id}/{patient_id}"
    
            if not os.path.exists(record_path + ".dat"):
                skip_reasons['missing_file'] = skip_reasons.get('missing_file', 0) + 1
                continue
    
            try:
                df, fs = load_wfdb_record(record_path)
                df_patient = extract_discrete_features(
                    df, fs, patient_id=patient_id, target_leads=target_leads
                )
                df_patient['label'] = label
                all_features_dfs.append(df_patient)
                patient_labels.append({'patient_id': patient_id, 'label': label})
                success_count += 1
            except Exception as e:
                reason = type(e).__name__
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                log.warning(f"Skipped {patient_id}: {reason}: {e}")

        log.info(f"Extraction complete: {success_count} success, skipped: {skip_reasons}")

        df_all_beats = pd.concat(all_features_dfs, ignore_index=True)
        patient_labels_df = pd.DataFrame(patient_labels)

        df_all_beats.to_csv(CACHE_DISCRETE, index=False)
        patient_labels_df.to_csv(CACHE_LABELS, index=False)
        log.info(f"[Cache] Saved {len(df_all_beats)} beats to {CACHE_DISCRETE}")

    log.info(f"Total beats: {len(df_all_beats)} | "
             f"Patients: {df_all_beats['patient_id'].nunique()}")

    # ── 3. Layer 1: Rule-Based ST Filter ─────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  Layer 1: Rule-Based ST Filter")
    log.info("=" * 60)

    # Apply Layer 1 per patient
    l1_results = []
    for pid, group in df_all_beats.groupby('patient_id'):
        l1_df = run_layer1_on_patient(group, pid, leads=('V1', 'V2', 'V3'))
        l1_results.append(l1_df)
    df_all_beats = pd.concat(l1_results, ignore_index=True)

    audit_df = build_layer1_audit(df_all_beats, output_path='layer1_audit.csv')

    # Check recall threshold
    brugada_patients = audit_df[audit_df['true_label'] == 1]
    l1_recall = brugada_patients['n_suspected_beats'].gt(0).mean()
    log.info(f"[Layer 1] Patient-level Brugada recall: {l1_recall:.2%}")

    # ── 4. Representative Beat Selection ─────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  Representative Beat Selection")
    log.info("=" * 60)

    df_repr = build_representative_dataset(df_all_beats)

    # ── 5. Lead Combination Sweep ────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  Lead Combination Sweep (Fast RF)")
    log.info("=" * 60)

    ALL_12 = ['I', 'II', 'III', 'aVR', 'AVR', 'aVL', 'AVL',
              'aVF', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    LEAD_COMBINATIONS = {
        'V1':               ['V1'],
        'V2':               ['V2'],
        'V3':               ['V3'],
        'V1_V2':            ['V1', 'V2'],
        'V1_V3':            ['V1', 'V3'],
        'V2_V3':            ['V2', 'V3'],
        'right_precordial': ['V1', 'V2', 'V3'],
        'precordial':       ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        'limb':             ['I', 'II', 'III', 'aVR', 'AVR', 'aVL', 'AVL',
                             'aVF', 'AVF'],
        'V1_V2_aVR':        ['V1', 'V2', 'aVR', 'AVR'],
        'all_12':           ALL_12,
    }

    # Feature columns for combo evaluation
    all_cols_available = [c for c in df_repr.columns if c not in METADATA_COLS
                         and not c.endswith('_st_extraction_quality')
                         and not c.endswith('_layer1_evidence')]

    patient_ids = df_repr['patient_id'].values
    unique_patients = np.unique(patient_ids)
    patient_y = np.array([
        df_repr[df_repr['patient_id'] == p]['label'].iloc[0]
        for p in unique_patients
    ])

    combo_results = {}
    for combo_name, leads in LEAD_COMBINATIONS.items():
        lead_cols = [c for c in all_cols_available
                     if any(c.startswith(f'{lead}_') for lead in leads)]
        if not lead_cols:
            log.info(f"  {combo_name:20s}  SKIPPED (no matching columns)")
            continue

        X_combo = df_repr[lead_cols].fillna(df_repr[lead_cols].median())
        y_combo = df_repr['label'].values.astype(int)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_f1s = []

        for train_p_idx, val_p_idx in skf.split(unique_patients, patient_y):
            train_pats = unique_patients[train_p_idx]
            val_pats = unique_patients[val_p_idx]
            train_mask = np.isin(patient_ids, train_pats)
            val_mask = np.isin(patient_ids, val_pats)

            rf = RandomForestClassifier(
                n_estimators=100, class_weight='balanced',
                random_state=42, n_jobs=-1,
            )
            imp = SimpleImputer(strategy='median')
            X_tr = imp.fit_transform(X_combo[train_mask])
            X_vl = imp.transform(X_combo[val_mask])

            rf.fit(X_tr, y_combo[train_mask])
            preds = rf.predict(X_vl)

            # Patient-level aggregation for fold
            val_pid = patient_ids[val_mask]
            fold_df = pd.DataFrame({
                'pid': val_pid, 'y': y_combo[val_mask], 'pred': preds
            })
            pat_agg = fold_df.groupby('pid').agg(y=('y', 'max'), pred=('pred', 'max'))
            fold_f1 = f1_score(
                pat_agg['y'].astype(int),
                pat_agg['pred'].astype(int),
                pos_label=1,
                zero_division=0
            )
            fold_f1s.append(fold_f1)

        mean_f1 = float(np.mean(fold_f1s))
        combo_results[combo_name] = mean_f1
        log.info(f"  {combo_name:20s}  Patient F1 = {mean_f1:.4f}")

    best_combo_name = max(combo_results, key=combo_results.get)
    best_leads = LEAD_COMBINATIONS[best_combo_name]
    log.info(f"\n[LeadSweep] WINNER: {best_combo_name} "
             f"(F1={combo_results[best_combo_name]:.4f})")

    ranking = sorted(combo_results.items(), key=lambda x: x[1], reverse=True)
    for name, score in ranking:
        log.info(f"  {name:20s}  {score:.4f}")

    # Save lead ranking figure
    os.makedirs('figures', exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [r[0] for r in ranking]
    scores = [r[1] for r in ranking]
    ax.barh(names[::-1], scores[::-1], color='steelblue')
    ax.set_xlabel('Patient-Level F1 Score')
    ax.set_title('Lead Combination Ranking')
    plt.tight_layout()
    fig.savefig('figures/lead_combo_ranking.png', dpi=150)
    log.info("Saved figures/lead_combo_ranking.png")

    # ── 6. Full Multi-Model Benchmark on Best Leads ──────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"  Multi-Model Benchmark on {best_combo_name}")
    log.info("=" * 60)

    best_lead_cols = [c for c in all_cols_available
                      if any(c.startswith(f'{lead}_') for lead in best_leads)]
    feature_columns = best_lead_cols

    from sklearn.model_selection import GroupShuffleSplit

    X = df_repr[feature_columns].values
    y = df_repr['label'].values
    groups = df_repr['patient_id'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    pids_train = groups[train_idx]
    pids_test = groups[test_idx]

    log.info(f"Train: {len(X_train)} beats | Test: {len(X_test)} beats")

    models_to_run = ['lightgbm', 'xgboost', 'random_forest', 'svm', 'knn']
    results_log = []

    for model_name in models_to_run:
        log.info(f"\n{'='*50}")
        log.info(f"Running {model_name.upper()}...")
        start = time.time()
        try:
            calibrated_model, opt_thresh, probs_test, preds_test = train_and_evaluate(
                X_train, y_train, X_test, y_test,
                feature_columns=feature_columns,
                model_name=model_name,
                n_iter=10, cv=3,
            )

            # Patient-level evaluation
            patient_agg = patient_level_evaluate(
                y_test, preds_test, probs_test, pids_test
            )

            elapsed = time.time() - start
            results_log.append({
                'model': model_name,
                'threshold': opt_thresh,
                'elapsed_s': elapsed,
            })

            # Layer 3+4 aggregation & residual analysis
            beat_pred_df = pd.DataFrame({
                'patient_id': pids_test,
                'y_true': y_test,
                'pred': preds_test,
                'prob': probs_test,
            })
            patient_class = aggregate_to_patient_level(beat_pred_df)
            patient_class.to_csv('patient_classification.csv', index=False)

            residual_df = run_residual_analysis(
                patient_class,
                output_path='residual_report.csv',
            )

        except Exception as e:
            log.error(f"Skipping {model_name}: {e}")

    # ── 7. Summary ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  Benchmark Summary")
    log.info("=" * 60)
    if results_log:
        results_df = pd.DataFrame(results_log)
        log.info(f"\n{results_df.to_string(index=False)}")

    log.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
