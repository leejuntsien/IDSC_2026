"""
beat_selector.py — Representative beat selection for Classic ML.
Each patient → 2 rows: median beat + outlier beat.
Patient-level labels preserved on both rows.
Intra-patient anomaly detection — no cross-patient comparison.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Columns to exclude from feature distance computation
METADATA_COLS = [
    'beat_index', 'beat_id', 'patient_id', 'period_s',
    'label', 'layer1_suspected', 'layer1_evidence',
    'st_extraction_quality',
]


def select_representative_beats(df_patient, variation_threshold=None):
    """
    For a single patient's beat DataFrame, selects:
      - median_beat  : row closest to the patient's own median feature vector
      - outlier_beat : row furthest from the patient's own median feature vector

    Returns two rows as a DataFrame with added columns:
      'beat_type' ∈ ['median', 'outlier']
      'intra_patient_distance'
      'is_homogeneous'
    """
    # Dynamically exclude metadata + any columns that end with known quality flags
    feature_cols = [
        c for c in df_patient.columns
        if c not in METADATA_COLS
        and not c.endswith('_st_extraction_quality')
        and not c.endswith('_layer1_evidence')
    ]
    X = df_patient[feature_cols].apply(pd.to_numeric, errors='coerce')
    X_filled = X.fillna(X.median())

    if len(X_filled) < 2:
        row = df_patient.iloc[[0]].copy()
        row['beat_type'] = 'median'
        row['intra_patient_distance'] = 0.0
        row['is_homogeneous'] = True
        return pd.concat([row, row.assign(beat_type='outlier')])

    # Z-score within patient to prevent scale dominance
    std = X_filled.std().replace(0, 1)
    X_norm = (X_filled - X_filled.mean()) / std

    patient_median = X_norm.median(axis=0).values.reshape(1, -1)
    distances = cdist(X_norm.values, patient_median, metric='euclidean').flatten()

    variation = float(np.std(distances))

    median_idx = int(np.argmin(distances))
    outlier_idx = int(np.argmax(distances))

    median_row = df_patient.iloc[[median_idx]].copy()
    outlier_row = df_patient.iloc[[outlier_idx]].copy()

    median_row['beat_type'] = 'median'
    median_row['intra_patient_distance'] = float(distances[median_idx])
    outlier_row['beat_type'] = 'outlier'
    outlier_row['intra_patient_distance'] = float(distances[outlier_idx])

    is_homo = (variation_threshold is not None) and (variation < variation_threshold)
    median_row['is_homogeneous'] = is_homo
    outlier_row['is_homogeneous'] = is_homo

    return pd.concat([median_row, outlier_row])


def build_representative_dataset(df_all_beats):
    """
    Applies representative beat selection across all patients.

    Args:
        df_all_beats: DataFrame with all beats, must have 'patient_id' and 'label'.

    Returns:
        df_repr: DataFrame with 2 rows per patient.
    """
    feature_cols = [
        c for c in df_all_beats.columns
        if c not in METADATA_COLS
        and not c.endswith('_st_extraction_quality')
        and not c.endswith('_layer1_evidence')
    ]
    X = df_all_beats[feature_cols].apply(pd.to_numeric, errors='coerce')
    # After X = df_all_beats[feature_cols].apply(pd.to_numeric, errors='coerce')
    n_all_nan_cols = X.isnull().all(axis=0).sum()
    n_valid_cols   = (X.notnull().any(axis=0)).sum()
    print(f"[BeatSelector] Feature columns: {len(feature_cols)} total, "
        f"{n_valid_cols} with valid data, {n_all_nan_cols} all-NaN (dropped)")

    # Drop all-NaN columns before distance computation
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.median())

    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns after numeric coercion. "
            "Cache likely contains non-numeric values — delete cache files and re-run."
        )

    # Compute per-patient variation to set population threshold
    patient_variations = []
    for pid, group in df_all_beats.groupby('patient_id'):
        if len(group) < 2:          # ← ADD THIS
            patient_variations.append(0.0)
            continue
        X_grp = X.loc[group.index]
        std = X_grp.std().replace(0, 1)
        X_norm = (X_grp - X_grp.mean()) / std
        median_vec = X_norm.median(axis=0).values.reshape(1, -1)
        dists = cdist(X_norm.values, median_vec, metric='euclidean').flatten()
        patient_variations.append(float(np.std(dists)))

    valid_variations = [v for v in patient_variations if not np.isnan(v)]
    variation_threshold = float(np.median(valid_variations)) if valid_variations else 1.0
    #variation_threshold = float(np.median(patient_variations))
    print(f"[BeatSelector] Population variation threshold: {variation_threshold:.4f}")
    print(f"[BeatSelector] Patients above threshold (heterogeneous): "
          f"{sum(v >= variation_threshold for v in patient_variations)} / "
          f"{len(patient_variations)}")

    all_repr = []
    for pid, group in df_all_beats.groupby('patient_id'):
        repr_beats = select_representative_beats(group, variation_threshold=variation_threshold)
        all_repr.append(repr_beats)

    df_repr = pd.concat(all_repr, ignore_index=True)
    print(f"[BeatSelector] Final dataset: {len(df_repr)} rows "
          f"({df_repr['patient_id'].nunique()} patients × 2 beats)")
    return df_repr
