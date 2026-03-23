"""
patient_aggregator.py — Layer 3 (Patient-Level Aggregation)
                       and Layer 4 (Residual Analysis).
"""
import numpy as np
import pandas as pd


def aggregate_to_patient_level(beat_predictions_df):
    """
    Aggregates beat-level predictions to patient level.
    Rule: any positive beat → patient positive.

    Args:
        beat_predictions_df: DataFrame with patient_id, y_true, pred, prob, beat_type.

    Returns:
        patient_classification DataFrame.
    """
    agg = beat_predictions_df.groupby('patient_id').agg(
        y_true=('y_true', 'max'),
        pred_positive=('pred', 'max'),
        max_prob=('prob', 'max'),
        n_beats_total=('pred', 'count'),
        n_beats_positive=('pred', 'sum'),
    ).reset_index()

    agg['evidence_score'] = agg['max_prob'] * (
        agg['n_beats_positive'] / agg['n_beats_total']
    )
    return agg


def run_residual_analysis(patient_agg_df, output_path='residual_report.csv'):
    """
    Categorises errors (FN, FP) and logs diagnostic information.
    """
    fn_patients = patient_agg_df[
        (patient_agg_df['y_true'] == 1) & (patient_agg_df['pred_positive'] == 0)
    ]
    fp_patients = patient_agg_df[
        (patient_agg_df['y_true'] == 0) & (patient_agg_df['pred_positive'] == 1)
    ]

    print(f"\n[Residual] False Negatives (missed Brugada): {len(fn_patients)}")
    print(f"[Residual] False Positives (normal mislabelled): {len(fp_patients)}")

    if 'layer1_suspected_any' in patient_agg_df.columns:
        fn_missed_by_l1 = fn_patients[fn_patients['layer1_suspected_any'] == False]
        print(f"[Residual] FNs never flagged by Layer 1: {len(fn_missed_by_l1)} "
              f"(concealed Brugada or low signal quality)")

    patient_agg_df = patient_agg_df.copy()
    patient_agg_df['error_type'] = 'correct'
    patient_agg_df.loc[fn_patients.index, 'error_type'] = 'false_negative'
    patient_agg_df.loc[fp_patients.index, 'error_type'] = 'false_positive'
    patient_agg_df.to_csv(output_path, index=False)
    print(f"[Residual] Report saved to {output_path}")
    return patient_agg_df
