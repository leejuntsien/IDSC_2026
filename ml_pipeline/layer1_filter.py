"""
Layer 1: Rule-based ST filter applied across all beats in a patient record.
Produces per-beat suspected flags and saves an audit CSV.
"""
import numpy as np
import pandas as pd
from ecg_pipeline_features import layer1_brugada_rule


def run_layer1_on_patient(df_features, patient_id, leads=('V1', 'V2', 'V3')):
    """
    Apply Layer 1 rule to each beat in a patient's feature DataFrame.

    Args:
        df_features : DataFrame from extract_discrete_features, one row per beat,
                      with lead-prefixed columns and beat_id, patient_id columns
        patient_id  : str, used to verify mapping
        leads       : leads to apply rule to

    Returns:
        df_features with added columns:
          layer1_suspected  (bool)
          layer1_evidence   (dict as str, for logging)
    """
    suspected_flags = []
    evidence_logs = []

    for _, row in df_features.iterrows():
        suspected, evidence = layer1_brugada_rule(row.to_dict(), leads_to_check=leads)
        suspected_flags.append(suspected)
        evidence_logs.append(str(evidence))

    df_features = df_features.copy()
    df_features['layer1_suspected'] = suspected_flags
    df_features['layer1_evidence'] = evidence_logs
    return df_features


def build_layer1_audit(all_patients_features_df, output_path='layer1_audit.csv'):
    """
    Runs Layer 1 across all patients and saves an audit CSV.
    Computes per-patient summary statistics.
    """
    records = []
    for patient_id, group in all_patients_features_df.groupby('patient_id'):
        n_beats = len(group)
        n_suspected = group['layer1_suspected'].sum()
        true_label = group['label'].iloc[0]
        records.append({
            'patient_id': patient_id,
            'true_label': true_label,
            'n_beats': n_beats,
            'n_suspected_beats': int(n_suspected),
            'pct_suspected': round(n_suspected / n_beats * 100, 1),
        })

    audit_df = pd.DataFrame(records)
    audit_df.to_csv(output_path, index=False)
    print(f"[Layer 1] Audit saved to {output_path}")

    brugada_recall = audit_df[audit_df['true_label'] == 1]['n_suspected_beats'].gt(0).mean()
    normal_fp_rate = audit_df[audit_df['true_label'] == 0]['n_suspected_beats'].gt(0).mean()

    print(f"[Layer 1] Patient-level recall on Brugada: {brugada_recall:.2%}")
    print(f"[Layer 1] Normal patients with >=1 suspected beat: {normal_fp_rate:.2%}")

    return audit_df
