"""
run_brugada_temporal_dl.py — Temporal DL runner for Brugada detection.
Step 8: CNN+BiGRU hybrid with lead combination experiments.
Separate from run_brugada_explainability.py (which uses traditional DL models).
"""
import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, matthews_corrcoef,
    roc_auc_score, average_precision_score, precision_recall_curve,
    brier_score_loss,
)

from ml_pipeline.data_loader import load_wfdb_record, extract_sequence_features
from ml_pipeline.dl_pipeline import (
    ECGTemporalCNN, ECGBeatSequenceDataset,
    train_epoch, evaluate,
)

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('pipeline_run.log', mode='a'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def build_patient_sequences(metadata, leads_mode='right_precordial',
                             target_len=200, n_beats_window=8):
    """
    Builds patient-level sequence dicts for ECGBeatSequenceDataset.

    Args:
        metadata: DataFrame with patient_id and brugada columns.
        leads_mode: 'right_precordial', 'rms', 'all', or list of lead names.

    Returns:
        sequences dict, rr_arrays dict, labels dict, in_channels int
    """
    sequences = {}
    rr_arrays = {}
    labels = {}

    for _, row in metadata.iterrows():
        pid = str(row['patient_id'])
        label = int(row['brugada'])
        record_path = f"brugada-huca/files/{pid}/{pid}"
        if not os.path.exists(record_path + ".dat"):
            continue

        try:
            df, fs = load_wfdb_record(record_path)

            if leads_mode == 'rms':
                seqs, rrs = extract_sequence_features(
                    df, fs, use_rms=True, method='interpolate',
                    target_len=target_len, return_rr=True,
                )
            elif leads_mode == 'all':
                seqs, rrs = extract_sequence_features(
                    df, fs, use_all_leads=True, method='interpolate',
                    target_len=target_len, return_rr=True,
                )
            elif leads_mode == 'vcg':
                seqs, rrs = extract_sequence_features(
                    df, fs, use_vcg=True, method='interpolate',
                    target_len=target_len, return_rr=True,
                )
            elif isinstance(leads_mode, list):
                # Multi-lead from specific leads
                selected = {c: df[c] for c in leads_mode if c in df.columns}
                if len(selected) == 0:
                    continue
                selected_df = pd.DataFrame(selected)
                seqs, rrs = extract_sequence_features(
                    selected_df, fs, use_all_leads=True, method='interpolate',
                    target_len=target_len, return_rr=True,
                )
            else:
                # Named presets
                lead_map = {
                    'right_precordial': ['V1', 'V2', 'V3'],
                }
                leads = lead_map.get(leads_mode, ['V1', 'V2', 'V3'])
                selected = {c: df[c] for c in leads if c in df.columns}
                if len(selected) == 0:
                    continue
                selected_df = pd.DataFrame(selected)
                seqs, rrs = extract_sequence_features(
                    selected_df, fs, use_all_leads=True, method='interpolate',
                    target_len=target_len, return_rr=True,
                )

            seqs = np.array(seqs, dtype=np.float32)
            if seqs.ndim == 2:
                seqs = seqs[:, :, np.newaxis]
            if seqs.ndim != 3 or seqs.shape[1] < 50:
                log.warning(f"Skipping {pid}: bad sequence shape {seqs.shape}")
                continue
            sequences[pid] = seqs
            rr_arrays[pid] = np.array(rrs, dtype=np.float32)
            labels[pid] = label
            # ─────────────────────────────────────────────────────────────

        except Exception as e:
            log.warning(f"Skipped {pid}: {e}")

    in_channels = list(sequences.values())[0].shape[-1] if sequences else 1
    log.info(f"Built sequences for {len(sequences)} patients, "
             f"in_channels={in_channels}")
    return sequences, rr_arrays, labels, in_channels


def train_temporal_model(sequences, rr_arrays, labels, in_channels,
                          n_beats=8, epochs=20, lr=1e-3, batch_size=16,
                          experiment_name='default'):
    """Trains ECGTemporalCNN with patient-level split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pids = np.array(list(labels.keys()))
    y_patients = np.array([labels[p] for p in pids])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(gss.split(pids, y_patients, groups=pids))
    train_pids = set(pids[train_idx])
    val_pids = set(pids[val_idx])

    train_seqs = {p: sequences[p] for p in train_pids if p in sequences}
    train_rrs = {p: rr_arrays[p] for p in train_pids if p in rr_arrays}
    train_labels = {p: labels[p] for p in train_pids}

    val_seqs = {p: sequences[p] for p in val_pids if p in sequences}
    val_rrs = {p: rr_arrays[p] for p in val_pids if p in rr_arrays}
    val_labels = {p: labels[p] for p in val_pids}

    train_ds = ECGBeatSequenceDataset(train_seqs, train_rrs, train_labels,
                                       n_beats=n_beats, stride=1)
    val_ds = ECGBeatSequenceDataset(val_seqs, val_rrs, val_labels,
                                     n_beats=n_beats, stride=1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Pos weight for class imbalance
    n_neg = sum(1 for v in train_labels.values() if v == 0)
    n_pos = sum(1 for v in train_labels.values() if v == 1)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    log.info(f"[DL] pos_weight={pos_weight.item():.2f} (neg={n_neg}, pos={n_pos})")

    model = ECGTemporalCNN(
        in_channels=in_channels, beat_len=200, n_beats=n_beats,
        cnn_embed_dim=64, gru_hidden=64, dropout=0.5,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5,
    )

    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                  device, is_temporal=True)
        # Val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, rr, y_b in val_loader:
                x, rr, y_b = x.to(device), rr.to(device), y_b.to(device)
                logits, _ = model(x, rr)
                val_loss += criterion(logits, y_b).item() * y_b.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        log.info(f"  Epoch {epoch+1:2d}/{epochs} | "
                 f"Train={train_loss:.4f} Val={val_loss:.4f}")

    # Evaluate
    preds, targets, probs = evaluate(model, val_loader, device, is_temporal=True)

    # Optimal threshold from PR curve
    precision_arr, recall_arr, thresholds = precision_recall_curve(targets, probs)
    f1_arr = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
    opt_thresh = float(thresholds[np.argmax(f1_arr[:-1])]) if len(thresholds) > 0 else 0.5

    preds_opt = (probs >= opt_thresh).astype(int)
    mcc = matthews_corrcoef(targets, preds_opt)
    roc_auc = roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.0
    pr_auc = average_precision_score(targets, probs) if len(np.unique(targets)) > 1 else 0.0

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, preds_opt)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
    else:
        sens = spec = 0.0

    log.info(f"\n[{experiment_name}] Optimal threshold: {opt_thresh:.4f}")
    log.info(f"  Sensitivity: {sens:.4f}  Specificity: {spec:.4f}")
    log.info(f"  MCC: {mcc:.4f}  ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}")

    return {
        'experiment_name': experiment_name,
        'in_channels': in_channels,
        'n_beats': n_beats,
        'sensitivity': sens,
        'specificity': spec,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'optimal_threshold': opt_thresh,
    }


def main():
    log.info("=" * 60)
    log.info("  Temporal DL Pipeline (CNN+BiGRU)")
    log.info("=" * 60)

    metadata_path = "brugada-huca/metadata.csv"
    if not os.path.exists(metadata_path):
        log.error(f"{metadata_path} not found.")
        return
    metadata = pd.read_csv(metadata_path)
    # Collapse Brugada Type 1 and Type 2 into single positive class
    # 0 = Normal, 1 = Brugada Type 1, 2 = Brugada Type 2  →  0 = Normal, 1 = Brugada
    metadata['brugada'] = (metadata['brugada'] > 0).astype(int)

    DL_LEAD_EXPERIMENTS = {
        'right_precordial_V1V2V3': 'right_precordial',
        'rms_1channel': 'rms',
        'vcg_3channel': 'vcg',
    }

    all_results = []
    for exp_name, leads_mode in DL_LEAD_EXPERIMENTS.items():
        log.info(f"\n{'='*50}")
        log.info(f"  Experiment: {exp_name}")
        log.info(f"{'='*50}")

        sequences, rr_arrays, labels, in_channels = build_patient_sequences(
            metadata, leads_mode=leads_mode
        )

        if len(sequences) < 10:
            log.warning(f"Too few patients for {exp_name}, skipping.")
            continue

        result = train_temporal_model(
            sequences, rr_arrays, labels, in_channels,
            n_beats=8, epochs=15, lr=1e-3,
            experiment_name=exp_name,
        )
        all_results.append(result)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('dl_lead_experiment_results.csv', index=False)
        log.info(f"\n{results_df.to_string(index=False)}")
        log.info("Saved dl_lead_experiment_results.csv")

    log.info("\nTemporal DL pipeline complete.")


if __name__ == "__main__":
    main()
