"""
run_brugada_explainability.py — DL Explainability runner (baseline DL models).
V2: uses full dataset, patient-level split, attention overlay + saliency maps.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit

from ml_pipeline.data_loader import load_wfdb_record, extract_sequence_features
from ml_pipeline.dl_pipeline import (
    ECGAttentionLSTM, ECGCNN1D, ECGTransformer,
    ECGSequenceDataset, plot_attention_overlay, plot_saliency_overlay,
    train_epoch, evaluate,
)

warnings.filterwarnings('ignore')


def main():
    print("=" * 60)
    print("  Deep Learning Explainability Pipeline (Full Dataset)")
    print("=" * 60)

    metadata_path = "brugada-huca/metadata.csv"
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return
    metadata = pd.read_csv(metadata_path)
    # Collapse Brugada Type 1 and Type 2 into single positive class
    # 0 = Normal, 1 = Brugada Type 1, 2 = Brugada Type 2  →  0 = Normal, 1 = Brugada
    metadata['brugada'] = (metadata['brugada'] > 0).astype(int)

    X_all = []
    y_all = []
    groups_all = []

    print("\n--- Extracting sequences (V1+V2+V3, interpolated) ---")
    for _, row in metadata.iterrows():
        pid = str(row['patient_id'])
        label = float(row['brugada'])
        record_path = f"brugada-huca/files/{pid}/{pid}"
        if not os.path.exists(record_path + ".dat"):
            continue
        try:
            df, fs = load_wfdb_record(record_path)
            # Use right precordial leads (V1, V2, V3)
            selected = {c: df[c] for c in ['V1', 'V2', 'V3'] if c in df.columns}
            if len(selected) == 0:
                continue
            selected_df = pd.DataFrame(selected)
            X_seq = extract_sequence_features(
                selected_df, fs, use_all_leads=True,
                method='interpolate', target_len=200,
            )
            X_all.append(X_seq)
            y_all.extend([label] * len(X_seq))
            groups_all.extend([pid] * len(X_seq))
        except Exception as e:
            print(f"Skipping {pid}: {e}")

    X_all = np.vstack(X_all)
    y_all = np.array(y_all)
    groups_all = np.array(groups_all)
    print(f"Total beats: {len(X_all)}, shape: {X_all.shape}")

    dataset = ECGSequenceDataset(X_all, y_all)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(gss.split(X_all, y_all, groups=groups_all))

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = X_all.shape[-1]

    # ── Attention LSTM ───────────────────────────────────────────────────────
    print("\n--- Training Attention BiLSTM ---")
    model = ECGAttentionLSTM(
        input_size=in_channels, hidden_size=64, num_layers=2, num_classes=1
    ).to(device)

    # Pos weight
    n_neg = (y_all[train_idx] == 0).sum()
    n_pos = (y_all[train_idx] == 1).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Epoch {epoch+1}/10 | Loss: {loss:.4f}")

    print("\n--- Evaluation ---")
    evaluate(model, val_loader, device)

    # ── Explainability ───────────────────────────────────────────────────────
    print("\n--- Attention Overlay on Brugada Beat ---")
    brugada_idx = np.where(y_all == 1)[0]
    if len(brugada_idx) > 0:
        sample_idx = brugada_idx[0]
        sample_beat = X_all[sample_idx]  # [200, channels]
        sample_tensor = (torch.tensor(sample_beat, dtype=torch.float32)
                         .unsqueeze(0).transpose(1, 2).to(device))
        model.eval()
        with torch.no_grad():
            logits, attn_weights = model(sample_tensor)
        prob = torch.sigmoid(logits).item()
        print(f"Beat #{sample_idx} → P(Brugada) = {prob:.4f}")

        os.makedirs('figures', exist_ok=True)
        signal_plot = sample_beat[:, 0]
        attn_plot = attn_weights.squeeze(0).cpu().numpy()
        plot_attention_overlay(
            signal_plot, attn_plot,
            title=f"Attention Map (P={prob:.4f})",
        )


if __name__ == "__main__":
    main()
