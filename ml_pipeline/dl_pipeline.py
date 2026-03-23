"""
dl_pipeline.py — PyTorch deep learning models for ECG Brugada classification.
V2 overhaul: adds ECGTemporalCNN (CNN+BiGRU hybrid with attention),
ECGBeatSequenceDataset, updated train/evaluate for (x, rr, y) batches,
plus existing ECGCNN1D, ECGAttentionLSTM, ECGTransformer as baselines.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

class ECGSequenceDataset(Dataset):
    """
    Basic Dataset: X [num_samples, seq_len, num_channels], y [num_samples].
    Transposes to [channels, length] for Conv1d.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ECGBeatSequenceDataset(Dataset):
    """
    Dataset for ECGTemporalCNN — sliding windows of N_beats consecutive beats.

    Args:
        sequences  : dict {patient_id: np.array [n_beats, beat_len, n_channels]}
        rr_arrays  : dict {patient_id: np.array [n_beats]} — period_s per beat
        labels     : dict {patient_id: int} — patient-level label
        n_beats    : window size (default 8)
        stride     : sliding window stride (default 1)

    Each item: (x [N_beats, channels, beat_len], rr [N_beats], label scalar)
    """
    def __init__(self, sequences, rr_arrays, labels, n_beats=8, stride=1):
        self.samples = []

        for pid, seq in sequences.items():
            n_total = seq.shape[0]
            label = labels[pid]
            rr = rr_arrays[pid]

            if n_total < n_beats:
                pad_len = n_beats - n_total
                seq = np.concatenate(
                    [seq, np.zeros((pad_len, *seq.shape[1:]))], axis=0
                )
                rr = np.concatenate([rr, np.zeros(pad_len)], axis=0)
                n_total = n_beats

            for start in range(0, n_total - n_beats + 1, stride):
                end = start + n_beats
                x_win = seq[start:end]
                rr_win = rr[start:end]
                self.samples.append((x_win, rr_win, label, pid))

        print(f"[ECGBeatSequenceDataset] {len(self.samples)} windows from "
              f"{len(sequences)} patients (N_beats={n_beats}, stride={stride})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_win, rr_win, label, pid = self.samples[idx]
        x_arr = np.array(x_win, dtype=np.float32)

        if x_arr.ndim == 2:
            x_arr = x_arr[:, :, np.newaxis]

        # Catch remaining bad shapes before permute — these should have been
        # filtered upstream but guard here as a safety net
        if x_arr.ndim != 3:
            raise ValueError(f"Patient {pid}: expected 3D array, got shape {x_arr.shape}")

        N_beats, beat_len, channels = x_arr.shape
        if beat_len < 10:
            raise ValueError(
                f"Patient {pid}: beat_len={beat_len} is too short — "
                f"likely a segmentation failure. Should have been filtered in build_patient_sequences."
            )

        x_tensor  = torch.tensor(x_arr).permute(0, 2, 1)  # [N_beats, channels, beat_len]
        rr_tensor = torch.tensor(rr_win, dtype=torch.float32)
        y_tensor  = torch.tensor(label,  dtype=torch.float32)
        return x_tensor, rr_tensor, y_tensor
      


# ──────────────────────────────────────────────────────────────────────────────
# Baseline Models (from v1, kept as baselines)
# ──────────────────────────────────────────────────────────────────────────────

class ECGCNN1D(nn.Module):
    """1D CNN for localized morphological changes in ECG."""
    def __init__(self, in_channels=1, num_classes=1):
        super(ECGCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        logits = self.fc(x).squeeze(1)
        return logits, None


class ECGAttentionLSTM(nn.Module):
    """Bidirectional LSTM with Attention Mechanism."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=1):
        super(ECGAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention_fc(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        logits = self.fc(context_vector).squeeze(-1)
        return logits, attn_weights.squeeze(-1)


class ECGTransformer(nn.Module):
    """Transformer-based ECG classifier."""
    def __init__(self, in_channels=1, d_model=64, nhead=4, num_layers=2, num_classes=1):
        super(ECGTransformer, self).__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        trans_out = self.transformer(x)
        pooled = trans_out.mean(dim=1)
        logits = self.fc(pooled).squeeze(1)
        return logits, None


# ──────────────────────────────────────────────────────────────────────────────
# NEW — Step 6: CNN+BiGRU Hybrid (ECGTemporalCNN)
# ──────────────────────────────────────────────────────────────────────────────

class ECGTemporalCNN(nn.Module):
    """
    Hierarchical CNN + BiGRU for beat-sequence Brugada classification.

    Level 1 (intra-beat): Shared 1D CNN encoder → morphological embedding per beat.
    Level 2 (inter-beat): BiGRU models how morphology evolves across N beats.
    RR interval appended to each beat embedding as scalar context.

    Input:
      x  : [batch, N_beats, in_channels, beat_len]
      rr : [batch, N_beats]

    Output:
      logits      : [batch] — Brugada logit
      beat_scores : [batch, N_beats] — per-beat attention weights
    """
    def __init__(self, in_channels=3, beat_len=200, n_beats=8,
                 cnn_embed_dim=64, gru_hidden=64, dropout=0.3):
        super(ECGTemporalCNN, self).__init__()
        self.n_beats = n_beats
        self.embed_dim = cnn_embed_dim

        # Shared CNN encoder
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, cnn_embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # RR interval projection
        self.rr_proj = nn.Linear(1, 8)

        # BiGRU over beat sequence
        gru_input_dim = cnn_embed_dim + 8
        self.bigru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Attention pooling
        self.beat_attn = nn.Linear(gru_hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_hidden * 2, 1)

    def encode_beat(self, x_beat):
        """x_beat: [batch, in_channels, beat_len] → [batch, cnn_embed_dim]"""
        feat = self.cnn_encoder(x_beat)
        return feat.squeeze(-1)

    def forward(self, x, rr):
        """
        x  : [batch, N_beats, in_channels, beat_len]
        rr : [batch, N_beats]
        """
        batch_size = x.size(0)

        # Encode each beat independently with shared CNN
        beat_embeddings = []
        for t in range(self.n_beats):
            emb = self.encode_beat(x[:, t, :, :])
            beat_embeddings.append(emb)
        beat_embeddings = torch.stack(beat_embeddings, dim=1)

        # RR interval projection
        rr_emb = self.rr_proj(rr.unsqueeze(-1))
        seq_input = torch.cat([beat_embeddings, rr_emb], dim=-1)

        # BiGRU
        gru_out, _ = self.bigru(seq_input)

        # Attention pooling
        attn_scores = self.beat_attn(gru_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = (attn_weights.unsqueeze(-1) * gru_out).sum(dim=1)

        logits = self.fc(self.dropout(context)).squeeze(-1)
        return logits, attn_weights


# ──────────────────────────────────────────────────────────────────────────────
# Training & Evaluation — supports both basic and temporal datasets
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device, is_temporal=False):
    """Trains one epoch. Set is_temporal=True for ECGBeatSequenceDataset batches."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        if is_temporal:
            X_batch, rr_batch, y_batch = batch
            X_batch = X_batch.to(device)
            rr_batch = rr_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_batch, rr_batch)
        else:
            X_batch, y_batch = batch
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(X_batch)

        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_batch.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device, is_temporal=False, threshold=0.5):
    """Evaluates model. Returns (all_preds, all_targets, all_probs)."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            if is_temporal:
                X_batch, rr_batch, y_batch = batch
                X_batch = X_batch.to(device)
                rr_batch = rr_batch.to(device)
                logits, _ = model(X_batch, rr_batch)
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                logits, _ = model(X_batch)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())
            all_probs.extend(probs)

        from sklearn.metrics import classification_report, matthews_corrcoef

        all_targets = np.array(all_targets).astype(int)  # float32 → int
        all_preds   = np.array(all_preds).astype(int)

        print(classification_report(all_targets, all_preds,
                                    target_names=['Normal', 'Brugada']))
        mcc = matthews_corrcoef(all_targets, all_preds)
        print(f"MCC: {mcc:.4f}")

        return all_preds, all_targets, np.array(all_probs)


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_attention_overlay(signal, attention_weights,
                           title="Attention Transparency Layer"):
    """Overlays attention weights as heatmap on ECG signal."""
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.arange(len(signal))
    ax.plot(time, signal, color='black', linewidth=1.5, label='ECG Signal')
    ymin, ymax = np.min(signal) - 0.5, np.max(signal) + 0.5
    ax.set_ylim(ymin, ymax)
    attn_heatmap = np.expand_dims(attention_weights, axis=0)
    ax.imshow(attn_heatmap, aspect='auto', cmap='Reds', alpha=0.5,
              extent=[0, len(signal), ymin, ymax])
    ax.set_title(title)
    ax.set_xlabel("Time (Samples)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_saliency_overlay(signal, saliency_map,
                           title="Gradient Saliency Transparency"):
    """Overlays saliency map on ECG signal."""
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.arange(len(signal))
    ax.plot(time, signal, color='black', linewidth=1.5, label='ECG Signal')
    ymin, ymax = np.min(signal) - 0.5, np.max(signal) + 0.5
    ax.set_ylim(ymin, ymax)
    sal_norm = saliency_map / (np.max(saliency_map) + 1e-8)
    sal_heatmap = np.expand_dims(sal_norm, axis=0)
    ax.imshow(sal_heatmap, aspect='auto', cmap='Blues', alpha=0.5,
              extent=[0, len(signal), ymin, ymax])
    ax.set_title(title)
    ax.set_xlabel("Time (Samples)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Shape verification for ECGTemporalCNN
    dummy_x = torch.randn(4, 8, 3, 200)
    dummy_rr = torch.randn(4, 8)
    #model = ECGTemporalCNN(in_channels=3, beat_len=200, n_beats=8)
    logits, attn = model(dummy_x, dummy_rr)
    assert logits.shape == (4,), f"Logits shape error: {logits.shape}"
    assert attn.shape == (4, 8), f"Attention shape error: {attn.shape}"
    assert torch.allclose(attn.sum(dim=-1), torch.ones(4), atol=1e-5), \
        "Attention weights do not sum to 1"
    print("[PASS] ECGTemporalCNN forward pass verified.")
    print("Deep Learning Pipeline v2 ready.")
