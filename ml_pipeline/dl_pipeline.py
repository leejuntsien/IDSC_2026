import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class ECGSequenceDataset(Dataset):
    """
    Dataset for PyTorch to handle the time-series ECG sequences.
    X: numpy array of shape [num_samples, seq_len, num_channels]
    y: numpy array of shape [num_samples]
    """
    def __init__(self, X, y):
        # PyTorch CNNs expect [batch, channels, length] for 1D convolution
        # So we transpose X from [samples, length, channels] to [samples, channels, length]
        self.X = torch.tensor(X, dtype=torch.float32).transpose(1, 2)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ECGCNN1D(nn.Module):
    """
    A 1D CNN for detecting localized morphological changes in ECG/VCG.
    """
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
        self.pool3 = nn.AdaptiveMaxPool1d(1) # [batch, 64, 1]
        
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [batch, channels, seq_len]
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(x.size(0), -1) # flatten
        logits = self.fc(x).squeeze(1)
        return logits, None # Returning None for attention weights compatibility

class ECGAttentionLSTM(nn.Module):
    """
    Bidirectional LSTM with a custom Attention Mechanism to provide transparency.
    The attention weights highlight exactly which part of the ECG cycle (e.g. ST segment)
    triggered the anomaly.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=1):
        super(ECGAttentionLSTM, self).__init__()
        # LSTM expects [batch, seq, feature] if batch_first=True
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Attention layer
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x from dataset is [batch, channels, seq_len], we need [batch, seq_len, channels]
        x = x.transpose(1, 2)
        
        # lstm_out shape: [batch, seq_len, hidden_size * 2]
        lstm_out, _ = self.lstm(x)
        
        # Calculate attention weights
        # attn_scores shape: [batch, seq_len, 1]
        attn_scores = self.attention_fc(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1) # Softmax over seq_len
        
        # Aggregate lstm outputs using attention weights
        # context_vector shape: [batch, 1, hidden_size * 2]
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Classify
        logits = self.fc(context_vector).squeeze(-1)
        
        # Return logits and attention weights for transparency layer
        return logits, attn_weights.squeeze(-1)

class ECGTransformer(nn.Module):
    """
    A Transformer-based model for time-series ECG Classification.
    """
    def __init__(self, in_channels=1, d_model=64, nhead=4, num_layers=2, num_classes=1):
        super(ECGTransformer, self).__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: [batch, channels, seq_len], we need [batch, seq_len, channels]
        x = x.transpose(1, 2)
        
        # Project channels to d_model
        x = self.input_proj(x)
        
        # Transformer forward pass
        trans_out = self.transformer(x)
        
        # Global average pooling over the sequence
        pooled = trans_out.mean(dim=1)
        
        logits = self.fc(pooled).squeeze(1)
        return logits, None # Attention weights not explicitly returned here for simplicity

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(X_batch)
        loss = criterion(logits, y_batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits, _ = model(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())
            
    from sklearn.metrics import classification_report
    print(classification_report(all_targets, all_preds))

def plot_attention_overlay(signal, attention_weights, title="Attention Transparency Layer"):
    """
    Plots the ECG signal and overlays the attention weights as a heatmap background.
    Highlights the segment the model found anomalous (e.g. ST Coving in Brugada).
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Time axis
    time = np.arange(len(signal))
    
    # Plot signal
    ax.plot(time, signal, color='black', linewidth=1.5, label='ECG Signal')
    
    # Overlay attention as a colored background span
    # We use ax.pcolorfast or imshow to paint the background
    # Create an extent that covers the entire y-axis of the plot
    ymin, ymax = np.min(signal) - 0.5, np.max(signal) + 0.5
    ax.set_ylim(ymin, ymax)
    
    # Reshape attention for imshow
    attn_heatmap = np.expand_dims(attention_weights, axis=0)
    
    ax.imshow(attn_heatmap, aspect='auto', cmap='Reds', alpha=0.5, 
              extent=[0, len(signal), ymin, ymax])

    ax.set_title(title)
    ax.set_xlabel("Time (Samples)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_saliency_overlay(signal, saliency_map, title="Gradient Saliency Transparency"):
    """
    Plots the ECG signal and overlays the saliency map.
    The saliency map highlights the exact input samples that most influenced the model's decision.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.arange(len(signal))
    ax.plot(time, signal, color='black', linewidth=1.5, label='ECG Signal')
    
    ymin, ymax = np.min(signal) - 0.5, np.max(signal) + 0.5
    ax.set_ylim(ymin, ymax)
    
    # Normalize saliency to 0-1 for plotting
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
    print("Testing Transparency Plot...")
    # Mock signal and attention
    t = np.linspace(0, 10, 200)
    mock_signal = np.sin(t) + np.random.normal(0, 0.1, 200)
    mock_attn = np.zeros(200)
    mock_attn[100:120] = np.linspace(0, 1, 20) # Highlight a peak
    
    # Uncomment to test plot:
    # plot_attention_overlay(mock_signal, mock_attn)
    print("Deep Learning Pipeline and Modularity Setup Complete.")
