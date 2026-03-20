import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit

# Import our customized pipeline scripts
from ml_pipeline.data_loader import load_wfdb_record, extract_sequence_features
from ml_pipeline.dl_pipeline import ECGAttentionLSTM, ECGSequenceDataset, plot_attention_overlay, train_epoch, evaluate

def main():
    print("==========================================================")
    print("   Deep Learning Explainability Pipeline (12-Lead CNN)    ")
    print("==========================================================")
    print("\\n--- 1. Loading Metadata ---")
    metadata_path = "brugada-huca/metadata.csv"
    metadata = pd.read_csv(metadata_path)
    
    # For demonstration, we'll take a subset of 10 Brugada and 10 Normal subjects.
    subset_meta = pd.concat([
        metadata[metadata['brugada'] == 1].head(10),
        metadata[metadata['brugada'] == 0].head(10)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_all = []
    y_all = []
    groups_all = []
    
    print("--- 2. Extracting 12-lead Sequences & Building Dataset ---")
    for idx, row in subset_meta.iterrows():
        patient_id = str(row['patient_id'])
        label = float(row['brugada']) # 1.0 for Brugada, 0.0 for Normal
        
        record_path = f"brugada-huca/files/{patient_id}/{patient_id}"
        if not os.path.exists(record_path + ".dat"):
            continue
            
        try:
            df, fs = load_wfdb_record(record_path)
            
            # Extract ALL 12 leads and interpolate to 200 samples per beat
            X_seq = extract_sequence_features(df, fs, use_all_leads=True, method='interpolate', target_len=200)
            
            # X_seq shape is [num_beats, 200, 12].
            X_all.append(X_seq)
            y_all.extend([label] * len(X_seq))
            groups_all.extend([patient_id] * len(X_seq))
        except Exception as e:
            print(f"Skipping patient {patient_id} due to error: {e}")
            
    X_all = np.vstack(X_all) # Shape: [total_beats, 200, 12]
    y_all = np.array(y_all)  # Shape: [total_beats]
    groups_all = np.array(groups_all)
    
    print(f"Total Beats Extracted: {len(X_all)}")
    
    print("\\n--- 3. Preparing PyTorch DataLoaders ---")
    dataset = ECGSequenceDataset(X_all, y_all)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(X_all, y_all, groups=groups_all))
    
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    print("\\n--- 4. Initializing Explainability Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Input channels = 12 because we are processing the entire multi-lead ECG
    model = ECGAttentionLSTM(input_size=12, hidden_size=64, num_layers=2, num_classes=1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss() 
    
    print("\\n--- 5. Training the 12-Lead Attention LSTM Model ---")
    epochs = 3
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {loss:.4f}")
        
    print("\\n--- 6. Evaluating the Model ---")
    evaluate(model, val_loader, device)
    
    print("\\n--- 7. EXPLAINABILITY: GRADIENT SALIENCY MAP ---")
    # Pick a Brugada beat to map gradients onto
    brugada_indices = np.where(y_all == 1)[0]
    
    if len(brugada_indices) > 0:
        sample_idx = brugada_indices[0]
        sample_beat = X_all[sample_idx] # shape: [200, 12]
        
        # PyTorch expects [batch, channels, seq_len]
        sample_tensor = torch.tensor(sample_beat, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to(device)
        
        model.eval()
        with torch.no_grad():
            logits, attn_weights = model(sample_tensor)
            
        prob = torch.sigmoid(logits).item()
        print(f"Evaluating 12-lead beat #{sample_idx} -> Probability of Brugada: {prob:.4f}")
        
        # Plot raw signal mapped against its attention weights
        signal_to_plot = sample_beat[:, 0] 
        attn_to_plot = attn_weights.squeeze(0).cpu().numpy()
        
        print("Generating precise Attention heatmap over the identified ECG problematic segment...")
        plot_attention_overlay(signal_to_plot, attn_to_plot, title=f"Attention Transparency Map (Prob: {prob:.4f})")

if __name__ == "__main__":
    main()
