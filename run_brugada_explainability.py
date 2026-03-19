import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Import our customized pipeline scripts
from ml_pipeline.data_loader import load_wfdb_record, extract_sequence_features
from ml_pipeline.dl_pipeline import ECGCNN1D, ECGSequenceDataset, plot_saliency_overlay, train_epoch, evaluate

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
        except Exception as e:
            print(f"Skipping patient {patient_id} due to error: {e}")
            
    X_all = np.vstack(X_all) # Shape: [total_beats, 200, 12]
    y_all = np.array(y_all)  # Shape: [total_beats]
    
    print(f"Total Beats Extracted: {len(X_all)}")
    
    print("\\n--- 3. Preparing PyTorch DataLoaders ---")
    dataset = ECGSequenceDataset(X_all, y_all)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    print("\\n--- 4. Initializing Explainability Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Input channels = 12 because we are processing the entire multi-lead ECG
    model = ECGCNN1D(in_channels=12, num_classes=1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss() 
    
    print("\\n--- 5. Training the 12-Lead CNN Model ---")
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
        
        # We explicitly require gradients for the input to compute Saliency
        sample_tensor.requires_grad_()
        
        model.eval()
        # Ensure we compute gradients
        model.zero_grad()
        logits, _ = model(sample_tensor)
        
        # Backpropagate to extract the numerical influence of each input sample
        logits.backward()
        
        # Raw gradients -> Saliency (Take absolute magnitude)
        # Saliency shape is [12 channels, 200 samples]
        saliency_map_12_lead = sample_tensor.grad.abs().squeeze(0).cpu().numpy() 
            
        prob = torch.sigmoid(logits).item()
        print(f"Evaluating 12-lead beat #{sample_idx} -> Probability of Brugada: {prob:.4f}")
        
        # Let's plot the raw signal of Lead 0 (e.g. Lead I) mapped against its channel's gradient saliency
        signal_to_plot = sample_beat[:, 0] 
        attn_to_plot = saliency_map_12_lead[0, :]
        
        print("Generating precise Saliency heatmap over the identified ECG problematic segment...")
        plot_saliency_overlay(signal_to_plot, attn_to_plot, title=f"Gradient Saliency Transparency Map (Prob: {prob:.2f})")

if __name__ == "__main__":
    main()
