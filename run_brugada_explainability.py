import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Import our customized pipeline scripts
from ml_pipeline.data_loader import load_wfdb_record, extract_sequence_features
from ml_pipeline.dl_pipeline import ECGAttentionLSTM, ECGSequenceDataset, plot_attention_overlay, train_epoch, evaluate

def main():
    print("--- 1. Loading Metadata ---")
    metadata_path = "brugada-huca/metadata.csv"
    metadata = pd.read_csv(metadata_path)
    
    # For demonstration, we'll take a subset of 10 Brugada and 10 Normal subjects.
    # To run on the full dataset, remove the .head(10) slicing!
    subset_meta = pd.concat([
        metadata[metadata['brugada'] == 1].head(10),
        metadata[metadata['brugada'] == 0].head(10)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Arrays to hold all beat sequences and their labels
    X_all = []
    y_all = []
    
    print("--- 2. Extracting Sequences & Building Dataset ---")
    for idx, row in subset_meta.iterrows():
        patient_id = str(row['patient_id'])
        label = float(row['brugada']) # 1.0 for Brugada, 0.0 for Normal
        
        record_path = f"brugada-huca/files/{patient_id}/{patient_id}"
        if not os.path.exists(record_path + ".dat"):
            continue
            
        try:
            df, fs = load_wfdb_record(record_path)
            
            # Extract 3-channel VCG sequences, interpolated to exactly 200 samples per beat
            X_seq = extract_sequence_features(df, fs, use_vcg=True, method='interpolate', target_len=200)
            
            # X_seq shape is [num_beats, 200, 3]. We append it to our master list
            X_all.append(X_seq)
            
            # Since prediction is beat-by-beat, duplicate the patient label for each extracted beat
            y_all.extend([label] * len(X_seq))
        except Exception as e:
            print(f"Skipping patient {patient_id} due to error: {e}")
            
    # Stack all arrays into a single massive tensor
    X_all = np.vstack(X_all) # Shape: [total_beats, 200, 3]
    y_all = np.array(y_all)  # Shape: [total_beats]
    
    print(f"Total Beats Extracted: {len(X_all)}")
    
    print("\\n--- 3. Preparing PyTorch DataLoaders ---")
    dataset = ECGSequenceDataset(X_all, y_all)
    
    # 80/20 Train-Validation Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    print("\\n--- 4. Initializing Explainability Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Input size = 3 because we are using VCG_x, VCG_y, VCG_z
    model = ECGAttentionLSTM(input_size=3, hidden_size=64, num_layers=2, num_classes=1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Binary Cross Entropy for Brugada vs Normal
    criterion = nn.BCEWithLogitsLoss() 
    
    print("\\n--- 5. Training the Model ---")
    epochs = 3
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {loss:.4f}")
        
    print("\\n--- 6. Evaluating the Model ---")
    evaluate(model, val_loader, device)
    
    print("\\n--- 7. EXPLAINABILITY HEATMAP ---")
    # We will pick a beat known to belong to a Brugada patient to visualize the model's focus
    brugada_indices = np.where(y_all == 1)[0]
    
    if len(brugada_indices) > 0:
        sample_idx = brugada_indices[0]
        sample_beat = X_all[sample_idx] # shape: [200, 3]
        
        # Format the single sample for the PyTorch model
        sample_tensor = torch.tensor(sample_beat, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to(device)
        
        # Run forward pass to get logits and the internal explainability weights
        model.eval()
        with torch.no_grad():
            logits, attention_weights = model(sample_tensor)
            
        prob = torch.sigmoid(logits).item()
        print(f"Evaluating beat #{sample_idx} -> Probability of Brugada: {prob:.4f}")
        
        # Prepare the signal (we'll plot VCG_x, which is channel 0) and the attention weights
        signal_to_plot = sample_beat[:, 0] 
        attn_to_plot = attention_weights.squeeze(0).cpu().numpy()
        
        plot_attention_overlay(signal_to_plot, attn_to_plot, title=f"Explainability Heatmap (Brugada Prob: {prob:.2f})")

if __name__ == "__main__":
    main()
