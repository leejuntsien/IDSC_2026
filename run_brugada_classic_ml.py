import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ml_pipeline.data_loader import load_wfdb_record, extract_discrete_features, extract_sequence_features
from ml_pipeline.classic_pipeline import train_and_evaluate

def main():
    print("==================================================")
    print("   Classic ML Pipeline (Discrete Beat Features)   ")
    print("==================================================")
    
    # 1. Load Metadata
    metadata_path = "brugada-huca/metadata.csv"
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found. Please ensure Physionet data is downloaded.")
        return
        
    metadata = pd.read_csv(metadata_path)
    
    # Demonstration subset (10 Brugada, 10 Normal) to make the run fast.
    subset_meta = pd.concat([
        metadata[metadata['brugada'] == 1].head(10),
        metadata[metadata['brugada'] == 0].head(10)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Extract Data
    all_features_dfs = []
    all_raw_sequences = [] # We'll store the raw beats here to "catch and plot the problematic segment"
    
    print("Extracting tabular features & raw signals per beat from WFDB records...")
    for idx, row in subset_meta.iterrows():
        patient_id = str(row['patient_id'])
        label = int(row['brugada'])
        
        record_path = f"brugada-huca/files/{patient_id}/{patient_id}"
        if not os.path.exists(record_path + ".dat"):
            continue
            
        try:
            df, fs = load_wfdb_record(record_path)
            
            # Extract pd.DataFrame of features (e.g. QRS_duration, VoltDiff_P_T)
            df_patient_feats = extract_discrete_features(df, fs, target_lead='I')
            
            # ALSO Extract raw time-series sequences of Lead I just for plotting comparisons later!
            X_seq = extract_sequence_features(df, fs, target_lead='I', method='pad', target_len=200)
            
            df_patient_feats['label'] = label
            df_patient_feats['patient_id'] = patient_id
            
            all_features_dfs.append(df_patient_feats)
            all_raw_sequences.append(X_seq.squeeze(-1)) # Shape: [num_beats, 200]
        except Exception as e:
            print(f"Skipped patient {patient_id}: {e}")

    # Combine into huge tabular datasets
    df_all = pd.concat(all_features_dfs, ignore_index=True)
    X_seq_all_flat = np.vstack(all_raw_sequences)
    print(f"\\nTotal beats extracted: {len(df_all)}")
    
    # 3. Train / Test Split
    drop_cols = ['label', 'patient_id', 'beat_index', 'normalized_signal']
    X = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns])
    y = df_all['label'].values
    
    feature_columns = list(X.columns)
    
    # Notice we slice the raw sequences array synchronously using the state splitter
    X_train, X_test, y_train, y_test, seq_train, seq_test = train_test_split(
        X.values, y, X_seq_all_flat,
        test_size=0.30, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # 4. Evaluate Advanced Classic ML
    model_choice = 'lightgbm' 
    try:
        import lightgbm
    except ImportError:
        model_choice = 'random_forest'
        
    print(f"\\nRunning full evaluation on {model_choice}...")
    best_model = train_and_evaluate(
        X_train, y_train, 
        X_test, y_test, 
        feature_columns=feature_columns, 
        model_name=model_choice, 
        n_iter=5, cv=3 
    )
    
    # 5. Explainability Layer (Catching Flagged Beats vs Normal Beats)
    print("\\n--- Generating Flagged Segment Comparisons ---")
    preds = best_model.predict(X_test)
    
    # Find True Positives (Model caught an Anomaly/Brugada)
    tp_idx = np.where((y_test == 1) & (preds == 1))[0]
    # Find True Negatives (Model confirmed Normal)
    tn_idx = np.where((y_test == 0) & (preds == 0))[0]
    
    if len(tp_idx) > 0 and len(tn_idx) > 0:
        tp_seq = seq_test[tp_idx[0]]
        tn_seq = seq_test[tn_idx[0]]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        axes[0].plot(tn_seq, color='green', lw=2)
        axes[0].set_title("Normal Beat Signal (Correctly Classified)")
        axes[0].set_xlabel("Time (Samples)")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(tp_seq, color='red', lw=2)
        axes[1].set_title("Brugada Beat Signal (Anomaly Flagged by ML)")
        axes[1].set_xlabel("Time (Samples)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle("Classic ML Transparency: Comparing Clean vs Flagged Segments", fontweight='bold')
        plt.tight_layout()
        plt.show()
        print("Plotted side-by-side normal vs anomaly segments successfully.")

if __name__ == "__main__":
    main()
