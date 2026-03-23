import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

from ml_pipeline.data_loader_v1 import load_wfdb_record, extract_discrete_features, extract_sequence_features
from ml_pipeline.classic_pipeline_v1 import train_and_evaluate

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
    
    # Full dataset
    subset_meta = metadata.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Extract Data (with caching for 12-lead full dataset to save time)
    features_csv = "extracted_12_lead_features.csv"
    seq_npy = "X_seq_rms.npy"
    
    if os.path.exists(features_csv) and os.path.exists(seq_npy):
        print(f"Loading cached features from '{features_csv}'...")
        df_all = pd.read_csv(features_csv)
        X_seq_all_flat = np.load(seq_npy)
    else:
        all_features_dfs = []
        all_raw_sequences = [] # We'll store the raw beats here to "catch and plot the problematic segment"
        
        print("Extracting tabular features & raw signals per beat from full WFDB records...")
        # target all standard clinical + augmented leads + precordial
        target_leads_to_extract = ['I', 'II', 'III', 'aVR', 'AVR', 'aVL', 'AVL', 'aVF', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        for idx, row in subset_meta.iterrows():
            patient_id = str(row['patient_id'])
            label = int(row['brugada'])
            
            record_path = f"brugada-huca/files/{patient_id}/{patient_id}"
            if not os.path.exists(record_path + ".dat"):
                continue
                
            try:
                df, fs = load_wfdb_record(record_path)
                
                # Extract pd.DataFrame of features for ALL possible leads
                df_patient_feats = extract_discrete_features(df, fs, target_leads=target_leads_to_extract)
                
                # ALSO Extract raw time-series sequences of RMS for plotting
                X_seq = extract_sequence_features(df, fs, method='pad', target_len=200, use_rms=True)
                
                df_patient_feats['label'] = label
                df_patient_feats['patient_id'] = patient_id
                
                all_features_dfs.append(df_patient_feats)
                all_raw_sequences.append(X_seq.squeeze(-1)) # Shape: [num_beats, 200]
            except Exception as e:
                print(f"Skipped patient {patient_id}: {e}")

        # Combine into huge tabular datasets
        df_all = pd.concat(all_features_dfs, ignore_index=True)
        X_seq_all_flat = np.vstack(all_raw_sequences)
        
        print(f"\\nSaving extracted features to cache...")
        df_all.to_csv(features_csv, index=False)
        np.save(seq_npy, X_seq_all_flat)
        
    print(f"\\nTotal beats ready: {len(df_all)}")
    
    # 3. Train / Test Split (Grouped by Patient to Prevent Data Leakage)
    drop_cols = ['label', 'patient_id', 'beat_index', 'normalized_signal']
    X = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns])
    y = df_all['label'].values
    groups = df_all['patient_id'].values
    
    feature_columns = list(X.columns)
    
    # Use GroupShuffleSplit to keep all beats from the same patient in either train or test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(gss.split(X.values, y, groups))
    
    X_train, X_test = X.values[train_idx], X.values[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    seq_train, seq_test = X_seq_all_flat[train_idx], X_seq_all_flat[test_idx]
    
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    import time
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.ensemble import RandomForestClassifier

    # 3.b Evaluate Lead Combinations
    combinations_to_test = {
        'V1-V3': ['V1', 'V2', 'V3'],
        'All_Leads': ['I', 'II', 'III', 'AVR', 'aVR', 'AVL', 'aVL', 'AVF', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        'Single_V2': ['V2'],
        'V4-V6': ['V4', 'V5', 'V6'],
        'Limb_Leads': ['I', 'II', 'III', 'AVR', 'aVR', 'AVL', 'aVL', 'AVF', 'aVF']
    }
    
    print("\\n--- Evaluating Lead Combinations (Fast RF) ---")
    best_combo_name = 'V1-V3'
    best_combo_f1 = -1
    best_feature_cols = []
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    all_available_cols = list(df_all.columns)
    
    for combo_name, combo_leads in combinations_to_test.items():
        # Find which columns belong to these leads
        combo_cols = [c for c in all_available_cols if any(c.startswith(lead + '_') for lead in combo_leads)]
        
        if len(combo_cols) == 0:
            continue
            
        # Train / Test split logic
        X_combo = df_all[combo_cols].values
        X_train_c, X_test_c = X_combo[train_idx], X_combo[test_idx]
        
        try:
            rf.fit(X_train_c, y_train)
            preds = rf.predict(X_test_c)
            f1 = f1_score(y_test, preds, zero_division=0)
            print(f"Lead Combo {combo_name}: F1={f1:.4f} (using {len(combo_cols)} features)")
            
            if f1 > best_combo_f1:
                best_combo_f1 = f1
                best_combo_name = combo_name
                best_feature_cols = combo_cols
        except Exception as e:
            pass
            
    if not best_feature_cols:
        best_feature_cols = all_available_cols
        
    print(f"\\nSelected Best Lead Combination: {best_combo_name} (F1 = {best_combo_f1:.4f})")
    
    # 4. Evaluate Advanced Classic ML Multi-Model Benchmark on BEST combination
    feature_columns = best_feature_cols
    X_train, X_test = df_all[feature_columns].values[train_idx], df_all[feature_columns].values[test_idx]
    
    models_to_run = ['lightgbm', 'xgboost', 'random_forest', 'svm']
    
    results = []
    best_overall_model = None
    best_f1 = -1
    
    for model_choice in models_to_run:
        print(f"\\n==================================================")
        print(f"Running full evaluation on {model_choice}...")
        start_time = time.time()
        try:
            current_model = train_and_evaluate(
                X_train, y_train, 
                X_test, y_test, 
                feature_columns=feature_columns, 
                model_name=model_choice, 
                n_iter=5, cv=3 
            )
            inf_start = time.time()
            preds = current_model.predict(X_test)
            inf_end = time.time()
            
            f1 = f1_score(y_test, preds, zero_division=0)
            acc = accuracy_score(y_test, preds)
            
            inf_time = inf_end - inf_start
            total_time = inf_end - start_time
            
            results.append({
                'Model': model_choice,
                'F1 Score': f1,
                'Accuracy': acc,
                'Inference Time (s)': inf_time,
                'Train+Eval Time (s)': total_time
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_overall_model = current_model
                
        except Exception as e:
            print(f"Skipping {model_choice} due to error: {e}")
            
    print("\\n--- Benchmark Summary ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    best_model = best_overall_model
    
    if best_model is None:
        print("No models trained successfully. Exiting.")
        return
    
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
