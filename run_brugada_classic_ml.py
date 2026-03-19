import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ml_pipeline.data_loader import load_wfdb_record, extract_discrete_features
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
    
    print("Extracting tabular features per beat from WFDB records...")
    for idx, row in subset_meta.iterrows():
        patient_id = str(row['patient_id'])
        label = int(row['brugada'])
        
        record_path = f"brugada-huca/files/{patient_id}/{patient_id}"
        if not os.path.exists(record_path + ".dat"):
            continue
            
        try:
            df, fs = load_wfdb_record(record_path)
            # Extracts pd.DataFrame of features (e.g. QRS_duration, VoltDiff_P_T)
            df_patient_feats = extract_discrete_features(df, fs, target_lead='I')
            
            # Stamp with target label
            df_patient_feats['label'] = label
            df_patient_feats['patient_id'] = patient_id
            
            all_features_dfs.append(df_patient_feats)
        except Exception as e:
            print(f"Skipped patient {patient_id}: {e}")

    # Combine into huge tabular dataset
    df_all = pd.concat(all_features_dfs, ignore_index=True)
    print(f"\\nTotal beats extracted: {len(df_all)}")
    
    # 3. Train / Test Split
    # Drop identifying metadata columns before training
    drop_cols = ['label', 'patient_id', 'beat_index', 'normalized_signal']
    X = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns])
    y = df_all['label'].values
    
    feature_columns = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, 
        test_size=0.30, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Features: {feature_columns}")
    
    # 4. Evaluate Advanced Classic ML!
    # Try lightgbm as it's typically best on tabular data, or random_forest
    model_choice = 'lightgbm' 
    # Fallback if lightgbm isn't installed
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
        n_iter=5, # Number of randomized search combinations
        cv=3      # CV folds
    )

if __name__ == "__main__":
    main()
