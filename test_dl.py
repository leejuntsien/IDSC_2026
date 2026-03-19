import sys
import pandas as pd
from ml_pipeline.data_loader import load_wfdb_record, extract_discrete_features, extract_sequence_features

def main():
    record_path = "brugada-huca/files/188981/188981"
    print(f"Loading record: {record_path}")
    df, fs = load_wfdb_record(record_path)
    
    print(f"Shape: {df.shape}, fs: {fs}")
    
    features = extract_discrete_features(df, fs, target_lead='I')
    print(f"Extracted {len(features)} discrete beats.")
    
    seqs_pad = extract_sequence_features(df, fs, target_lead='I', use_vcg=False, method='pad', target_len=300)
    print(f"Padded sequences shape (1D ECG): {seqs_pad.shape}")
    
    seqs_interp = extract_sequence_features(df, fs, target_lead='I', use_vcg=True, method='interpolate', target_len=200)
    print(f"Interpolated sequences shape (3D VCG): {seqs_interp.shape}")
    print("Test successful!")

if __name__ == "__main__":
    main()
