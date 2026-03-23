import os
import sys
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample

# Add parent directory to path to import ecg_pipeline_features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ecg_pipeline_features as epf

def load_wfdb_record(record_path):
    """Loads a WFDB record and returns the signal DataFrame and sampling frequency."""
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal
    lead_names = record.sig_name
    fs = record.fs
    
    # Create DataFrame
    df = pd.DataFrame(signals, columns=lead_names)
    return df, fs

def extract_discrete_features(lead_signals_df, fs, target_leads=['V1', 'V2', 'V3']):
    """
    Extracts purely discrete/tabular beat-by-beat features for Classic ML.
    Target leads can be a single string or a list of strings.
    Extracts features for all specified leads and merges them.
    """
    if isinstance(target_leads, str):
        target_leads = [target_leads]
        
    all_leads_features = []
    
    for lead in target_leads:
        if lead not in lead_signals_df.columns:
            print(f"Warning: Lead {lead} not found in record.")
            continue
            
        signal = lead_signals_df[lead].values
        df_features = epf.process_single_lead(signal, sampling_rate=fs)
        
        # Rename columns to include the lead name, except for shared indices
        rename_dict = {col: f"{lead}_{col}" for col in df_features.columns if col not in ['beat_index', 'period_s']}
        df_features = df_features.rename(columns=rename_dict)
        all_leads_features.append(df_features)
        
    if not all_leads_features:
        raise ValueError(f"None of the target leads {target_leads} were found, or extraction failed.")
        
    # Merge all DataFrames on beat_index and period_s
    merged_df = all_leads_features[0]
    for df in all_leads_features[1:]:
        merged_df = pd.merge(merged_df, df, on=['beat_index', 'period_s'], how='inner')
        
    return merged_df

def pad_sequences(sequences, max_len=None):
    """Pads sequences with zeros to match max_len."""
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    
    padded = np.zeros((len(sequences), max_len))
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return padded, max_len

def interpolate_sequences(sequences, target_len=200):
    """Resamples sequences to a fixed target length using scipy.signal.resample."""
    resampled = []
    for seq in sequences:
        if len(seq) == 0:
            resampled.append(np.zeros(target_len))
        else:
            resampled.append(resample(seq, target_len))
    return np.array(resampled)

def extract_sequence_features(lead_signals_df, fs, target_lead='I', use_vcg=False, use_all_leads=False, use_rms=False, method='pad', target_len=200):
    """
    Extracts time-series sequence features for Deep Learning.
    If use_all_leads is True, returns [num_beats, seq_len, num_leads].
    If use_vcg is True, returns a 3D tensor of shape [num_beats, seq_len, 3] for VCG (X, Y, Z).
    If use_rms is True, calculates RMS of all leads and returns shape [num_beats, seq_len, 1].
    Otherwise returns [num_beats, seq_len, 1] for target_lead.
    method: 'pad' (zero-padding) or 'interpolate' (scipy.signal.resample)
    """
    if use_rms:
        signals_matrix = lead_signals_df.values
        rms_signal = np.sqrt(np.mean(np.square(signals_matrix), axis=1))
        
        sig_clean = epf.apply_notch_filter(rms_signal, sampling_rate=fs)
        rpeaks = epf.detect_peaks(sig_clean, sampling_rate=fs)
        segments = epf.segment_beats_by_rr(sig_clean, rpeaks, sampling_rate=fs)
        raw_seqs = [seg['signal'] for seg in segments]
        
        if method == 'pad':
            proc_seqs, _ = pad_sequences(raw_seqs, max_len=target_len if target_len else None)
        else:
            proc_seqs = interpolate_sequences(raw_seqs, target_len=target_len)
            
        return np.expand_dims(proc_seqs, axis=-1)
        
    elif use_all_leads or use_vcg:
        if use_vcg:
            vcg_df = epf.combine_to_vcg(lead_signals_df)
            signals = [vcg_df['VCG_x'].values, vcg_df['VCG_y'].values, vcg_df['VCG_z'].values]
            ref_signal = lead_signals_df[target_lead].values if target_lead in lead_signals_df.columns else signals[0]
        else:
            signals = [lead_signals_df[col].values for col in lead_signals_df.columns]
            ref_signal = signals[0]
            
        ref_clean = epf.apply_notch_filter(ref_signal, sampling_rate=fs)
        rpeaks = epf.detect_peaks(ref_clean, sampling_rate=fs)
        
        all_channels_seqs = []
        for sig in signals:
            sig_clean = epf.apply_notch_filter(sig, sampling_rate=fs)
            segments = epf.segment_beats_by_rr(sig_clean, rpeaks, sampling_rate=fs)
            raw_seqs = [seg['signal'] for seg in segments]
            
            if method == 'pad':
                proc_seqs, _ = pad_sequences(raw_seqs, max_len=target_len if target_len else None)
            else:
                proc_seqs = interpolate_sequences(raw_seqs, target_len=target_len)
                
            all_channels_seqs.append(proc_seqs) # shape: [num_beats, seq_len]
            
        # Stack into [num_beats, seq_len, num_channels]
        return np.stack(all_channels_seqs, axis=-1)
    
    else:
        if target_lead not in lead_signals_df.columns:
            raise ValueError(f"Lead {target_lead} not found.")
            
        signal = lead_signals_df[target_lead].values
        sig_clean = epf.apply_notch_filter(signal, sampling_rate=fs)
        rpeaks = epf.detect_peaks(sig_clean, sampling_rate=fs)
        segments = epf.segment_beats_by_rr(sig_clean, rpeaks, sampling_rate=fs)
        raw_seqs = [seg['signal'] for seg in segments]
        
        if method == 'pad':
            proc_seqs, _ = pad_sequences(raw_seqs, max_len=target_len if target_len else None)
        else:
            proc_seqs = interpolate_sequences(raw_seqs, target_len=target_len)
            
        # Shape: [num_beats, seq_len, 1]
        return np.expand_dims(proc_seqs, axis=-1)

if __name__ == "__main__":
    # Minimal test block
    print("DataLoader initialized.")
