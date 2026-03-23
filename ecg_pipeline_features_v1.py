import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import find_peaks

def apply_notch_filter(ecg_signal, sampling_rate=500, powerline=50):
    """1. Apply 50Hz notch filter."""
    return nk.signal_filter(ecg_signal, sampling_rate=sampling_rate, lowcut=None, highcut=None, method="powerline", powerline=powerline)

def detect_peaks(ecg_cleaned, sampling_rate=500):
    """2. Peak detection using ecg_peaks."""
    _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    return info["ECG_R_Peaks"]

def delineate_segments(ecg_cleaned, rpeaks, sampling_rate=500):
    """3. Segment delineation."""
    # Using generic delineate to capture P, Q, S, T waves
    _, waves = nk.ecg_delineate(ecg_cleaned, rpeaks, sampling_rate=sampling_rate, method="dwt")
    return dict(waves)

def extract_rr_peaks(rpeaks):
    """4. Extract RR peaks (returning the indices)."""
    return rpeaks

def segment_beats_by_rr(ecg_signal, rpeaks, sampling_rate=500):
    """5. Each RR peak as a segment -> calc period & time between peaks in a list."""
    segments = []
    for i in range(len(rpeaks) - 1):
        start = rpeaks[i]
        end = rpeaks[i+1]
        
        period_s = (end - start) / sampling_rate
        segments.append({
            "beat_idx": i,
            "signal": ecg_signal[start:end],
            "start_idx": start,
            "end_idx": end,
            "period_s": period_s
        })
    return segments

def normalize_voltage(segment_signal, ecg_signal, r1_idx, r2_idx):
    """6. Normalize voltage: v(t) = v(t) / sqrt(v_r1 * v_r2)."""
    v_r1 = abs(ecg_signal[r1_idx])
    v_r2 = abs(ecg_signal[r2_idx])
    
    denominator = np.sqrt(v_r1 * v_r2)
    if denominator == 0:
        denominator = 1e-6
        
    return segment_signal / denominator

def extract_segment_differences(waves, ecg_signal, beat_idx, sampling_rate=500):
    """7. Calculate segment differences (time gap & voltage differences between P, Q, R, S, T)."""
    # R peak is injected from earlier
    keys = ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']
    beat_peaks = {}
    
    for k in keys:
        if k in waves and beat_idx < len(waves[k]):
            idx = waves[k][beat_idx]
            if not np.isnan(idx):
                beat_peaks[k.replace('ECG_', '').replace('_Peaks', '')] = int(idx)
                
    features = {}
    
    # Calculate time gaps (in seconds) between available peaks
    peak_names = list(beat_peaks.keys())
    for i, p1 in enumerate(peak_names):
        for j, p2 in enumerate(peak_names):
            if i < j:
                idx1 = beat_peaks[p1]
                idx2 = beat_peaks[p2]
                features[f'TimeGap_{p1}_{p2}_s'] = abs(idx2 - idx1) / sampling_rate
                features[f'VoltDiff_{p1}_{p2}'] = ecg_signal[idx2] - ecg_signal[idx1]
                
    return features

def extract_qrs_time(waves, beat_idx, sampling_rate=500):
    """8. Extract QRS time."""
    try:
        q_start = waves['ECG_R_Onsets'][beat_idx]
        s_end = waves['ECG_R_Offsets'][beat_idx]
        if not np.isnan(q_start) and not np.isnan(s_end):
            return (s_end - q_start) / sampling_rate
    except (KeyError, IndexError):
        pass
    return np.nan

def extract_st_segment(waves, beat_idx, sampling_rate=500):
    """9. Extract ST segment duration."""
    try:
        st_start = waves['ECG_R_Offsets'][beat_idx]
        st_end = waves['ECG_T_Onsets'][beat_idx]
        if not np.isnan(st_start) and not np.isnan(st_end):
            return (st_end - st_start) / sampling_rate
    except (KeyError, IndexError):
        pass
    return np.nan

def detect_u_waves(ecg_signal, waves, beat_idx, sampling_rate=500):
    """10. Detect U waves. Search for a peak between T_Offset and next P_Onset (or +400ms)."""
    try:
        t_offset = waves['ECG_T_Offsets'][beat_idx]
        if np.isnan(t_offset): return False, np.nan
        t_offset = int(t_offset)
        
        # Define search window for U wave
        search_end = t_offset + int(0.4 * sampling_rate)
        if search_end >= len(ecg_signal):
            search_end = len(ecg_signal) - 1
            
        window = ecg_signal[t_offset:search_end]
        peaks, _ = find_peaks(window, height=0.05 * np.max(ecg_signal)) # Minimum height heuristic
        
        if len(peaks) > 0:
            u_peak_idx = t_offset + peaks[0]
            return True, u_peak_idx
    except (KeyError, IndexError):
        pass
    return False, np.nan

def detect_inversion(ecg_signal, waves, beat_idx):
    """11. Detect T-wave inversion."""
    try:
        t_peak = waves['ECG_T_Peaks'][beat_idx]
        if not np.isnan(t_peak):
            is_inverted = ecg_signal[int(t_peak)] < 0
            return is_inverted
    except (KeyError, IndexError):
        pass
    return None

def combine_to_vcg(lead_signals_df):
    """Combine signals from 12 leads to form a VCG using the Kors regression matrix."""
    # Lead order: I, II, V1, V2, V3, V4, V5, V6
    kors_weights = np.array([
        [ 0.38, -0.07, -0.13,  0.05, -0.01,  0.14,  0.06,  0.54],
        [-0.07,  0.93,  0.06, -0.02, -0.05,  0.06, -0.17,  0.13],
        [ 0.11, -0.23, -0.43, -0.06, -0.04, -0.02, -0.10, -0.38]
    ])
    
    leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Ensure missing leads are zeroed out (or handle gracefully)
    for lead in leads:
        if lead not in lead_signals_df.columns:
            lead_signals_df[lead] = 0.0
            
    ecg_matrix = lead_signals_df[leads].values
    vcg_matrix = ecg_matrix.dot(kors_weights.T)
    
    vcg = pd.DataFrame(vcg_matrix, columns=['VCG_x', 'VCG_y', 'VCG_z'], index=lead_signals_df.index)
    return vcg

def process_single_lead(ecg_signal, sampling_rate=500):
    """Process a single ECG lead to extract beat-by-beat numerical features."""
    ecg_clean = apply_notch_filter(ecg_signal, sampling_rate)
    rpeaks = detect_peaks(ecg_clean, sampling_rate)
    
    # Needs ECG_R_Peaks injected for delineate mapping correctly
    # nk.ecg_delineate expects raw indices or dict with 'ECG_R_Peaks', but array of indices is standard
    waves = delineate_segments(ecg_clean, rpeaks, sampling_rate)
    waves['ECG_R_Peaks'] = rpeaks # Inject for convenience
    
    segments = segment_beats_by_rr(ecg_clean, rpeaks, sampling_rate)
    
    beat_features = []
    
    for seg in segments:
        idx = seg['beat_idx']
        
        r1_idx = seg['start_idx']
        r2_idx = seg['end_idx']
        v_norm = normalize_voltage(seg['signal'], ecg_clean, r1_idx, r2_idx)
        
        diffs = extract_segment_differences(waves, ecg_clean, idx, sampling_rate)
        qrs = extract_qrs_time(waves, idx, sampling_rate)
        st = extract_st_segment(waves, idx, sampling_rate)
        has_u, u_idx = detect_u_waves(ecg_clean, waves, idx, sampling_rate)
        invert = detect_inversion(ecg_clean, waves, idx)
        
        beat_params = {
            "beat_index": idx,
            "period_s": seg['period_s'],
            # "normalized_signal": v_norm.tolist(), # Optional: can be very large in a DataFrame, uncomment to store
            "QRS_duration_s": qrs,
            "ST_segment_s": st,
            "has_U_wave": has_u,
            "T_wave_inversion": invert,
        }
        beat_params.update(diffs)
        beat_features.append(beat_params)
        
    return pd.DataFrame(beat_features)

if __name__ == "__main__":
    # Test execution block to ensure pipeline logic runs successfully.
    print("NeuroKit pipeline initialized. Use process_single_lead(signal, fs) to extract numerical beat features.")
    
    # Mock data for brief testing
    ecg = nk.ecg_simulate(duration=10, sampling_rate=500)
    df_features = process_single_lead(ecg, sampling_rate=500)
    print("Successfully processed simulated ECG. Number of beats:", len(df_features))
    print(df_features.head())
