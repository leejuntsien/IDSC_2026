"""
data_loader.py — WFDB record loading, multi-lead discrete feature extraction,
                 and sequence extraction for DL.
V2 overhaul: beat_id/patient_id traceability, interpolation default,
             return_rr option, RMS sequence support.
"""
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
    df = pd.DataFrame(signals, columns=lead_names)
    return df, fs


def extract_discrete_features(lead_signals_df, fs, patient_id,
                               target_leads=None):
    """
    Extracts discrete/tabular beat-by-beat features for Classic ML.
    Multi-lead: runs process_single_lead per lead, prefixes columns, merges on beat_index.

    Args:
        lead_signals_df : DataFrame of raw signals, columns = lead names
        fs              : sampling frequency (Hz)
        patient_id      : str, patient identifier for traceability
        target_leads    : list of lead names or single string. Default: all available.

    Returns:
        DataFrame with columns prefixed by lead name, plus beat_id and patient_id.
    """
    if target_leads is None:
        target_leads = list(lead_signals_df.columns)
    if isinstance(target_leads, str):
        target_leads = [target_leads]

    all_leads_features = []

    for lead in target_leads:
        if lead not in lead_signals_df.columns:
            continue

        signal = lead_signals_df[lead].values
        try:
            df_features = epf.process_single_lead(signal, sampling_rate=fs)
        except Exception:
            continue

        if len(df_features) == 0:
            continue

        # Rename columns to include lead prefix (except shared indices)
        rename_dict = {
            col: f"{lead}_{col}"
            for col in df_features.columns
            if col not in ['beat_index', 'period_s']
        }
        df_features = df_features.rename(columns=rename_dict)
        all_leads_features.append(df_features)

    if not all_leads_features:
        raise ValueError(
            f"Patient {patient_id}: None of the target leads "
            f"{target_leads} yielded features."
        )

    # Merge all DataFrames on beat_index and period_s
    merged_df = all_leads_features[0]
    for df in all_leads_features[1:]:
        merged_df = pd.merge(merged_df, df, on=['beat_index', 'period_s'], how='inner')

    # Insert traceability columns
    merged_df.insert(0, 'patient_id', patient_id)
    first_lead = target_leads[0]
    merged_df.insert(1, 'beat_id', [
        f"{patient_id}_{first_lead}_{i}" for i in merged_df['beat_index']
    ])

    return merged_df


# ── Sequence helpers ─────────────────────────────────────────────────────────

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


def extract_sequence_features(lead_signals_df, fs, target_lead='I',
                               use_vcg=False, use_all_leads=False,
                               use_rms=False, method='interpolate',
                               target_len=200, return_rr=False):
    """
    Extracts time-series sequence features for Deep Learning.

    Args:
        lead_signals_df : DataFrame of raw signals
        fs              : sampling frequency
        target_lead     : lead to extract if single-lead mode
        use_vcg         : if True, convert to VCG (3 channels)
        use_all_leads   : if True, extract all leads (N channels)
        use_rms         : if True, compute RMS across all leads (1 channel)
        method          : 'interpolate' (default, preserves morphology) or 'pad'
        target_len      : target sequence length per beat
        return_rr       : if True, also return RR intervals array

    Returns:
        sequences : np.array [num_beats, seq_len, num_channels]
        rr_array  : np.array [num_beats] (only if return_rr=True)
    """
    def _process_single(signal):
        sig_clean = epf.apply_notch_filter(signal, sampling_rate=fs)
        rpeaks = epf.detect_peaks(sig_clean, sampling_rate=fs)
        segments = epf.segment_beats_by_rr(sig_clean, rpeaks, sampling_rate=fs)
        raw_seqs = [seg['signal'] for seg in segments]
        rr_vals = [seg['period_s'] for seg in segments]
        if method == 'pad':
            proc_seqs, _ = pad_sequences(raw_seqs, max_len=target_len)
        else:
            proc_seqs = interpolate_sequences(raw_seqs, target_len=target_len)
        return proc_seqs, np.array(rr_vals)

    def _process_multi(signals, ref_signal):
        ref_clean = epf.apply_notch_filter(ref_signal, sampling_rate=fs)
        rpeaks = epf.detect_peaks(ref_clean, sampling_rate=fs)
        segments_ref = epf.segment_beats_by_rr(ref_clean, rpeaks, sampling_rate=fs)
        rr_vals = [seg['period_s'] for seg in segments_ref]

        all_channels_seqs = []
        for sig in signals:
            sig_clean = epf.apply_notch_filter(sig, sampling_rate=fs)
            segments = epf.segment_beats_by_rr(sig_clean, rpeaks, sampling_rate=fs)
            raw_seqs = [seg['signal'] for seg in segments]
            if method == 'pad':
                proc_seqs, _ = pad_sequences(raw_seqs, max_len=target_len)
            else:
                proc_seqs = interpolate_sequences(raw_seqs, target_len=target_len)
            all_channels_seqs.append(proc_seqs)

        result = np.stack(all_channels_seqs, axis=-1)
        return result, np.array(rr_vals)

    if use_rms:
        signals_matrix = lead_signals_df.values
        rms_signal = np.sqrt(np.mean(np.square(signals_matrix), axis=1))
        proc_seqs, rr_vals = _process_single(rms_signal)
        result = np.expand_dims(proc_seqs, axis=-1)

    elif use_all_leads:
        signals = [lead_signals_df[col].values for col in lead_signals_df.columns]
        ref_signal = signals[0]
        result, rr_vals = _process_multi(signals, ref_signal)

    elif use_vcg:
        vcg_df = epf.combine_to_vcg(lead_signals_df)
        signals = [vcg_df['VCG_x'].values, vcg_df['VCG_y'].values, vcg_df['VCG_z'].values]
        ref_signal = (lead_signals_df[target_lead].values
                      if target_lead in lead_signals_df.columns else signals[0])
        result, rr_vals = _process_multi(signals, ref_signal)

    else:
        if target_lead not in lead_signals_df.columns:
            raise ValueError(f"Lead {target_lead} not found.")
        signal = lead_signals_df[target_lead].values
        proc_seqs, rr_vals = _process_single(signal)
        result = np.expand_dims(proc_seqs, axis=-1)

    if return_rr:
        return result, rr_vals
    return result


if __name__ == "__main__":
    print("DataLoader v2 initialized.")
