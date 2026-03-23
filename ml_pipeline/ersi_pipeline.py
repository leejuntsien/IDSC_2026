import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import neurokit2 as nk
import warnings

from ersi.ersi import ERSI
from ersi.entropy_measures import SimpleEntropy, NonExtensiveEntropy, entropy_funcs

warnings.filterwarnings('ignore')

def process_patient(ecg_signal, fs=1000, window_sec=5.0, step_sec=2.5):
    """
    Process patient ECG signal to compute the full ERSI pipeline.
    
    Parameters
    ----------
    ecg_signal : array-like
        The raw ECG signal.
    fs : int
        The sampling frequency of the signal.
    window_sec : float
        Sliding window size in seconds.
    step_sec : float
        Step size between windows in seconds.
        
    Returns
    -------
    dict
        A dictionary containing the aggregate ERSI value and the mean of the standard entropies.
    """
    # Clean the ECG signal (NeuroKit2 handles NaNs and artifacts)
    try:
        cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs)
    except Exception:
        cleaned_ecg = ecg_signal # Fallback if cleaning fails
        
    # Define window size and step in samples
    window_size = max(10, int(window_sec * fs))
    step_size = max(1, int(step_sec * fs))
    
    # Compute sliding window entropies
    df_entropy = SimpleEntropy.sliding_window_entropy(
        cleaned_ecg, funcs=entropy_funcs, window=window_size, step=step_size
    )
    
    entropies = list(entropy_funcs.keys())
    
    if df_entropy.empty:
        warnings.warn(f"Signal length ({len(ecg_signal)}) is shorter than window size ({window_size}). Cannot compute sliding windows.")
        results = {col: np.nan for col in entropies}
        for col in entropies:
            results[f"{col}_ERSI"] = np.nan
        results['ERSI_timeseries'] = np.nan
        results['ERSI_full'] = np.nan
        return results
        
    # Apply computational ERSI (computes simple _ERSI columns with time weights)
    df_ersi = ERSI.ERSI_computation(df_entropy, entropies)
    
    # Calculate ERSI_timeseries (sum of all _ERSI columns per window)
    if any(col.endswith('_ERSI') for col in df_ersi.columns):
        ersi_cols = [col for col in df_ersi.columns if col.endswith('_ERSI')]
        df_ersi['ERSI_timeseries'] = df_ersi[ersi_cols].sum(axis=1)
        
    # Calculate ERSI_full (dual-ranking: time and cross-entropy)
    df_full = ERSI.ERSI_full(df_entropy, entropies)
    df_ersi['ERSI_full'] = df_full['ERSI_full']
    
    # Collect results
    results = df_entropy.mean().to_dict()
    
    # 1. Simple Single-Measure ERSI (e.g. shannon_ERSI, tsallis_ERSI)
    for col in df_ersi.columns:
        if col.endswith('_ERSI'):
            results[col] = df_ersi[col].mean()
            
    # 2. ERSI_timeseries (Additive Fusion)
    if 'ERSI_timeseries' in df_ersi.columns:
        results['ERSI_timeseries'] = df_ersi['ERSI_timeseries'].mean()
    else:
        results['ERSI_timeseries'] = np.nan
        
    # 3. ERSI_full (Multiplicative Dual-Rank Fusion)
    if 'ERSI_full' in df_ersi.columns:
        results['ERSI_full'] = df_ersi['ERSI_full'].mean()
    else:
        results['ERSI_full'] = np.nan
        
    return results

def process_patient_tsallis(ecg_signal, fs=1000, q=2, window_sec=5.0, step_sec=2.5):
    """
    Process patient ECG signal to compute the Tsallis-only ERSI.
    
    Parameters
    ----------
    ecg_signal : array-like
        The raw ECG signal.
    fs : int
        The sampling frequency of the signal.
    q : float
        Tsallis entropy parameter.
    window_sec : float
        Sliding window size in seconds.
    step_sec : float
        Step size between windows in seconds.
        
    Returns
    -------
    float
        Aggregate Tsallis-ERSI value.
    """
    try:
        cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs)
    except Exception:
        cleaned_ecg = ecg_signal
        
    # Define window size and step in samples
    window_size = max(10, int(window_sec * fs))
    step_size = max(1, int(step_sec * fs))
    
    # Compute sliding window Tsallis entropy
    tsallis_vals = NonExtensiveEntropy.compute_custom_entropy_sliding(
        cleaned_ecg, entropy_func=NonExtensiveEntropy.tsallis, q=q, window=window_size, step=step_size, bins=10
    )
    
    if not tsallis_vals:
        warnings.warn(f"Signal length ({len(ecg_signal)}) is shorter than window size ({window_size}). Cannot compute sliding windows.")
        return np.nan
        
    # Backfill then fillna with 0 for safety before computation
    df_entropy = pd.DataFrame({"tsallis": tsallis_vals}).bfill().fillna(0)
    
    # Compute ERSI specifically for Tsallis
    df_ersi = ERSI.ERSI_computation(df_entropy, ["tsallis"])
    
    if "tsallis_ERSI" in df_ersi.columns:
        return df_ersi["tsallis_ERSI"].mean()
    return np.nan


def benchmark_ersi(healthy_signals, brugada_signals, fs=1000, window_sec=5.0, step_sec=2.5):
    """
    Given lists of healthy and brugada ECG signals,
    benchmarks standard entropies, Full ERSI, and Tsallis-ERSI 
    and returns p-values from Mann-Whitney U tests.
    
    Parameters
    ----------
    healthy_signals : list of array-like
        List of 1D arrays for healthy patient signals.
    brugada_signals : list of array-like
        List of 1D arrays for Brugada patient signals.
    fs : int
        Sampling frequency.
    window_sec : float
        Sliding window size in seconds.
    step_sec : float
        Step size between windows in seconds.
        
    Returns
    -------
    pd.Series
        Sorted p-values for each feature evaluating its discriminative power.
    """
    results_healthy = []
    print(f"Processing {len(healthy_signals)} healthy signals...")
    for i, sig in enumerate(healthy_signals):
        # Full pipeline
        res = process_patient(sig, fs=fs, window_sec=window_sec, step_sec=step_sec)
        # Tsallis ERSI
        res['tsallis_ERSI'] = process_patient_tsallis(sig, fs=fs, window_sec=window_sec, step_sec=step_sec)
        results_healthy.append(res)
        
    results_brugada = []
    print(f"Processing {len(brugada_signals)} Brugada signals...")
    for i, sig in enumerate(brugada_signals):
        # Full pipeline
        res = process_patient(sig, fs=fs, window_sec=window_sec, step_sec=step_sec)
        # Tsallis ERSI
        res['tsallis_ERSI'] = process_patient_tsallis(sig, fs=fs, window_sec=window_sec, step_sec=step_sec)
        results_brugada.append(res)

    df_h = pd.DataFrame(results_healthy)
    df_b = pd.DataFrame(results_brugada)
    
    p_values = {}
    for col in df_h.columns:
        x = df_h[col].dropna()
        y = df_b[col].dropna()
        
        # Mann-Whitney U test requires at least 1 observation in each array
        if len(x) > 0 and len(y) > 0:
            stat, p = mannwhitneyu(x, y, alternative='two-sided')
            p_values[col] = p
        else:
            p_values[col] = np.nan
            
    p_values_series = pd.Series(p_values).sort_values()
    print("\\nBenchmarking complete. P-values (lower is better):")
    print(p_values_series)
    return p_values_series
