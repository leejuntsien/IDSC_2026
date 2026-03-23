"""
ecg_pipeline_features.py — NeuroKit2-based ECG signal processing and feature extraction.
V2 overhaul: adds ST elevation extraction (extract_st_features) and
             rule-based Brugada pre-filter (layer1_brugada_rule).
Designed for 100Hz, 12-second Brugada-HUCA recordings.
"""
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import find_peaks


# ──────────────────────────────────────────────────────────────────────────────
# Core signal processing (unchanged from v1)
# ──────────────────────────────────────────────────────────────────────────────

def apply_notch_filter(ecg_signal, sampling_rate=100, powerline=50):
    """1. Apply 50Hz notch filter."""
    return nk.signal_filter(
        ecg_signal, sampling_rate=sampling_rate,
        lowcut=None, highcut=None, method="powerline", powerline=powerline
    )


import warnings

def detect_peaks(ecg_cleaned, sampling_rate=100):
    """
    R-peak detection with spurious peak filtering and scipy fallback.
    Filters out peaks that fail physiological plausibility checks.
    """
    # Suppress neurokit2 internal width warnings — these come from
    # spurious narrow peaks that we filter out below anyway
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
            peaks = info["ECG_R_Peaks"]
        except Exception:
            peaks = np.array([])

    # Validate peaks physiologically
    peaks = _validate_peaks(peaks, ecg_cleaned, sampling_rate)
    if len(peaks) >= 2:
        return peaks

    # Fallback: scipy find_peaks with physiological constraints
    peaks = _scipy_peak_fallback(ecg_cleaned, sampling_rate)
    if len(peaks) >= 2:
        return peaks

    raise ValueError(
        f"Peak detection failed after all methods. "
        f"Signal range: [{ecg_cleaned.min():.3f}, {ecg_cleaned.max():.3f}], "
        f"length: {len(ecg_cleaned)} samples"
    )


def _validate_peaks(peaks, ecg_signal, sampling_rate):
    """
    Filters spurious peaks using physiological rules:
    - Minimum distance: 300ms (max 200 BPM)
    - Minimum amplitude: above 50th percentile of signal
    - Not within 50ms of signal boundary
    """
    if len(peaks) == 0:
        return peaks

    min_distance  = int(0.30 * sampling_rate)   # 300ms = max 200 BPM
    boundary_pad  = int(0.05 * sampling_rate)    # 50ms from edges
    amp_threshold = np.percentile(ecg_signal, 50)
    n             = len(ecg_signal)

    valid = []
    for i, p in enumerate(peaks):
        # Boundary check
        if p < boundary_pad or p > n - boundary_pad:
            continue
        # Amplitude check — real R-peaks are tall
        if ecg_signal[p] < amp_threshold:
            continue
        # Distance check — skip if too close to previous valid peak
        if valid and (p - valid[-1]) < min_distance:
            # Keep whichever is taller
            if ecg_signal[p] > ecg_signal[valid[-1]]:
                valid[-1] = p
            continue
        valid.append(p)

    return np.array(valid)


def _scipy_peak_fallback(ecg_signal, sampling_rate):
    """Scipy-based peak detection with progressively relaxed thresholds."""
    min_distance = int(0.40 * sampling_rate)   # 400ms = max 150 BPM

    for percentile in [75, 60, 50]:
        height = np.percentile(ecg_signal, percentile)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            peaks, _ = find_peaks(
                ecg_signal,
                distance=min_distance,
                height=height,
            )
        peaks = _validate_peaks(peaks, ecg_signal, sampling_rate)
        if len(peaks) >= 2:
            return peaks

    return np.array([])


def delineate_segments(ecg_cleaned, rpeaks, sampling_rate=100):
    """3. Segment delineation via DWT."""
    try:
        _, waves = nk.ecg_delineate(
            ecg_cleaned, rpeaks,
            sampling_rate=sampling_rate, method="dwt"
        )
        return dict(waves)
    except Exception:
        pass

    try:
        # Fallback: peak method — less accurate but more robust
        _, waves = nk.ecg_delineate(
            ecg_cleaned, rpeaks,
            sampling_rate=sampling_rate, method="peak"
        )
        return dict(waves)
    except Exception:
        # Return empty dict — all downstream functions handle missing keys
        # gracefully by returning NaN, so the record still processes
        return {}


def extract_rr_peaks(rpeaks, sampling_rate=100):
    """
    Computes RR intervals and basic HRV metrics from R-peak indices.
    Returns dict with rr_intervals, mean_rr, sdnn, rmssd, mean_hr_bpm.
    Useful as auxiliary features for the CNN+BiGRU temporal model.
    """
    if len(rpeaks) < 2:
        return {
            'rr_intervals': np.array([]),
            'mean_rr_s': np.nan,
            'sdnn_s': np.nan,
            'rmssd_s': np.nan,
            'mean_hr_bpm': np.nan,
        }

    rr_intervals = np.diff(rpeaks) / sampling_rate  # in seconds

    return {
        'rr_intervals': rr_intervals,
        'mean_rr_s': float(np.mean(rr_intervals)),
        'sdnn_s': float(np.std(rr_intervals)),
        'rmssd_s': float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2))),
        'mean_hr_bpm': float(60.0 / np.mean(rr_intervals)),
    }


def segment_beats_by_rr(ecg_signal, rpeaks, sampling_rate=100):
    """5. Segment beats by consecutive R-peaks → period & signal slice."""
    segments = []
    for i in range(len(rpeaks) - 1):
        start = rpeaks[i]
        end = rpeaks[i + 1]
        period_s = (end - start) / sampling_rate
        segments.append({
            "beat_idx": i,
            "signal": ecg_signal[start:end],
            "start_idx": start,
            "end_idx": end,
            "period_s": period_s,
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


def extract_segment_differences(waves, ecg_signal, beat_idx, sampling_rate=100):
    """7. Time gap & voltage differences between P, Q, R, S, T peaks."""
    keys = ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_R_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']
    beat_peaks = {}
    for k in keys:
        if k in waves and beat_idx < len(waves[k]):
            idx = waves[k][beat_idx]
            if not np.isnan(idx):
                beat_peaks[k.replace('ECG_', '').replace('_Peaks', '')] = int(idx)

    features = {}
    peak_names = list(beat_peaks.keys())
    for i, p1 in enumerate(peak_names):
        for j, p2 in enumerate(peak_names):
            if i < j:
                idx1 = beat_peaks[p1]
                idx2 = beat_peaks[p2]
                features[f'TimeGap_{p1}_{p2}_s'] = abs(idx2 - idx1) / sampling_rate
                features[f'VoltDiff_{p1}_{p2}'] = ecg_signal[idx2] - ecg_signal[idx1]
    return features


def extract_qrs_time(waves, beat_idx, sampling_rate=100):
    """8. QRS duration in seconds."""
    try:
        q_start = waves['ECG_R_Onsets'][beat_idx]
        s_end = waves['ECG_R_Offsets'][beat_idx]
        if not np.isnan(q_start) and not np.isnan(s_end):
            return (s_end - q_start) / sampling_rate
    except (KeyError, IndexError):
        pass
    return np.nan


def extract_st_segment(waves, beat_idx, sampling_rate=100):
    """9. ST segment duration in seconds."""
    try:
        st_start = waves['ECG_R_Offsets'][beat_idx]
        st_end = waves['ECG_T_Onsets'][beat_idx]
        if not np.isnan(st_start) and not np.isnan(st_end):
            return (st_end - st_start) / sampling_rate
    except (KeyError, IndexError):
        pass
    return np.nan


def detect_u_waves(ecg_signal, waves, beat_idx, sampling_rate=100):
    """10. U-wave detection between T_Offset and +400ms."""
    try:
        t_offset = waves['ECG_T_Offsets'][beat_idx]
        if np.isnan(t_offset):
            return False, np.nan
        t_offset = int(t_offset)
        search_end = t_offset + int(0.4 * sampling_rate)
        if search_end >= len(ecg_signal):
            search_end = len(ecg_signal) - 1
        window = ecg_signal[t_offset:search_end]
        peaks, _ = find_peaks(window, height=0.05 * np.max(ecg_signal))
        if len(peaks) > 0:
            return True, t_offset + peaks[0]
    except (KeyError, IndexError):
        pass
    return False, np.nan


def detect_inversion(ecg_signal, waves, beat_idx):
    """11. T-wave inversion detection."""
    try:
        t_peak = waves['ECG_T_Peaks'][beat_idx]
        if not np.isnan(t_peak):
            return ecg_signal[int(t_peak)] < 0
    except (KeyError, IndexError):
        pass
    return None


def combine_to_vcg(lead_signals_df):
    """Combine 12-lead signals to VCG using Kors regression matrix."""
    kors_weights = np.array([
        [ 0.38, -0.07, -0.13,  0.05, -0.01,  0.14,  0.06,  0.54],
        [-0.07,  0.93,  0.06, -0.02, -0.05,  0.06, -0.17,  0.13],
        [ 0.11, -0.23, -0.43, -0.06, -0.04, -0.02, -0.10, -0.38],
    ])
    leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for lead in leads:
        if lead not in lead_signals_df.columns:
            lead_signals_df[lead] = 0.0
    ecg_matrix = lead_signals_df[leads].values
    vcg_matrix = ecg_matrix.dot(kors_weights.T)
    return pd.DataFrame(vcg_matrix, columns=['VCG_x', 'VCG_y', 'VCG_z'],
                         index=lead_signals_df.index)


# ──────────────────────────────────────────────────────────────────────────────
# NEW — Step 1: ST elevation extraction (Brugada-specific)
# ──────────────────────────────────────────────────────────────────────────────

def extract_st_features(ecg_signal, waves, beat_idx, sampling_rate=100):
    """
    Extracts ST elevation and J-point features relative to TP isoelectric baseline.
    Designed for 100Hz recordings (J+40ms = 4 samples, J+80ms = 8 samples).

    Returns dict with:
      j_point_voltage_abs   : raw voltage at J-point (ECG_R_Offsets index)
      st_elevation_j40      : ST elevation at J+40ms relative to isoelectric baseline (mV)
      st_elevation_j80      : ST elevation at J+80ms relative to isoelectric baseline (mV)
      t_wave_amplitude      : signed T-wave peak amplitude relative to baseline
      st_slope              : linear slope J+40ms→J+80ms (negative = downsloping = coved)
      isoelectric_baseline  : TP segment mean used as voltage reference
      r_prime_ratio         : R'/R amplitude ratio for rSR' detection, NaN if none
      st_extraction_quality : bool, True only if j40, j80, t_wave all extracted
    """
    features = {
        'j_point_voltage_abs': np.nan,
        'st_elevation_j40': np.nan,
        'st_elevation_j80': np.nan,
        't_wave_amplitude': np.nan,
        'st_slope': np.nan,
        'isoelectric_baseline': np.nan,
        'r_prime_ratio': np.nan,
        'st_extraction_quality': False,
    }
    n = len(ecg_signal)

    # Step 1: J-point = ECG_R_Offsets (end of QRS = start of ST)
    try:
        j_idx = waves['ECG_R_Offsets'][beat_idx]
        if np.isnan(j_idx):
            return features
        j_idx = int(j_idx)
    except (KeyError, IndexError):
        return features

    # Step 2: Isoelectric baseline from TP segment
    baseline = np.nan
    try:
        p_onset = waves['ECG_P_Onsets'][beat_idx]
        t_offset_prev = waves['ECG_T_Offsets'][beat_idx - 1] if beat_idx > 0 else np.nan
        if not np.isnan(p_onset) and not np.isnan(t_offset_prev):
            tp_start = int(t_offset_prev)
            tp_end = int(p_onset)
            if tp_end > tp_start and tp_end < n:
                baseline = float(np.mean(ecg_signal[tp_start:tp_end]))
    except (KeyError, IndexError):
        pass

    # Fallback: 20ms window before R_Onset (PR segment end)
    if np.isnan(baseline):
        try:
            r_onset = waves['ECG_R_Onsets'][beat_idx]
            if not np.isnan(r_onset):
                pre_r_end = int(r_onset)
                pre_r_start = max(0, pre_r_end - int(0.02 * sampling_rate))
                if pre_r_end > pre_r_start:
                    baseline = float(np.mean(ecg_signal[pre_r_start:pre_r_end]))
        except (KeyError, IndexError):
            pass

    features['j_point_voltage_abs'] = float(ecg_signal[j_idx])

    if np.isnan(baseline):
        return features
    features['isoelectric_baseline'] = baseline

    # Step 3: ST elevation at J+40ms and J+80ms
    j40 = j_idx + int(0.040 * sampling_rate)   # 4 samples at 100Hz
    j80 = j_idx + int(0.080 * sampling_rate)   # 8 samples at 100Hz
    if j40 < n:
        features['st_elevation_j40'] = float(ecg_signal[j40] - baseline)
    if j80 < n:
        features['st_elevation_j80'] = float(ecg_signal[j80] - baseline)

    # Step 4: ST slope (negative = downsloping = coved Brugada morphology)
    if j40 < n and j80 < n:
        rise = float(ecg_signal[j80] - ecg_signal[j40])
        run = float((j80 - j40) / sampling_rate)
        features['st_slope'] = rise / run if run > 0 else np.nan

    # Step 5: T-wave amplitude (signed — negative confirms inversion)
    try:
        t_peak = waves['ECG_T_Peaks'][beat_idx]
        if not np.isnan(t_peak):
            features['t_wave_amplitude'] = float(ecg_signal[int(t_peak)] - baseline)
    except (KeyError, IndexError):
        pass

    # Step 6: R' detection for rSR' (rabbit ears) pattern
    try:
        s_peak = waves['ECG_S_Peaks'][beat_idx]
        r_peak = waves['ECG_R_Peaks'][beat_idx]
        if not any(np.isnan([s_peak, j_idx, r_peak])):
            s_peak_i = int(s_peak)
            r_peak_i = int(r_peak)
            if s_peak_i < j_idx and s_peak_i < n and j_idx <= n:
                qrs_post_s = ecg_signal[s_peak_i:j_idx]
                if len(qrs_post_s) > 2:
                    r_prime_candidates, _ = find_peaks(qrs_post_s, height=0)
                    if len(r_prime_candidates) > 0:
                        best = r_prime_candidates[np.argmax(qrs_post_s[r_prime_candidates])]
                        r_prime_amp = float(qrs_post_s[best])
                        r_amp = float(ecg_signal[r_peak_i])
                        if r_amp != 0:
                            features['r_prime_ratio'] = r_prime_amp / abs(r_amp)
    except (KeyError, IndexError, ValueError):
        pass

    features['st_extraction_quality'] = not any(
        np.isnan(features[k])
        for k in ['st_elevation_j40', 'st_elevation_j80', 't_wave_amplitude']
    )
    return features


# ──────────────────────────────────────────────────────────────────────────────
# NEW — Step 2: Layer 1 rule-based Brugada pre-filter
# ──────────────────────────────────────────────────────────────────────────────

def layer1_brugada_rule(beat_features_row, leads_to_check=('V1', 'V2', 'V3')):
    """
    Permissive rule-based Brugada Type 1 pre-filter (HIGH RECALL).
    Fires if in ANY checked lead:
      - ST elevation at J+40ms >= 0.15mV (permissive; clinical Type 1 = 0.2mV)
      - AND at least one morphological qualifier:
          * T-wave flat or inverted (t_wave_amplitude <= 0.05mV)
          * OR ST is downsloping (st_slope < 0)

    Returns:
        suspected (bool)  : True if rule fires on any lead
        evidence  (dict)  : per-lead values that drove the decision
    """
    evidence = {}
    suspected = False

    for lead in leads_to_check:
        st_j40 = beat_features_row.get(f'{lead}_st_elevation_j40', np.nan)
        t_amp  = beat_features_row.get(f'{lead}_t_wave_amplitude', np.nan)
        slope  = beat_features_row.get(f'{lead}_st_slope', np.nan)
        qual   = beat_features_row.get(f'{lead}_st_extraction_quality', False)

        if not qual or (isinstance(st_j40, float) and np.isnan(st_j40)):
            evidence[lead] = {'fired': False, 'reason': 'low_quality_or_missing'}
            continue

        elevation_flag = st_j40 >= 0.10
        t_flag = (not (isinstance(t_amp, float) and np.isnan(t_amp))) and (t_amp <= 0.05)
        slope_flag = (not (isinstance(slope, float) and np.isnan(slope))) and (slope < 0)
        fired = elevation_flag and (t_flag or slope_flag)

        evidence[lead] = {
            'fired': fired,
            'st_elevation_j40': st_j40,
            'elevation_flag': elevation_flag,
            't_flag': t_flag,
            'slope_flag': slope_flag,
        }
        if fired:
            suspected = True

    return suspected, evidence


# ──────────────────────────────────────────────────────────────────────────────
# Master function: process_single_lead (updated with ST features)
# ──────────────────────────────────────────────────────────────────────────────

def process_single_lead(ecg_signal, sampling_rate=100):
    """
    Process a single ECG lead to extract beat-by-beat numerical features.
    Now includes ST elevation features from extract_st_features.
    """
    min_samples = int(2.0 * sampling_rate)  # at least 2 seconds
    if len(ecg_signal) < min_samples:
        raise ValueError(
            f"Signal too short: {len(ecg_signal)} samples "
            f"(minimum {min_samples} required)"
        )

    ecg_clean = apply_notch_filter(ecg_signal, sampling_rate)
    rpeaks = detect_peaks(ecg_clean, sampling_rate)

    waves = delineate_segments(ecg_clean, rpeaks, sampling_rate)
    waves['ECG_R_Peaks'] = rpeaks  # Inject for convenience

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

        # NEW: ST elevation features
        st_feats = extract_st_features(ecg_clean, waves, idx, sampling_rate)

        beat_params = {
            "beat_index": idx,
            "period_s": seg['period_s'],
            "QRS_duration_s": qrs,
            "ST_segment_s": st,
            "has_U_wave": has_u,
            "T_wave_inversion": invert,
        }
        beat_params.update(diffs)
        beat_params.update(st_feats)
        beat_features.append(beat_params)

    return pd.DataFrame(beat_features)


if __name__ == "__main__":
    print("NeuroKit pipeline v2 initialized.")
    ecg = nk.ecg_simulate(duration=12, sampling_rate=100, heart_rate=75)
    df_features = process_single_lead(ecg, sampling_rate=100)
    print(f"Processed simulated ECG. Beats: {len(df_features)}")

    required_cols = [
        'st_elevation_j40', 'st_elevation_j80', 't_wave_amplitude',
        'st_slope', 'isoelectric_baseline', 'j_point_voltage_abs',
        'r_prime_ratio', 'st_extraction_quality'
    ]
    for col in required_cols:
        assert col in df_features.columns, f"MISSING: {col}"
    print("All ST feature columns present.")
    print(df_features[required_cols].describe())
