"""
ersi_detector.py — Adapter wrapping ERSIPipelineValidator for Brugada-HUCA.

ERSI = Entropy-Ranked Stability Index.
Sliding window entropies (Shannon, Tsallis, Rényi, sample, spectral, SVD)
→ ranked by time position (1/rank weighting)
→ ERSI_full: dual-ranked (time × cross-entropy multiplicative fusion)

At 100Hz / 12s recordings:
  window_sec=2.0 → 200 samples (~2 beat lengths at 60 BPM)
  step_sec=1.0   → 100 samples step → ~10 windows per recording
"""
import numpy as np
import pandas as pd
import pickle
import os
import warnings

from ml_pipeline.ersi_pipeline import process_patient, process_patient_tsallis
from ml_pipeline.ersi_val_pipeline import ERSIPipelineValidator, ERSIDataPrep
from ml_pipeline.ersi import ERSI as ERSIClass
from ml_pipeline.entropy_measures import entropy_funcs

warnings.filterwarnings('ignore')

WINDOW_SEC = 2.0
STEP_SEC   = 1.0


class BrugadaERSIDetector:
    """
    Fits ERSI_full thresholds on Normal training patients.
    At inference, scores a single patient's raw V1 ECG signal.

    Detection logic:
      - Compute ERSI_full score (dual-ranked) per patient
      - Patient flagged if score exceeds 95th percentile of Normal patient scores
      - Returns per-window timeline for Streamlit visualisation
    """

    def __init__(self, fs=100, window_sec=WINDOW_SEC, step_sec=STEP_SEC,
                 target_percentile=95):
        self.fs                = fs
        self.window_sec        = window_sec
        self.step_sec          = step_sec
        self.target_percentile = target_percentile
        self.validator         = ERSIPipelineValidator(
            fs=fs, window_sec=window_sec, step_sec=step_sec
        )
        self.selected_features_ = None
        self.threshold_full_    = None
        self.fitted_            = False

    def fit(self, signals_train, y_train):
        """
        Fit ERSI_full threshold on training data.

        Args:
            signals_train : list of 1D np.arrays — one raw V1 signal per patient
            y_train       : list/array of int labels (0=Normal, 1=Brugada)
        """
        y_train = np.array(y_train)
        print(f"[ERSI] Extracting features for {len(signals_train)} training patients...")

        patient_dfs = self.validator.extract_features(signals_train)

        # Mann-Whitney feature selection on training set only
        self.selected_features_ = self.validator.feature_selection(
            patient_dfs, y_train.tolist(), top_k=3
        )

        # Compute all ERSI modes per patient
        df_scores = self.validator.compute_ersi_modes(patient_dfs, self.selected_features_)

        # Set threshold from Normal patient ERSI_full scores only
        normal_idx = np.where(y_train == 0)[0]
        valid_count = len(df_scores)
        normal_idx  = normal_idx[normal_idx < valid_count]

        if 'ERSI_full' in df_scores.columns:
            normal_full          = df_scores.iloc[normal_idx]['ERSI_full'].dropna()
            self.threshold_full_ = float(np.percentile(normal_full, self.target_percentile))
            print(f"[ERSI] ERSI_full threshold ({self.target_percentile}th pct "
                  f"of {len(normal_full)} Normal patients): {self.threshold_full_:.6f}")
        else:
            raise ValueError("ERSI_full column not found in scores. "
                             "Check that ersi.py ERSI_full method is working correctly.")

        self.fitted_ = True
        return self

    def score_patient(self, ecg_signal_v1):
        """
        Score a single patient's V1 signal.

        Returns:
            ersi_full  : float — ERSI_full aggregate score (primary detection signal)
            is_brugada : bool
            evidence   : dict — full diagnostic detail for Streamlit display
        """
        if not self.fitted_:
            raise RuntimeError("BrugadaERSIDetector must be fitted before scoring.")

        # Full process_patient gives all entropy means + ERSI variants
        results = process_patient(
            ecg_signal_v1, fs=self.fs,
            window_sec=self.window_sec, step_sec=self.step_sec
        )

        ersi_full       = results.get('ERSI_full', np.nan)
        ersi_timeseries = results.get('ERSI_timeseries', np.nan)

        is_brugada = (
            not np.isnan(ersi_full) and
            self.threshold_full_ is not None and
            ersi_full > self.threshold_full_
        )

        # Compute per-window ERSI_full timeline for Streamlit bar chart
        windows = ERSIDataPrep.clean_and_window_signal(
            ecg_signal_v1, fs=self.fs,
            window_sec=self.window_sec, step_sec=self.step_sec
        )

        window_scores = []
        for w in windows:
            row = {}
            for name, func in entropy_funcs.items():
                try:
                    row[name] = func(w)
                except Exception:
                    row[name] = np.nan
            window_scores.append(row)

        df_windows = pd.DataFrame(window_scores).bfill().fillna(0)
        window_ersi_timeline  = []
        window_ersi_timeseries = []

        if not df_windows.empty and self.selected_features_:
            valid_feats = [f for f in self.selected_features_ if f in df_windows.columns]
            if valid_feats:
                try:
                    # ERSI_full per window (dual-ranked — primary decision)
                    df_full = ERSIClass.ERSI_full(df_windows, valid_feats)
                    window_ersi_timeline = df_full['ERSI_full'].tolist()

                    # ERSI_timeseries per window (additive — for visual context)
                    df_ts = ERSIClass.ERSI_timeseries(df_windows, valid_feats)
                    if 'ERSI_timeseries' in df_ts.columns:
                        window_ersi_timeseries = df_ts['ERSI_timeseries'].tolist()
                except Exception as e:
                    warnings.warn(f"[ERSI] Window timeline computation failed: {e}")

        evidence = {
            'ersi_full':               ersi_full,
            'ersi_timeseries':         ersi_timeseries,
            'threshold_full':          self.threshold_full_,
            'selected_features':       self.selected_features_,
            'window_ersi_timeline':    window_ersi_timeline,    # ERSI_full per window
            'window_ersi_timeseries':  window_ersi_timeseries,  # additive fusion per window
            'n_windows':               len(windows),
            'variant_used':            'ERSI_full',
            'window_sec':              self.window_sec,
            'step_sec':                self.step_sec,
            'all_entropy_scores':      results,
        }

        return ersi_full, is_brugada, evidence

    def evaluate(self, signals_test, y_test):
        """Evaluate on test set. Returns summary DataFrame from ERSIPipelineValidator."""
        patient_dfs = self.validator.extract_features(signals_test)
        df_scores   = self.validator.compute_ersi_modes(patient_dfs, self.selected_features_)
        return self.validator.evaluate(df_scores, y_test)

    def save(self, path='models/ersi_detector.pkl'):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[ERSI] Saved to {path}")

    @classmethod
    def load(cls, path='models/ersi_detector.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
