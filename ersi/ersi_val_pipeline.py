"""
ERSI Validation Pipeline
=========================
Comprehensive validation of the ERSI pipeline against standard entropies
for distinguishing between patient groups (e.g., Brugada vs Control).
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import neurokit2 as nk
import warnings

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    warnings.warn("scikit-learn is required for AUC and ROC curve computation. Please install it.")

from ersi.ersi import ERSI
from ersi.entropy_measures import SimpleEntropy, NonExtensiveEntropy, entropy_funcs

warnings.filterwarnings('ignore')

class ERSIDataPrep:
    @staticmethod
    def clean_and_window_signal(ecg_signal, fs=1000, window_sec=5.0, step_sec=2.5):
        """Clean and segment ECG signal into 5s windows with 2.5s overlap."""
        try:
            cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs)
        except Exception:
            cleaned_ecg = ecg_signal
            
        window_size = max(10, int(window_sec * fs))
        step_size = max(1, int(step_sec * fs))
        
        windows = []
        for i in range(0, len(cleaned_ecg) - window_size + 1, step_size):
            windows.append(cleaned_ecg[i : i + window_size])
            
        return windows
        
    @staticmethod
    def compute_vcg_magnitude(lead_signals_df):
        """
        Combines signals from 12 leads into a VCG spatial magnitude sequence.
        Returns a 1D numpy array representing the VCG magnitude.
        """
        try:
            vcg = nk.ecg_vcg(lead_signals_df, method="kors")
            # Calculate the spatial magnitude at each time point
            vcg_mag = np.sqrt(vcg['VCG_x']**2 + vcg['VCG_y']**2 + vcg['VCG_z']**2)
            return vcg_mag.values
        except Exception as e:
            warnings.warn(f"Failed to compute VCG magnitude: {e}")
            return None

class ERSIPipelineValidator:
    def __init__(self, fs=1000, window_sec=5.0, step_sec=2.5):
        self.fs = fs
        self.window_sec = window_sec
        self.step_sec = step_sec
        
        # Pre-defined modes based on theory
        self.modes = {
            "Morphology": ["svd_entropy", "shannon", "sample_entropy"],
            "Anomaly": ["tsallis_q0.5", "renyi_a0.5"],
            "Regularity": ["sample_entropy", "app_entropy", "perm_entropy"],
            "Full": [
                "shannon", "sample_entropy", "app_entropy", "perm_entropy", 
                "spectral_entropy", "svd_entropy",
                "tsallis_q0.5", "tsallis_q1.5", "renyi_a0.5", "renyi_a2"
            ],
            "Tsallis solo": ["tsallis_q0.5", "tsallis_q1.5"],
            "Renyi solo": ["renyi_a0.5", "renyi_a2"]
        }
        
    def _compute_patient_entropies(self, ecg_signal):
        """
        Compute all standard and non-extensive entropies for all windows.
        Returns a DataFrame of windows x entropies.
        """
        windows = ERSIDataPrep.clean_and_window_signal(
            ecg_signal, fs=self.fs, window_sec=self.window_sec, step_sec=self.step_sec
        )
        
        if len(windows) == 0:
            return pd.DataFrame()
            
        results = []
        for w in windows:
            row = {}
            # Standard entropies
            for name, func in entropy_funcs.items():
                try:
                    row[name] = func(w)
                except Exception:
                    row[name] = np.nan
                    
            # Non-extensive entropies
            try:
                prob = NonExtensiveEntropy.compute_probabilities(w, bins=10)
                row["tsallis_q0.5"] = NonExtensiveEntropy.tsallis(prob, q=0.5)
                row["tsallis_q1.5"] = NonExtensiveEntropy.tsallis(prob, q=1.5)
                row["renyi_a0.5"] = NonExtensiveEntropy.renyi(prob, q=0.5)
                row["renyi_a2"] = NonExtensiveEntropy.renyi(prob, q=2)
            except Exception:
                row["tsallis_q0.5"] = np.nan
                row["tsallis_q1.5"] = np.nan
                row["renyi_a0.5"] = np.nan
                row["renyi_a2"] = np.nan
                
            results.append(row)
            
        return pd.DataFrame(results)

    def extract_features(self, X):
        """
        Extract entropy features for a list of signals.
        Returns a list of DataFrames (one per patient).
        """
        patient_dfs = []
        for sig in X:
            df = self._compute_patient_entropies(sig)
            patient_dfs.append(df)
        return patient_dfs

    def feature_selection(self, patient_dfs_train, y_train, top_k=3):
        """
        Feature Selection on Training Set.
        Computes per-patient mean and selects top_k entropies based on Mann-Whitney U test.
        """
        # Aggregate per patient
        X_train_means = [df.mean().to_dict() for df in patient_dfs_train if not df.empty]
        df_train = pd.DataFrame(X_train_means)
        
        y_train_filtered = [y for y, df in zip(y_train, patient_dfs_train) if not df.empty]
        
        mask_brugada = np.array(y_train_filtered) == 1
        mask_control = np.array(y_train_filtered) == 0
        
        p_values = {}
        for col in df_train.columns:
            group1 = df_train.loc[mask_brugada, col].dropna()
            group2 = df_train.loc[mask_control, col].dropna()
            if len(group1) > 0 and len(group2) > 0:
                stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
                p_values[col] = p
            else:
                p_values[col] = np.nan
                
        p_values_series = pd.Series(p_values).dropna().sort_values()
        
        print("\nFeature Selection (Training Set) P-values:")
        print(p_values_series)
        
        top_features = p_values_series.head(top_k).index.tolist()
        print(f"\nSelected Top {top_k} Features: {top_features}")
        return top_features

    def compute_ersi_modes(self, patient_dfs, selected_features):
        """
        Compute ERSI for each mode per patient.
        Aggregate each to a single number per patient (mean over windows).
        """
        results = []
        
        for df in patient_dfs:
            if df.empty:
                continue
                
            # Keep a copy of columns to avoid modifying the loop collection inside
            cols = list(df.columns)
            
            # Forward and backward fill for any NaNs
            df = df.bfill().fillna(0)
            
            patient_res = {}
            
            # Helper to run ERSI
            def _run_ersi(feats, label, use_full=True):
                try:
                    valid_feats = [f for f in feats if f in df.columns]
                    if len(valid_feats) == 0:
                        patient_res[label] = np.nan
                        return
                    if use_full:
                        d = ERSI.ERSI_full(df, valid_feats)
                        patient_res[label] = d["ERSI_full"].mean()
                    else:
                        d = ERSI.ERSI_computation(df[valid_feats], valid_feats)
                        patient_res[label] = d[f"{valid_feats[0]}_ERSI"].mean()
                except Exception:
                    patient_res[label] = np.nan

            # 1. ERSI Selected
            _run_ersi(selected_features, "ERSI_selected", use_full=True)
                
            # 2. ERSI Morphology
            _run_ersi(self.modes["Morphology"], "ERSI_morphology", use_full=True)

            # 3. ERSI Anomaly
            _run_ersi(self.modes["Anomaly"], "ERSI_anomaly", use_full=True)

            # 4. ERSI Regularity
            _run_ersi(self.modes["Regularity"], "ERSI_regularity", use_full=True)

            # 5. ERSI Full
            _run_ersi(self.modes["Full"], "ERSI_full", use_full=True)
                
            # 6. Solo ERSIs (Time-ranking only)
            solo_modes = [
                ("Tsallis_ERSI_q0.5", "tsallis_q0.5"),
                ("Tsallis_ERSI_q1.5", "tsallis_q1.5"),
                ("Renyi_ERSI_a0.5", "renyi_a0.5"),
                ("Renyi_ERSI_a2", "renyi_a2")
            ]
            for label, f in solo_modes:
                _run_ersi([f], label, use_full=False)
                    
            # 7. Add raw individual entropies for comparison
            for col in cols:
                patient_res[f"Raw_{col}"] = df[col].mean()

            results.append(patient_res)
            
        return pd.DataFrame(results)

    def evaluate(self, df_results, y_test, plot=False, title_suffix=""):
        """
        Evaluate on Test Set:
        Mann-Whitney p-value, AUC, Optimal threshold (Sens/Spec).
        Returns a summary DataFrame table.
        """
        mask_brugada = np.array(y_test) == 1
        mask_control = np.array(y_test) == 0
        
        evaluation = []
        roc_data = {}
        
        for col in df_results.columns:
            group1 = df_results.loc[mask_brugada, col].dropna()
            group2 = df_results.loc[mask_control, col].dropna()
            
            p_val = np.nan
            if len(group1) > 0 and len(group2) > 0:
                _, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                
            # AUC & Youden's
            auc = np.nan
            sens = np.nan
            spec = np.nan
            
            valid_idx = df_results[col].dropna().index
            if len(valid_idx) > 1:
                y_true = np.array(y_test)[valid_idx]
                y_scores = df_results.loc[valid_idx, col].values
                
                try:
                    # Depending on the direction of the score, AUC might be < 0.5
                    # We compute AUC directly. If < 0.5, it means the correlation is negative
                    # so we invert the scores.
                    auc = roc_auc_score(y_true, y_scores)
                    if auc < 0.5:
                        y_scores = -y_scores
                        auc = roc_auc_score(y_true, y_scores)
                        
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    # Youden's J statistic = TPR + TNR - 1 = TPR - FPR
                    J = tpr - fpr
                    best_idx = np.argmax(J)
                    sens = tpr[best_idx]
                    spec = 1.0 - fpr[best_idx]
                    
                    if plot:
                        roc_data[col] = (fpr, tpr, auc)
                except Exception:
                    pass
                    
            evaluation.append({
                "Method": col,
                "p-value": p_val,
                "AUC": auc,
                "Sensitivity": sens,
                "Specificity": spec
            })
            
        df_eval = pd.DataFrame(evaluation)
        # Sort by AUC descending
        df_eval = df_eval.sort_values(by="AUC", ascending=False).reset_index(drop=True)
        
        if plot and roc_data:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 8))
                
                # Plot only the top 5 to avoid clutter, plus ERSI variants
                top_methods = df_eval["Method"].head(5).tolist()
                plot_methods = list(set(top_methods + [m for m in df_eval["Method"] if "ERSI" in m]))
                
                for method in plot_methods:
                    if method in roc_data:
                        fpr, tpr, a = roc_data[method]
                        # Bold the ERSI methods
                        if "ERSI" in method:
                            plt.plot(fpr, tpr, label=f"{method} (AUC = {a:.3f})", linewidth=2.5)
                        else:
                            plt.plot(fpr, tpr, label=f"{method} (AUC = {a:.3f})", linewidth=1.5, linestyle="--")
                            
                plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curves - Entropy Measures Validation {title_suffix}")
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.show()
            except ImportError:
                warnings.warn("matplotlib is required to plot ROC curves.")
                
        return df_eval

    def simulate_noise_evaluation(self, X_test, y_test, selected_features, noise_level=0.1):
        """
        Add Gaussian noise to test set signals, recompute ERSI, see if AUC drops.
        """
        X_test_noisy = []
        for sig in X_test:
            noise = np.random.normal(0, noise_level * np.std(sig), len(sig))
            X_test_noisy.append(sig + noise)
            
        # Re-run extraction and computation
        patient_dfs_noisy = self.extract_features(X_test_noisy)
        df_results_noisy = self.compute_ersi_modes(patient_dfs_noisy, selected_features)
        df_eval_noisy = self.evaluate(df_results_noisy, y_test)
        return df_eval_noisy
