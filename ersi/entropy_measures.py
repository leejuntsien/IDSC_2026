"""
Entropy Measures
=================
Non-extensive entropy functions (Tsallis, Rényi) and Shannon entropy with
sliding window support. Includes both custom and antropy-based implementations.

Source: ersi_helper.py
"""
import numpy as np
import pandas as pd


class NonExtensiveEntropy:
    """
    Non-extensive entropy functions consisting of:
    - probability computation using histogram method
    - Tsallis and Rényi entropy computation
    - sliding window integration
    """

    @staticmethod
    def tsallis(prob, q=2):
        """Compute Tsallis entropy."""
        prob = prob[prob > 0]
        return (1 - np.sum(prob**q)) / (q - 1) if q != 1 else -np.sum(prob * np.log(prob))

    @staticmethod
    def renyi(prob, q=2):
        """Compute Rényi entropy."""
        prob = prob[prob > 0]
        return np.log(np.sum(prob**q)) / (1 - q) if q != 1 else -np.sum(prob * np.log(prob))

    @staticmethod
    def compute_probabilities(window, bins=10):
        """Compute probability distribution from a data window using histogram."""
        counts, _ = np.histogram(window, bins=bins)
        prob = counts / counts.sum()
        return prob

    @staticmethod
    def compute_custom_entropy_sliding(series, entropy_func, q=2, window=30, step=1, bins=10):
        """
        Compute entropy over sliding windows using a custom entropy function.

        Parameters
        ----------
        series : pd.Series or np.array
            Input time series.
        entropy_func : callable
            Entropy function (e.g., NonExtensiveEntropy.tsallis).
        q : float
            Entropy order parameter.
        window : int
            Window size.
        step : int
            Step size between windows.
        bins : int
            Number of histogram bins.

        Returns
        -------
        list
            List of entropy values for each window.
        """
        results = []
        values = np.array(series)
        for i in range(0, len(values) - window + 1, step):
            w = values[i : i + window]
            prob = NonExtensiveEntropy.compute_probabilities(w, bins)
            results.append(entropy_func(prob, q))
        return results


class SimpleEntropy:
    """Simple entropy computation with Shannon entropy and antropy integration."""

    @staticmethod
    def compute_entropy_1(series):
        """Custom Shannon entropy computation."""
        series = np.array(series)
        series = series[~np.isnan(series)]
        if len(series) == 0:
            return np.nan
        counts = np.histogram(series, bins=10)[0]
        prob = counts / counts.sum()
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))

    @staticmethod
    def compute_entropy(entropy_func, series):
        """Wrapper for antropy functions."""
        series = np.array(series)
        series = series[~np.isnan(series)]
        return entropy_func(series) if len(series) > 0 else np.nan

    @staticmethod
    def sliding_window_entropy(series, funcs, window=30, step=1):
        """
        Compute multiple entropy measures over sliding windows.

        Parameters
        ----------
        series : pd.Series or np.array
            Input data.
        funcs : dict
            Dictionary of {name: entropy_function}.
        window : int
            Window size.
        step : int
            Step size.

        Returns
        -------
        pd.DataFrame
            DataFrame with entropy values for each window position.
        """
        results = []
        indices = []
        values = np.array(series)

        for i in range(0, len(values) - window + 1, step):
            w = values[i : i + window]
            row = {}
            for name, func in funcs.items():
                try:
                    row[name] = func(w)
                except Exception:
                    row[name] = np.nan
            results.append(row)
            indices.append(i)

        return pd.DataFrame(results, index=indices)

    @staticmethod
    def impute_entropy_by_task(df, task_col="Task", method="mean"):
        """
        Impute missing entropy values by task group.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        task_col : str
            Column to group by.
        method : str
            Imputation method ('mean' or 'median').

        Returns
        -------
        pd.DataFrame
            DataFrame with imputed values.
        """
        df = df.copy()
        entropy_cols = [
            col for col in df.columns
            if col != task_col and df[col].dtype in [float, int, np.float64, np.int64]
        ]
        df[entropy_cols] = df[entropy_cols].replace([np.inf, -np.inf], np.nan)

        for task, group in df.groupby(task_col):
            mask = df[task_col] == task
            if method == "mean":
                df.loc[mask, entropy_cols] = group[entropy_cols].fillna(
                    group[entropy_cols].mean()
                )
            elif method == "median":
                df.loc[mask, entropy_cols] = group[entropy_cols].fillna(
                    group[entropy_cols].median()
                )

        return df


def _build_entropy_funcs():
    """
    Build entropy function registry.

    Tries to import antropy for advanced measures, falls back to
    Shannon-only if not available.

    Returns
    -------
    dict
        Registry of {name: function} pairs.
    """
    funcs = {"shannon": SimpleEntropy.compute_entropy_1}

    try:
        import antropy as ant

        funcs.update({
            "app_entropy": lambda series: SimpleEntropy.compute_entropy(ant.app_entropy, series),
            "sample_entropy": lambda series: SimpleEntropy.compute_entropy(ant.sample_entropy, series),
            "perm_entropy": lambda series: SimpleEntropy.compute_entropy(ant.perm_entropy, series),
            "spectral_entropy": lambda series: SimpleEntropy.compute_entropy(ant.spectral_entropy, series),
            "svd_entropy": lambda series: SimpleEntropy.compute_entropy(ant.svd_entropy, series),
        })
    except ImportError:
        pass  # antropy not installed; only Shannon available

    return funcs


# Module-level registry
entropy_funcs = _build_entropy_funcs()
