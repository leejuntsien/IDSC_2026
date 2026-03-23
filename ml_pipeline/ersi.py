"""
Entropy-Ranked Stability Index (ERSI)
======================================
Heuristic tool for signal stability analysis using informational entropy.

Provides methods for:
- Simple ERSI computation (rank/weight per entropy column)
- Aggregate ERSI (single scalar per entropy measure)
- Time-series ERSI (combined multi-entropy measure over time)
- Region-specific ERSI (body region grouping)

Source: ersi_helper.py
"""
import numpy as np
import pandas as pd


class ERSI:
    """
    Entropy-Ranked Stability Index (ERSI).

    Methods
    -------
    ERSI_computation(df, cols)
        Compute ERSI for selected entropy columns.
    ERSI_aggregate(df, entropies)
        Aggregate ERSI across all entropy columns.
    ERSI_timeseries(df, entropies)
        Time-series ERSI combining multiple entropy measures.
    ERSI_by_region_timeseries(df, entropies, regions, ...)
        ERSI time series per body region.
    ERSI_by_region_aggregate(df, entropies, regions, ...)
        ERSI aggregate per body region.
    """
    @staticmethod
    def ERSI_full(df, entropy_cols):
        """Full dual‑ranking ERSI."""
        # Time ranking (within columns)
        ranks_time = df[entropy_cols].rank(ascending=True, method="min")
        weights_time = 1.0 / ranks_time
        
        # Cross‑entropy ranking (within rows)
        ranks_cross = df[entropy_cols].rank(ascending=True, method="min", axis=1)
        weights_cross = 1.0 / ranks_cross
        
        # Multiplicative fusion
        ersi = df[entropy_cols] * weights_time * weights_cross
        ersi['ERSI_full'] = ersi.mean(axis=1)
        return ersi

    @staticmethod
    def ERSI_computation(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        Compute ERSI for selected entropy columns.

        Steps:
        1. Rank values within each column.
        2. Weight = 1/rank.
        3. ERSI = entropy * weight.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with entropy columns.
        cols : list
            List of entropy column names.

        Returns
        -------
        pd.DataFrame
            DataFrame with added _rank, _weight, and _ERSI columns.
        """
        df = df.copy()
        for col in cols:
            rank_col = f"{col}_rank"
            weight_col = f"{col}_weight"
            ersi_col = f"{col}_ERSI"

            df[rank_col] = df[col].rank(ascending=True, method="min")
            df[weight_col] = 1.0 / df[rank_col]
            df[ersi_col] = df[col] * df[weight_col]

        return df

    @staticmethod
    def ERSI_aggregate(df: pd.DataFrame, entropies: list) -> pd.Series:
        """
        Aggregate ERSI across all entropy columns.
        Returns one scalar per entropy column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with entropy columns.
        entropies : list
            List of entropy column names.

        Returns
        -------
        pd.Series
            Aggregate ERSI values.
        """
        result = {}
        for col in entropies:
            if col in df.columns:
                ranks = df[col].rank(ascending=True, method="min")
                weights = 1.0 / ranks
                ersi_vals = df[col] * weights
                result[col] = ersi_vals.sum()
        return pd.Series(result)

    @staticmethod
    def ERSI_timeseries(df: pd.DataFrame, entropies: list) -> pd.DataFrame:
        """
        Time-series ERSI combining multiple entropy measures.

        Adds *_rank, *_weight, *_ERSI for each entropy column,
        plus a combined 'ERSI_timeseries' column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with entropy columns.
        entropies : list
            List of entropy column names.

        Returns
        -------
        pd.DataFrame
            DataFrame with ERSI time series.
        """
        df = ERSI.ERSI_computation(df, entropies)

        ersi_cols = [f"{col}_ERSI" for col in entropies if f"{col}_ERSI" in df.columns]
        if ersi_cols:
            df["ERSI_timeseries"] = df[ersi_cols].sum(axis=1)

        return df

    @staticmethod
    def _minmax_norm(s: pd.Series) -> pd.Series:
        """Normalize a Series to [0,1]."""
        smin, smax = s.min(), s.max()
        if smax - smin == 0:
            return pd.Series(0.0, index=s.index)
        return (s - smin) / (smax - smin)

    @staticmethod
    def _select_cols(
        df: pd.DataFrame,
        entropies: list,
        signal_type=None,
        phase=None,
    ) -> list:
        """Select numerical columns matching entropy names, signal type, and/or phase."""
        cols = [c for c in df.select_dtypes(include=[np.number]).columns]
        filtered = []
        for c in cols:
            match_entropy = any(e.lower() in c.lower() for e in entropies)
            match_signal = signal_type is None or signal_type.lower() in c.lower()
            match_phase = phase is None or phase.lower() in c.lower()
            if match_entropy and match_signal and match_phase:
                filtered.append(c)
        return filtered

    @staticmethod
    def _group_by_region(cols: list, regions: list) -> dict:
        """Group columns by region keyword (hand, leg, smartwatch, etc.)."""
        grouped = {r: [] for r in regions}
        for c in cols:
            for r in regions:
                if r.lower() in c.lower():
                    grouped[r].append(c)
                    break
        return grouped

    @staticmethod
    def _ersi_matrix_for_region(df_region: pd.DataFrame) -> pd.Series:
        """
        Compute ERSI matrix for one region across multiple entropy columns.
        Combines intra-column ranking (weights by time) and cross-entropy ranking (weights by row).
        """
        ranks = df_region.rank(ascending=True, method="min")
        weights = 1.0 / ranks
        ersi = df_region * weights
        return ersi.sum(axis=1)

    @staticmethod
    def ERSI_by_region_timeseries(
        df: pd.DataFrame,
        entropies: list,
        regions: list,
        signal_type=None,
        phase=None,
        normalize: bool = True,
        add_timeindex: bool = True,
    ) -> pd.DataFrame:
        """
        Compute ERSI time series per region.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        entropies : list
            List of entropy measure names.
        regions : list
            List of region keywords (e.g., ['hand', 'leg', 'smartwatch']).
        signal_type : str, optional
            Filter by signal type (e.g., 'HR', 'temp').
        phase : str, optional
            Filter by phase (e.g., 'pre', 'post').
        normalize : bool
            Whether to min-max normalize results.
        add_timeindex : bool
            Whether to add a TimeIndex column.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['TimeIndex', region1, region2, ...].
        """
        cols = ERSI._select_cols(df, entropies, signal_type, phase)
        grouped = ERSI._group_by_region(cols, regions)

        result = pd.DataFrame()
        for region, region_cols in grouped.items():
            if region_cols:
                region_ersi = ERSI._ersi_matrix_for_region(df[region_cols])
                result[region] = ERSI._minmax_norm(region_ersi) if normalize else region_ersi

        if add_timeindex:
            result.insert(0, "TimeIndex", range(len(result)))

        return result

    @staticmethod
    def ERSI_by_region_aggregate(
        df: pd.DataFrame,
        entropies: list,
        regions: list,
        signal_type=None,
        phase=None,
        normalize: bool = True,
    ) -> pd.Series:
        """
        Compute ERSI aggregate per region.

        Returns a Series with one scalar ERSI per region.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        entropies : list
            List of entropy measure names.
        regions : list
            List of region keywords.
        signal_type : str, optional
            Filter by signal type.
        phase : str, optional
            Filter by phase.
        normalize : bool
            Whether to normalize.

        Returns
        -------
        pd.Series
            Aggregate ERSI per region.
        """
        ts = ERSI.ERSI_by_region_timeseries(
            df, entropies, regions, signal_type, phase, normalize, add_timeindex=False
        )
        return ts.sum()
