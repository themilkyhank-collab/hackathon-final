"""
ENGIN - Common utilities and constants
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


# Column definitions
FREQ_COLS: List[str] = [f"mV_{i}" for i in range(21)]
SPECTRUM_COLS: List[str] = FREQ_COLS.copy()

# Target columns
LABEL_COL: str = "label"
SEVERITY_COL: str = "severity"

# Class labels
LABELS: List[str] = ["ok", "zakoksowany", "lejacy", "pompa", "iglica", "unknown"]
FAULT_LABELS: List[str] = ["zakoksowany", "lejacy", "pompa", "iglica"]
SEVERITIES: List[str] = ["male", "srednie", "duze", "nie_dotyczy"]

# Feature groups
STATS_FEATURES: List[str] = [
    "mean", "std", "min", "max", "median", "range", 
    "rms", "energy", "skewness", "kurtosis"
]

GRADIENT_FEATURES: List[str] = [
    "diff_mean", "diff_std", "diff_max", 
    "second_diff_mean", "second_diff_std"
]

PEAK_FEATURES: List[str] = [
    "n_peaks", "peak_freq_idx", "peak_height", "second_peak_height"
]

BAND_FEATURES: List[str] = [f"energy_band_{i}" for i in range(7)]

ENGINE_DELTA_FEATURES: List[str] = [f"engine_delta_{i}" for i in range(21)]
ENGINE_ZSCORE_FEATURES: List[str] = [f"engine_zscore_{i}" for i in range(21)]

ENGINE_RELATIVE_FEATURES: List[str] = [
    "engine_max_delta", "engine_mean_zscore", "engine_distance"
]


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file with proper handling of missing values."""
    return pd.read_csv(filepath)


def clean_spectrum(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """
    Clean spectrum data by interpolating missing values.
    
    Args:
        df: DataFrame with spectrum columns
        cols: List of column names to clean (default: FREQ_COLS)
    
    Returns:
        DataFrame with interpolated values
    """
    if cols is None:
        cols = FREQ_COLS
    
    df = df.copy()
    
    # Interpolate along the spectrum axis (row-wise)
    df[cols] = df[cols].interpolate(method='linear', axis=1)
    
    # Forward fill then backward fill for edge cases
    df[cols] = df[cols].ffill(axis=1).bfill(axis=1)
    
    return df


def validate_spectrum(df: pd.DataFrame, cols: List[str] = None) -> Tuple[bool, str]:
    """
    Validate that spectrum data is complete and valid.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if cols is None:
        cols = FREQ_COLS
    
    # Check for missing columns
    missing_cols = set(cols) - set(df.columns)
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check for NaN values
    if df[cols].isna().any().any():
        n_nan = df[cols].isna().sum().sum()
        return False, f"Found {n_nan} NaN values in spectrum columns"
    
    # Check for Inf values
    if np.isinf(df[cols].values).any():
        n_inf = np.isinf(df[cols].values).sum()
        return False, f"Found {n_inf} Inf values in spectrum columns"
    
    # Check for negative values (should be non-negative amplitudes)
    if (df[cols] < 0).any().any():
        n_neg = (df[cols] < 0).sum().sum()
        return False, f"Found {n_neg} negative values in spectrum columns"
    
    return True, "OK"
