"""
ENGIN - Feature Engineering Module

Extracts acoustic features from spectrum data.
All features are physically motivated for engine diagnostics.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import zscore

# Importy lokalne - obsługa zarówno importu względnego jak i bezpośredniego
try:
    from .common import (
        FREQ_COLS, STATS_FEATURES, GRADIENT_FEATURES, PEAK_FEATURES,
        BAND_FEATURES, ENGINE_DELTA_FEATURES, ENGINE_ZSCORE_FEATURES,
        ENGINE_RELATIVE_FEATURES
    )
except ImportError:
    from common import (
        FREQ_COLS, STATS_FEATURES, GRADIENT_FEATURES, PEAK_FEATURES,
        BAND_FEATURES, ENGINE_DELTA_FEATURES, ENGINE_ZSCORE_FEATURES,
        ENGINE_RELATIVE_FEATURES
    )


def extract_basic_features(spectrum: np.ndarray) -> Dict[str, float]:
    """
    Extract basic statistical features from spectrum.
    
    Features:
    - mean, std, min, max, median, range
    - rms (root mean square)
    - energy (sum of squares)
    - skewness, kurtosis
    """
    features = {}
    
    features['mean'] = np.mean(spectrum)
    features['std'] = np.std(spectrum)
    features['min'] = np.min(spectrum)
    features['max'] = np.max(spectrum)
    features['median'] = np.median(spectrum)
    features['range'] = features['max'] - features['min']
    features['rms'] = np.sqrt(np.mean(spectrum**2))
    features['energy'] = np.sum(spectrum**2)
    
    # Skewness and kurtosis (handle zero std)
    if features['std'] > 1e-10:
        features['skewness'] = float(((spectrum - features['mean'])**3).mean() / (features['std']**3))
        features['kurtosis'] = float(((spectrum - features['mean'])**4).mean() / (features['std']**4) - 3)
    else:
        features['skewness'] = 0.0
        features['kurtosis'] = 0.0
    
    return features


def extract_gradient_features(spectrum: np.ndarray) -> Dict[str, float]:
    """
    Extract gradient-based features (first and second derivatives).
    
    These capture how quickly the spectrum changes across frequencies.
    """
    features = {}
    
    # First derivative
    diffs = np.diff(spectrum)
    features['diff_mean'] = float(np.mean(diffs))
    features['diff_std'] = float(np.std(diffs))
    features['diff_max'] = float(np.max(np.abs(diffs)))
    
    # Second derivative
    if len(diffs) > 1:
        second_diffs = np.diff(diffs)
        features['second_diff_mean'] = float(np.mean(second_diffs))
        features['second_diff_std'] = float(np.std(second_diffs))
    else:
        features['second_diff_mean'] = 0.0
        features['second_diff_std'] = 0.0
    
    return features


def extract_peak_features(spectrum: np.ndarray) -> Dict[str, float]:
    """
    Extract peak-related features from spectrum.
    
    Peaks in specific frequency bands can indicate mechanical issues.
    """
    features = {}
    
    mean_val = np.mean(spectrum)
    peaks, properties = find_peaks(spectrum, height=mean_val)
    
    features['n_peaks'] = len(peaks)
    
    if len(peaks) > 0:
        # Highest peak
        peak_idx = int(peaks[np.argmax(spectrum[peaks])])
        features['peak_freq_idx'] = peak_idx
        features['peak_height'] = float(spectrum[peak_idx])
        
        # Second highest peak
        if len(peaks) > 1:
            sorted_peaks = sorted(peaks, key=lambda p: spectrum[p], reverse=True)
            features['second_peak_height'] = float(spectrum[sorted_peaks[1]])
        else:
            features['second_peak_height'] = 0.0
    else:
        features['peak_freq_idx'] = 0
        features['peak_height'] = float(np.max(spectrum))
        features['second_peak_height'] = 0.0
    
    return features


def extract_band_energy_features(spectrum: np.ndarray) -> Dict[str, float]:
    """
    Extract energy in different frequency bands.
    
    Bands are chosen to cover low, mid, and high frequency ranges
    which may correspond to different mechanical phenomena.
    """
    features = {}
    
    # Divide spectrum into 7 bands (approximately 3 kHz each)
    bands = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21)]
    
    for i, (start, end) in enumerate(bands):
        band_energy = float(np.sum(spectrum[start:end]**2))
        features[f'energy_band_{i}'] = band_energy
    
    return features


def extract_engine_relative_features(
    spectrum: np.ndarray,
    engine_data: pd.DataFrame,
    cylinder: int
) -> Dict[str, float]:
    """
    Extract features relative to other cylinders in the same engine.
    
    This is a leave-one-out comparison that helps detect anomalies
    specific to a single cylinder while being robust to engine-to-engine
    variation.
    
    IMPORTANT: This uses only OTHER cylinders from the same engine,
    preventing data leakage during training.
    """
    features = {}
    
    # Get other cylinders from same engine
    other_cylinders = engine_data[engine_data['cylinder'] != cylinder]
    
    if len(other_cylinders) > 0:
        other_spectra = other_cylinders[FREQ_COLS].values.astype(float)
        median_other = np.median(other_spectra, axis=0)
        std_other = np.std(other_spectra, axis=0) + 1e-8
        
        # Deviation from median of other cylinders (per frequency)
        for i in range(21):
            features[f'engine_delta_{i}'] = float(spectrum[i] - median_other[i])
            features[f'engine_zscore_{i}'] = float((spectrum[i] - median_other[i]) / std_other[i])
        
        # Summary statistics
        features['engine_max_delta'] = float(np.max(np.abs(spectrum - median_other)))
        features['engine_mean_zscore'] = float(np.mean(np.abs((spectrum - median_other) / std_other)))
        features['engine_distance'] = float(np.sqrt(np.sum((spectrum - median_other)**2)))
    else:
        # No other cylinders available (single-cylinder engine or missing data)
        for i in range(21):
            features[f'engine_delta_{i}'] = 0.0
            features[f'engine_zscore_{i}'] = 0.0
        features['engine_max_delta'] = 0.0
        features['engine_mean_zscore'] = 0.0
        features['engine_distance'] = 0.0
    
    return features


class FeatureExtractor:
    """
    Main feature extraction class.
    
    Extracts all features from raw spectrum data in a consistent manner.
    Designed to be used both during training and inference.
    """
    
    def __init__(self):
        self.feature_names: Optional[List[str]] = None
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeatureExtractor':
        """
        Fit the feature extractor (determine feature names).
        
        Args:
            df: DataFrame with engine_id and cylinder columns
        
        Returns:
            self
        """
        # Build feature name list
        feature_names = []
        
        # Raw spectrum (21 features)
        feature_names.extend(FREQ_COLS)
        
        # Statistical features (10)
        feature_names.extend(STATS_FEATURES)
        
        # Gradient features (5)
        feature_names.extend(GRADIENT_FEATURES)
        
        # Peak features (4)
        feature_names.extend(PEAK_FEATURES)
        
        # Band energy features (7)
        feature_names.extend(BAND_FEATURES)
        
        # Engine-relative features (21 + 21 + 3 = 45)
        feature_names.extend(ENGINE_DELTA_FEATURES)
        feature_names.extend(ENGINE_ZSCORE_FEATURES)
        feature_names.extend(ENGINE_RELATIVE_FEATURES)
        
        self.feature_names = feature_names
        self._fitted = True
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from DataFrame.
        
        Args:
            df: DataFrame with spectrum columns and engine_id, cylinder
        
        Returns:
            DataFrame with extracted features
        """
        if not self._fitted:
            raise RuntimeError("FeatureExtractor must be fitted before transform")
        
        features_list = []
        
        # Group by engine for engine-relative features
        if 'engine_id' in df.columns:
            grouped = df.groupby('engine_id')
        else:
            # Create dummy groups if no engine_id
            df = df.copy()
            df['engine_id'] = 'dummy'
            grouped = df.groupby('engine_id')
        
        for idx, row in df.iterrows():
            features = {}
            
            # Metadata
            features['engine_id'] = row['engine_id']
            features['cylinder'] = row['cylinder']
            features['n_cylinders'] = row.get('n_cylinders', 16)
            
            # Extract spectrum
            spectrum = row[FREQ_COLS].values.astype(float)
            
            # Raw spectrum features
            for i, val in enumerate(spectrum):
                features[f'mV_{i}'] = val
            
            # Basic statistical features
            features.update(extract_basic_features(spectrum))
            
            # Gradient features
            features.update(extract_gradient_features(spectrum))
            
            # Peak features
            features.update(extract_peak_features(spectrum))
            
            # Band energy features
            features.update(extract_band_energy_features(spectrum))
            
            # Engine-relative features
            engine_id = row['engine_id']
            try:
                engine_df = grouped.get_group(engine_id)
                features.update(extract_engine_relative_features(
                    spectrum, engine_df, row['cylinder']
                ))
            except KeyError:
                # Engine not found - use defaults
                features.update(extract_engine_relative_features(
                    spectrum, pd.DataFrame(), row['cylinder']
                ))
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names (excluding metadata)."""
        if not self._fitted:
            raise RuntimeError("FeatureExtractor must be fitted first")
        
        # Exclude metadata columns
        exclude = {'engine_id', 'cylinder', 'n_cylinders'}
        return [f for f in self.feature_names if f not in exclude]
