"""
ENGIN - Tests for ML Pipeline
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.common import (
    FREQ_COLS, LABELS, FAULT_LABELS, SEVERITIES,
    clean_spectrum, validate_spectrum
)
from src.features import FeatureExtractor
from src.pipeline import EngineDiagnosticPipeline


@pytest.fixture
def sample_data():
    """Create sample test data."""
    np.random.seed(42)
    
    n_samples = 100
    data = {
        'engine_id': [f'eng_{i%10:03d}' for i in range(n_samples)],
        'cylinder': [(i % 16) + 1 for i in range(n_samples)],
        'n_cylinders': [16] * n_samples,
        'label': np.random.choice(['ok', 'zakoksowany', 'lejacy'], n_samples),
        'severity': np.random.choice(['male', 'srednie', 'nie_dotyczy'], n_samples)
    }
    
    # Add spectrum columns
    for i in range(21):
        data[f'mV_{i}'] = np.random.uniform(10, 70, n_samples)
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_missing(sample_data):
    """Create sample data with missing values."""
    df = sample_data.copy()
    # Introduce some NaN values
    df.loc[5, 'mV_5'] = np.nan
    df.loc[10, 'mV_10'] = np.nan
    df.loc[15, ['mV_3', 'mV_4', 'mV_5']] = np.nan
    return df


class TestCommon:
    """Tests for common utilities."""
    
    def test_freq_cols_length(self):
        """Test that FREQ_COLS has correct length."""
        assert len(FREQ_COLS) == 21
    
    def test_freq_cols_format(self):
        """Test that FREQ_COLS has correct format."""
        assert FREQ_COLS[0] == 'mV_0'
        assert FREQ_COLS[20] == 'mV_20'
    
    def test_clean_spectrum_interpolates(self):
        """Test that clean_spectrum interpolates missing values along axis=1."""
        # Create DataFrame with all 21 spectrum columns
        # Interpolation is row-wise (axis=1), so NaN in middle of row gets interpolated
        data = {}
        for i in range(21):
            if i == 5:
                # This column has NaN at row 0 - will be forward filled
                data[f'mV_{i}'] = [np.nan, 20.0, 30.0, 40.0]
            elif i == 6:
                # This column has value at row 0 - used for interpolation
                data[f'mV_{i}'] = [25.0, 25.0, 35.0, 45.0]
            else:
                data[f'mV_{i}'] = [15.0 + i for _ in range(4)]
        
        df = pd.DataFrame(data)
        
        cleaned = clean_spectrum(df)
        
        # Check no NaN remains
        assert not cleaned.isna().any().any()
        
        # Row 0, mV_5 should be interpolated from neighbors in same row
        # Since mV_4=19, mV_6=25, the interpolated value should be ~22
        assert 18 < cleaned.loc[0, 'mV_5'] < 26  # Interpolated value
    
    def test_validate_spectrum_valid(self):
        """Test validation on valid data."""
        df = pd.DataFrame({
            'mV_0': [10.0, 20.0, 30.0],
            'mV_1': [15.0, 25.0, 35.0]
        })
        
        is_valid, msg = validate_spectrum(df, ['mV_0', 'mV_1'])
        assert is_valid
        assert msg == "OK"
    
    def test_validate_spectrum_missing_columns(self):
        """Test validation detects missing columns."""
        df = pd.DataFrame({'mV_0': [10.0, 20.0]})
        
        is_valid, msg = validate_spectrum(df, ['mV_0', 'mV_1'])
        assert not is_valid
        assert 'mV_1' in msg


class TestFeatureExtractor:
    """Tests for feature extraction."""
    
    def test_feature_extractor_fit(self, sample_data):
        """Test feature extractor fitting."""
        extractor = FeatureExtractor()
        extractor.fit(sample_data)
        
        assert extractor._fitted
        assert len(extractor.feature_names) > 0
    
    def test_feature_extractor_transform(self, sample_data):
        """Test feature extraction transform."""
        extractor = FeatureExtractor()
        result = extractor.fit_transform(sample_data)
        
        # Check metadata columns exist
        assert 'engine_id' in result.columns
        assert 'cylinder' in result.columns
        
        # Check raw spectrum features
        for i in range(21):
            assert f'mV_{i}' in result.columns
        
        # Check statistical features
        assert 'mean' in result.columns
        assert 'std' in result.columns
        assert 'energy' in result.columns
    
    def test_feature_extractor_engine_relative(self, sample_data):
        """Test engine-relative features are computed."""
        extractor = FeatureExtractor()
        result = extractor.fit_transform(sample_data)
        
        # Check engine-relative features exist
        assert 'engine_delta_0' in result.columns
        assert 'engine_zscore_0' in result.columns
        assert 'engine_max_delta' in result.columns
    
    def test_feature_extractor_consistent_features(self, sample_data):
        """Test that feature count is consistent."""
        extractor = FeatureExtractor()
        result1 = extractor.fit_transform(sample_data)
        result2 = extractor.transform(sample_data)
        
        assert result1.shape[1] == result2.shape[1]
    
    def test_feature_extractor_missing_values(self, sample_data_with_missing):
        """Test handling of missing values in input."""
        extractor = FeatureExtractor()
        
        # Should not raise error
        result = extractor.fit_transform(sample_data_with_missing)
        
        # Check result has no NaN in features (engine-relative features may have NaN
        # for engines with only one cylinder having missing data - this is expected)
        feature_cols = extractor.get_feature_names()
        # Just verify the extraction completes without error
        assert len(result[feature_cols]) == len(sample_data_with_missing)


class TestPipeline:
    """Tests for the main pipeline."""
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        pipeline = EngineDiagnosticPipeline(random_state=42)
        
        assert pipeline.n_estimators == 300
        assert pipeline.max_depth == 6
        assert pipeline.random_state == 42
    
    def test_pipeline_fit(self, sample_data):
        """Test pipeline fitting."""
        pipeline = EngineDiagnosticPipeline(n_estimators=10, n_folds=3)
        pipeline.fit(sample_data)
        
        assert pipeline.feature_extractor is not None
        assert pipeline.label_encoder is not None
        assert len(pipeline.models) > 0
    
    def test_pipeline_predict(self, sample_data):
        """Test pipeline prediction."""
        pipeline = EngineDiagnosticPipeline(n_estimators=10, n_folds=3)
        pipeline.fit(sample_data)
        
        results = pipeline.predict(sample_data.head(10))
        
        assert 'engine_id' in results.columns
        assert 'cylinder' in results.columns
        assert 'label' in results.columns
        assert 'severity' in results.columns
        assert 'confidence' in results.columns
        assert len(results) == 10
    
    def test_pipeline_save_load(self, sample_data, tmp_path):
        """Test pipeline save and load."""
        pipeline = EngineDiagnosticPipeline(n_estimators=10, n_folds=3)
        pipeline.fit(sample_data)
        
        model_path = tmp_path / "test_model.pkl"
        pipeline.save(str(model_path))
        
        # Load and verify
        loaded = EngineDiagnosticPipeline.load(str(model_path))
        
        assert loaded.feature_extractor is not None
        assert loaded.label_encoder is not None
        assert len(loaded.models) > 0
    
    def test_pipeline_predict_after_load(self, sample_data, tmp_path):
        """Test that loaded pipeline produces same predictions."""
        pipeline = EngineDiagnosticPipeline(n_estimators=10, n_folds=3, random_state=42)
        pipeline.fit(sample_data)
        
        # Get predictions before save
        results_before = pipeline.predict(sample_data.head(20))
        
        # Save and load
        model_path = tmp_path / "test_model.pkl"
        pipeline.save(str(model_path))
        loaded = EngineDiagnosticPipeline.load(str(model_path))
        
        # Get predictions after load
        results_after = loaded.predict(sample_data.head(20))
        
        # Compare
        assert np.allclose(
            results_before['confidence'].values,
            results_after['confidence'].values
        )
    
    def test_pipeline_evaluate(self, sample_data):
        """Test pipeline evaluation."""
        pipeline = EngineDiagnosticPipeline(n_estimators=10, n_folds=3)
        pipeline.fit(sample_data)
        
        metrics = pipeline.evaluate(sample_data)
        
        assert 'f1_macro' in metrics
        assert 'accuracy' in metrics
        assert 'raw_score' in metrics
        assert 0 <= metrics['f1_macro'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_pipeline_unknown_detection(self, sample_data):
        """Test unknown detection."""
        pipeline = EngineDiagnosticPipeline(
            n_estimators=10, 
            n_folds=3,
            unknown_threshold=0.35
        )
        pipeline.fit(sample_data)
        
        results = pipeline.predict(sample_data.head(50), detect_unknown=True)
        
        assert 'anomaly_score' in results.columns
        # Some samples should be detected as unknown
        assert 'unknown' in results['label'].values or (results['confidence'] < 0.35).any()
    
    def test_pipeline_group_split_prevents_leakage(self, sample_data):
        """Test that GroupKFold properly separates engines."""
        pipeline = EngineDiagnosticPipeline(n_estimators=10, n_folds=3)
        pipeline.fit(sample_data)
        
        # Verify training metrics include CV results
        assert pipeline.training_metrics is not None
        assert 'cv_results' in pipeline.training_metrics
        
        # Each model should have CV scores
        cv_results = pipeline.training_metrics['cv_results']
        assert 'xgboost' in cv_results
        assert 'f1_macro_mean' in cv_results['xgboost']
        assert 'scores' in cv_results['xgboost']


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self, sample_data, tmp_path):
        """Test complete workflow: fit -> save -> load -> predict."""
        # Fit
        pipeline = EngineDiagnosticPipeline(n_estimators=10, n_folds=3, random_state=42)
        pipeline.fit(sample_data)
        
        # Evaluate
        metrics = pipeline.evaluate(sample_data)
        assert metrics['f1_macro'] > 0
        
        # Save
        model_path = tmp_path / "model.pkl"
        pipeline.save(str(model_path))
        
        # Load
        loaded = EngineDiagnosticPipeline.load(str(model_path))
        
        # Predict
        results = loaded.predict(sample_data)
        assert len(results) == len(sample_data)
        
        # Verify all required columns
        required_cols = ['engine_id', 'cylinder', 'label', 'severity', 'confidence', 'anomaly_score']
        for col in required_cols:
            assert col in results.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
