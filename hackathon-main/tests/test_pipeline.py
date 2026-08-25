"""
Tests for the SVC-based ML Pipeline
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
    for i in range(21):
        data[f'mV_{i}'] = np.random.uniform(10, 70, n_samples)
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_missing(sample_data):
    """Create sample data with missing values."""
    df = sample_data.copy()
    df.loc[5, 'mV_5'] = np.nan
    df.loc[10, 'mV_10'] = np.nan
    df.loc[15, ['mV_3', 'mV_4', 'mV_5']] = np.nan
    return df


class TestCommon:
    def test_freq_cols_length(self):
        assert len(FREQ_COLS) == 21

    def test_freq_cols_format(self):
        assert FREQ_COLS[0] == 'mV_0'
        assert FREQ_COLS[20] == 'mV_20'

    def test_clean_spectrum_interpolates(self):
        data = {}
        for i in range(21):
            if i == 5:
                data[f'mV_{i}'] = [np.nan, 20.0, 30.0, 40.0]
            elif i == 6:
                data[f'mV_{i}'] = [25.0, 25.0, 35.0, 45.0]
            else:
                data[f'mV_{i}'] = [15.0 + i for _ in range(4)]
        df = pd.DataFrame(data)
        cleaned = clean_spectrum(df)
        assert not cleaned.isna().any().any()
        assert 18 < cleaned.loc[0, 'mV_5'] < 26

    def test_validate_spectrum_valid(self):
        df = pd.DataFrame({'mV_0': [10.0, 20.0, 30.0], 'mV_1': [15.0, 25.0, 35.0]})
        is_valid, msg = validate_spectrum(df, ['mV_0', 'mV_1'])
        assert is_valid
        assert msg == "OK"

    def test_validate_spectrum_missing_columns(self):
        df = pd.DataFrame({'mV_0': [10.0, 20.0]})
        is_valid, msg = validate_spectrum(df, ['mV_0', 'mV_1'])
        assert not is_valid
        assert 'mV_1' in msg


class TestFeatureExtractor:
    def test_feature_extractor_fit(self, sample_data):
        extractor = FeatureExtractor()
        extractor.fit(sample_data)
        assert extractor._fitted
        assert len(extractor.feature_names) > 0

    def test_feature_extractor_transform(self, sample_data):
        extractor = FeatureExtractor()
        result = extractor.fit_transform(sample_data)
        assert 'engine_id' in result.columns
        assert 'cylinder' in result.columns
        for i in range(21):
            assert f'mV_{i}' in result.columns
        assert 'mean' in result.columns
        assert 'std' in result.columns
        assert 'energy' in result.columns

    def test_feature_extractor_engine_relative(self, sample_data):
        extractor = FeatureExtractor()
        result = extractor.fit_transform(sample_data)
        assert 'engine_delta_0' in result.columns
        assert 'engine_zscore_0' in result.columns
        assert 'engine_max_delta' in result.columns

    def test_feature_extractor_consistent_features(self, sample_data):
        extractor = FeatureExtractor()
        result1 = extractor.fit_transform(sample_data)
        result2 = extractor.transform(sample_data)
        assert result1.shape[1] == result2.shape[1]

    def test_feature_extractor_missing_values(self, sample_data_with_missing):
        extractor = FeatureExtractor()
        result = extractor.fit_transform(sample_data_with_missing)
        feature_cols = extractor.get_feature_names()
        assert len(result[feature_cols]) == len(sample_data_with_missing)


class TestPipeline:
    def test_pipeline_init(self):
        pipeline = EngineDiagnosticPipeline(random_state=42)
        assert pipeline.n_estimators == 300
        assert pipeline.max_depth == 6
        assert pipeline.random_state == 42

    def test_pipeline_fit(self, sample_data):
        pipeline = EngineDiagnosticPipeline(n_folds=3)
        pipeline.fit(sample_data)
        assert pipeline.feature_extractor is not None
        assert pipeline.label_encoder is not None
        assert len(pipeline.models) > 0

    def test_pipeline_predict(self, sample_data):
        pipeline = EngineDiagnosticPipeline(n_folds=3)
        pipeline.fit(sample_data)
        results = pipeline.predict(sample_data.head(10))
        assert 'engine_id' in results.columns
        assert 'cylinder' in results.columns
        assert 'label' in results.columns
        assert 'severity' in results.columns
        assert 'confidence' in results.columns
        assert len(results) == 10

    def test_pipeline_save_load(self, sample_data, tmp_path):
        pipeline = EngineDiagnosticPipeline(n_folds=3)
        pipeline.fit(sample_data)
        model_path = tmp_path / "test_model.pkl"
        pipeline.save(str(model_path))
        loaded = EngineDiagnosticPipeline.load(str(model_path))
        assert loaded.feature_extractor is not None
        assert loaded.label_encoder is not None
        assert len(loaded.models) > 0

    def test_pipeline_predict_after_load(self, sample_data, tmp_path):
        pipeline = EngineDiagnosticPipeline(n_folds=3, random_state=42)
        pipeline.fit(sample_data)
        results_before = pipeline.predict(sample_data.head(20))
        model_path = tmp_path / "test_model.pkl"
        pipeline.save(str(model_path))
        loaded = EngineDiagnosticPipeline.load(str(model_path))
        results_after = loaded.predict(sample_data.head(20))
        assert np.allclose(results_before['confidence'].values, results_after['confidence'].values)

    def test_pipeline_evaluate(self, sample_data):
        pipeline = EngineDiagnosticPipeline(n_folds=3)
        pipeline.fit(sample_data)
        metrics = pipeline.evaluate(sample_data)
        assert 'f1_macro' in metrics
        assert 'accuracy' in metrics
        assert 'raw_score' in metrics
        assert 0 <= metrics['f1_macro'] <= 1
        assert 0 <= metrics['accuracy'] <= 1

    def test_pipeline_unknown_detection(self, sample_data):
        """An obviously shifted spectrum should trigger anomaly detection."""
        pipeline = EngineDiagnosticPipeline(n_folds=3, unknown_threshold=0.35)
        pipeline.fit(sample_data)
        # Multiply by a factor to create a distribution shift that affects d1 features.
        # Adding a constant would not work because d1 (derivative) is translation-invariant.
        shifted = sample_data.head(10).copy()
        shifted[FREQ_COLS] = shifted[FREQ_COLS] * 2.0
        results = pipeline.predict(shifted, detect_unknown=True)
        assert 'anomaly_score' in results.columns
        assert (results['anomaly_score'] > pipeline.anomaly_threshold).any()
        assert 'unknown' in results['label'].values or (results['confidence'] < 0.35).any()

    def test_pipeline_group_split_prevents_leakage(self, sample_data):
        """Test that GroupKFold metrics use the production SVC."""
        pipeline = EngineDiagnosticPipeline(n_folds=3)
        pipeline.fit(sample_data)
        assert pipeline.training_metrics is not None
        assert 'cv_results' in pipeline.training_metrics
        cv_results = pipeline.training_metrics['cv_results']
        assert 'svc' in cv_results
        assert 'f1_macro_mean' in cv_results['svc']
        assert 'scores' in cv_results['svc']


class TestIntegration:
    def test_full_workflow(self, sample_data, tmp_path):
        pipeline = EngineDiagnosticPipeline(n_folds=3, random_state=42)
        pipeline.fit(sample_data)
        metrics = pipeline.evaluate(sample_data)
        assert metrics['f1_macro'] > 0
        model_path = tmp_path / "model.pkl"
        pipeline.save(str(model_path))
        loaded = EngineDiagnosticPipeline.load(str(model_path))
        results = loaded.predict(sample_data)
        assert len(results) == len(sample_data)
        required_cols = ['engine_id', 'cylinder', 'label', 'severity', 'confidence', 'anomaly_score']
        for col in required_cols:
            assert col in results.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
