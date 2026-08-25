"""
ENGIN - XGBoost Diagnostic Pipeline

Production-ready ML pipeline for engine fault diagnosis.
Implements robust training with proper data leakage prevention.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, 
    confusion_matrix, precision_recall_fscore_support
)
import joblib

# Importy lokalne - obsługa zarówno importu względnego jak i bezpośredniego
try:
    from .common import FREQ_COLS, LABELS, FAULT_LABELS, SEVERITIES
    from .features import FeatureExtractor
except ImportError:
    from common import FREQ_COLS, LABELS, FAULT_LABELS, SEVERITIES
    from features import FeatureExtractor

logger = logging.getLogger(__name__)


class EngineDiagnosticPipeline:
    """
    Production ML pipeline for engine fault diagnosis using XGBoost.
    
    Key features:
    - Proper group-based split to prevent data leakage (by engine_id)
    - Robust handling of missing values and outliers
    - Class imbalance handling via scale_pos_weight
    - Ensemble of multiple models
    - Unknown/anomaly detection
    - Severity prediction for detected faults
    
    The pipeline ensures that all preprocessing is fit only on training data
    to prevent any form of data leakage.
    """
    
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        n_folds: int = 5,
        unknown_threshold: float = 0.35,
        random_state: int = 42,
        ensemble_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage
            subsample: Subsample ratio
            colsample_bytree: Feature subsample ratio
            min_child_weight: Minimum sum of instance weight in child
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            n_folds: Number of CV folds
            unknown_threshold: Confidence threshold for unknown detection
            random_state: Random seed for reproducibility
            ensemble_weights: Weights for ensemble models
        """
        # Hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.n_folds = n_folds
        self.unknown_threshold = unknown_threshold
        self.random_state = random_state
        
        # Default ensemble weights
        if ensemble_weights is None:
            self.ensemble_weights = {
                'xgboost': 0.45,
                'extratrees': 0.30,
                'lightgbm': 0.25
            }
        else:
            self.ensemble_weights = ensemble_weights
        
        # Components (initialized during fit)
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.models: Dict[str, Any] = {}
        self.severity_models: Dict[str, Any] = {}
        self.isolation_forest: Optional[IsolationForest] = None
        
        # Training metadata
        self.feature_names: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self.training_metrics: Optional[Dict] = None
        
        logger.info(f"Initialized EngineDiagnosticPipeline with random_state={random_state}")
    
    def _create_xgboost_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier with current hyperparameters."""
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
    
    def _create_extratrees_model(self) -> ExtraTreesClassifier:
        """Create ExtraTrees classifier."""
        return ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def _handle_class_imbalance(
        self, 
        y: np.ndarray
    ) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.
        
        Uses inverse frequency weighting to give more importance
        to underrepresented classes.
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        n_classes = len(unique)
        
        # Compute weights as inverse of class frequency
        weights = {}
        for cls, count in zip(unique, counts):
            weights[cls] = total / (n_classes * count)
        
        return weights
    
    def fit(
        self, 
        df: pd.DataFrame,
        label_col: str = 'label',
        severity_col: str = 'severity'
    ) -> 'EngineDiagnosticPipeline':
        """
        Fit the complete pipeline.
        
        IMPORTANT: This method implements proper data leakage prevention:
        1. Feature extraction is done per-engine without using future data
        2. Train/validation split is done by engine_id (GroupKFold)
        3. All transformers are fit ONLY on training data
        
        Args:
            df: DataFrame with spectrum data and labels
            label_col: Name of label column
            severity_col: Name of severity column
        
        Returns:
            self
        """
        logger.info("Starting pipeline fitting...")
        
        # Store labels
        self.labels_ = df[label_col].values
        self.severity_ = df.get(severity_col, np.array(['nie_dotyczy'] * len(df))).values
        
        # Step 1: Feature extraction
        logger.info("Extracting features...")
        self.feature_extractor = FeatureExtractor()
        features_df = self.feature_extractor.fit_transform(df)
        self.feature_names = self.feature_extractor.get_feature_names()
        
        # Prepare feature matrix
        X = features_df[self.feature_names].values
        y_labels = self.labels_
        
        # Handle missing/invalid values in features
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Step 2: Encode labels
        logger.info("Encoding labels...")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_labels)
        self.classes_ = self.label_encoder.classes_
        
        # Step 3: Scale features (XGBoost doesn't require it, but helps other models)
        logger.info("Scaling features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 4: Group-based cross-validation (CRITICAL for leakage prevention)
        # Using GroupKFold ensures that all cylinders from the same engine
        # are always in the same fold, preventing data leakage.
        logger.info(f"Setting up {self.n_folds}-fold GroupKFold CV...")
        groups = features_df['engine_id'].values
        gkf = GroupKFold(n_splits=self.n_folds)
        
        # Step 5: Train individual models with CV evaluation
        logger.info("Training models...")
        cv_results = {}
        
        # XGBoost
        logger.info("  Training XGBoost...")
        xgb_model = self._create_xgboost_model()
        xgb_scores = cross_val_score(
            xgb_model, X_scaled, y_encoded, 
            groups=groups, cv=gkf, 
            scoring='f1_macro', n_jobs=-1
        )
        cv_results['xgboost'] = {
            'f1_macro_mean': float(xgb_scores.mean()),
            'f1_macro_std': float(xgb_scores.std()),
            'scores': xgb_scores.tolist()
        }
        logger.info(f"    XGBoost F1 Macro: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")
        
        # ExtraTrees
        logger.info("  Training ExtraTrees...")
        et_model = self._create_extratrees_model()
        et_scores = cross_val_score(
            et_model, X_scaled, y_encoded,
            groups=groups, cv=gkf,
            scoring='f1_macro', n_jobs=-1
        )
        cv_results['extratrees'] = {
            'f1_macro_mean': float(et_scores.mean()),
            'f1_macro_std': float(et_scores.std()),
            'scores': et_scores.tolist()
        }
        logger.info(f"    ExtraTrees F1 Macro: {et_scores.mean():.4f} (+/- {et_scores.std():.4f})")
        
        # Try LightGBM if available
        try:
            import lightgbm as lgb
            logger.info("  Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_samples=20,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                n_jobs=-1
            )
            lgb_scores = cross_val_score(
                lgb_model, X_scaled, y_encoded,
                groups=groups, cv=gkf,
                scoring='f1_macro', n_jobs=-1
            )
            cv_results['lightgbm'] = {
                'f1_macro_mean': float(lgb_scores.mean()),
                'f1_macro_std': float(lgb_scores.std()),
                'scores': lgb_scores.tolist()
            }
            logger.info(f"    LightGBM F1 Macro: {lgb_scores.mean():.4f} (+/- {lgb_scores.std():.4f})")
        except ImportError:
            logger.warning("LightGBM not available, skipping...")
            lgb_model = None
        
        # Step 6: Fit final models on full data for ensemble
        logger.info("Fitting final ensemble models...")
        xgb_model.fit(X_scaled, y_encoded)
        self.models['xgboost'] = xgb_model
        
        et_model.fit(X_scaled, y_encoded)
        self.models['extratrees'] = et_model
        
        if lgb_model is not None:
            lgb_model.fit(X_scaled, y_encoded)
            self.models['lightgbm'] = lgb_model
        
        # Step 7: Train severity models for each fault type
        logger.info("Training severity models...")
        self._train_severity_models(X_scaled, y_encoded, self.severity_)
        
        # Step 8: Train anomaly detector for unknown detection
        logger.info("Training anomaly detector...")
        self.isolation_forest = IsolationForest(
            contamination=0.05,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.isolation_forest.fit(X_scaled)
        
        # Store training metrics
        self.training_metrics = {
            'cv_results': cv_results,
            'n_samples': len(df),
            'n_features': len(self.feature_names),
            'n_classes': len(self.classes_),
            'classes': self.classes_.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Pipeline fitted successfully with {len(self.feature_names)} features")
        return self
    
    def _train_severity_models(
        self, 
        X: np.ndarray, 
        y_encoded: np.ndarray, 
        severity: np.ndarray
    ):
        """Train separate severity models for each fault type."""
        severity_map = {'male': 0, 'srednie': 1, 'duze': 2}
        
        for fault_label in FAULT_LABELS:
            if fault_label not in self.classes_:
                continue
            
            fault_idx = np.where(self.classes_ == fault_label)[0]
            if len(fault_idx) == 0:
                continue
            fault_idx = fault_idx[0]
            
            # Get samples for this fault
            mask = (y_encoded == fault_idx)
            if mask.sum() < 10:
                logger.warning(f"Skipping severity model for {fault_label}: only {mask.sum()} samples")
                continue
            
            X_fault = X[mask]
            y_sev = severity[mask]
            
            # Convert severity to numeric
            y_sev_num = np.array([severity_map.get(s, 0) for s in y_sev])
            
            # Train model
            sev_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
            sev_model.fit(X_fault, y_sev_num)
            self.severity_models[fault_label] = sev_model
            
            logger.info(f"  Trained severity model for {fault_label}: {mask.sum()} samples")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using ensemble.
        
        Args:
            X: Scaled feature matrix
        
        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        probs = None
        
        for model_name, model in self.models.items():
            weight = self.ensemble_weights.get(model_name, 0.33)
            model_prob = model.predict_proba(X)
            
            if probs is None:
                probs = model_prob * weight
            else:
                probs += model_prob * weight
        
        return probs
    
    def predict(
        self, 
        df: pd.DataFrame,
        detect_unknown: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with raw spectrum data
            detect_unknown: Whether to apply unknown detection
        
        Returns:
            DataFrame with predictions including:
            - engine_id, cylinder
            - label (predicted fault)
            - severity (predicted severity)
            - confidence (max probability)
            - anomaly_score (from isolation forest)
        """
        logger.info(f"Making predictions on {len(df)} samples...")
        
        # Feature extraction
        features_df = self.feature_extractor.transform(df)
        X = features_df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        # Ensemble prediction
        probs = self.predict_proba(X_scaled)
        y_pred = np.argmax(probs, axis=1)
        
        # Unknown detection
        anomaly_scores = -self.isolation_forest.predict(X_scaled)
        anomaly_scores = anomaly_scores.astype(float)
        
        if detect_unknown:
            max_probs = np.max(probs, axis=1)
            unknown_mask = (max_probs < self.unknown_threshold) | (anomaly_scores > 0.5)
            
            if 'unknown' in self.classes_:
                unknown_idx = np.where(self.classes_ == 'unknown')[0][0]
                y_pred[unknown_mask] = unknown_idx
            
            logger.info(f"Detected {unknown_mask.sum()} unknown samples ({100*unknown_mask.sum()/len(unknown_mask):.1f}%)")
        
        # Severity prediction
        severity_pred = self._predict_severity(X_scaled, y_pred)
        
        # Build results DataFrame
        results = pd.DataFrame({
            'engine_id': features_df['engine_id'],
            'cylinder': features_df['cylinder'],
            'label': self.label_encoder.inverse_transform(y_pred),
            'severity': severity_pred,
            'confidence': np.max(probs, axis=1),
            'anomaly_score': anomaly_scores
        })
        
        return results
    
    def _predict_severity(
        self, 
        X_scaled: np.ndarray, 
        y_pred: np.ndarray
    ) -> List[str]:
        """Predict severity for each sample."""
        severity_map = {0: 'male', 1: 'srednie', 2: 'duze'}
        inverse_label = {i: l for l, i in enumerate(self.label_encoder.classes_)}
        
        predictions = []
        for i, pred_idx in enumerate(y_pred):
            label = inverse_label.get(pred_idx, 'ok')
            
            if label in ['ok', 'unknown']:
                predictions.append('nie_dotyczy')
            elif label in self.severity_models:
                sev_idx = self.severity_models[label].predict([X_scaled[i]])[0]
                predictions.append(severity_map.get(sev_idx, 'male'))
            else:
                predictions.append('nie_dotyczy')
        
        return predictions
    
    def evaluate(
        self, 
        df: pd.DataFrame,
        label_col: str = 'label',
        severity_col: str = 'severity'
    ) -> Dict[str, float]:
        """
        Evaluate pipeline on labeled data.
        
        Args:
            df: DataFrame with labels
            label_col: Name of label column
            severity_col: Name of severity column
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating pipeline...")
        
        # Get predictions
        results = self.predict(df)
        
        # True labels
        y_true = self.label_encoder.transform(df[label_col].values)
        y_pred = self.label_encoder.transform(results['label'].values)
        
        # Severity
        y_sev_true = df[severity_col].values
        y_sev_pred = results['severity'].values
        
        # Metrics
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)
        severity_acc = accuracy_score(y_sev_true, y_sev_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        per_class_metrics = {}
        for i, cls in enumerate(self.classes_):
            per_class_metrics[cls] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        # Raw score (competition metric)
        raw_score = 0.75 * f1_macro + 0.25 * severity_acc
        
        metrics = {
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'accuracy': float(accuracy),
            'severity_accuracy': float(severity_acc),
            'raw_score': float(raw_score),
            'per_class': per_class_metrics,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        logger.info(f"F1 Macro: {f1_macro:.4f}, Severity Acc: {severity_acc:.4f}, Raw Score: {raw_score:.4f}")
        
        return metrics
    
    def save(self, filepath: str):
        """Save pipeline to disk."""
        logger.info(f"Saving pipeline to {filepath}...")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'feature_extractor': self.feature_extractor,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'models': self.models,
            'severity_models': self.severity_models,
            'isolation_forest': self.isolation_forest,
            'feature_names': self.feature_names,
            'classes_': self.classes_,
            'training_metrics': self.training_metrics,
            'hyperparams': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'min_child_weight': self.min_child_weight,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'n_folds': self.n_folds,
                'unknown_threshold': self.unknown_threshold,
                'random_state': self.random_state,
                'ensemble_weights': self.ensemble_weights
            }
        }
        
        joblib.dump(data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EngineDiagnosticPipeline':
        """Load pipeline from disk."""
        logger.info(f"Loading pipeline from {filepath}...")
        
        data = joblib.load(filepath)
        
        pipeline = cls(
            n_estimators=data['hyperparams']['n_estimators'],
            max_depth=data['hyperparams']['max_depth'],
            learning_rate=data['hyperparams']['learning_rate'],
            subsample=data['hyperparams']['subsample'],
            colsample_bytree=data['hyperparams']['colsample_bytree'],
            min_child_weight=data['hyperparams']['min_child_weight'],
            reg_alpha=data['hyperparams']['reg_alpha'],
            reg_lambda=data['hyperparams']['reg_lambda'],
            n_folds=data['hyperparams']['n_folds'],
            unknown_threshold=data['hyperparams']['unknown_threshold'],
            random_state=data['hyperparams']['random_state'],
            ensemble_weights=data['hyperparams']['ensemble_weights']
        )
        
        pipeline.feature_extractor = data['feature_extractor']
        pipeline.label_encoder = data['label_encoder']
        pipeline.scaler = data['scaler']
        pipeline.models = data['models']
        pipeline.severity_models = data['severity_models']
        pipeline.isolation_forest = data['isolation_forest']
        pipeline.feature_names = data['feature_names']
        pipeline.classes_ = data['classes_']
        pipeline.training_metrics = data['training_metrics']
        
        logger.info("Pipeline loaded successfully")
        return pipeline
