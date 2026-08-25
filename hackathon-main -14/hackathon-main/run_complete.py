#!/usr/bin/env python3
"""
🚀 ENGIN Hackathon - Kompletny Pipeline (Samodzielny Skrypt)

Ten skrypt zawiera CAŁY kod potrzebny do:
1. Wczytania danych
2. Feature Engineering (188 cech)
3. Porównania modeli (XGBoost, ExtraTrees, RandomForest, LightGBM)
4. Detekcji anomalii dla klasy unknown
5. Modeli severity
6. Ensemble
7. Predykcji i generowania predictions.csv

Uruchomienie:
    python run_complete.py

Wymagania:
    pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly tqdm joblib scipy
"""

# ======================
# 🎯 HIPERPARAMETRY
# ======================

HYPERPARAMS = {
    'xgboost': {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    },
    'extratrees': {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    },
    'randomforest': {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    },
    'lightgbm': {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    },
    'ensemble_weights': {
        'xgboost': 0.45,
        'extratrees': 0.30,
        'lightgbm': 0.25
    },
    'unknown_threshold': 0.35,
    'n_folds': 5
}

print("✅ HIPERPARAMETRY ZAŁADOWANE")
print(f"   - XGBoost: {HYPERPARAMS['xgboost']['n_estimators']} estymatorów")
print(f"   - ExtraTrees: {HYPERPARAMS['extratrees']['n_estimators']} drzew")
print(f"   - RandomForest: {HYPERPARAMS['randomforest']['n_estimators']} drzew")
print(f"   - LightGBM: {HYPERPARAMS['lightgbm']['n_estimators']} estymatorów")
print(f"   - Ensemble weights: {HYPERPARAMS['ensemble_weights']}")
print(f"   - Unknown threshold: {HYPERPARAMS['unknown_threshold']}")
print(f"   - CV folds: {HYPERPARAMS['n_folds']}")

# ======================
# 📦 IMPORTY
# ======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
from tqdm import tqdm
import os
from scipy.signal import find_peaks
from scipy.stats import zscore

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

print("\n✅ WSZYSTKIE BIBLIOTEKI ZAIMPORTOWANE")

# ======================
# 📊 WCZYTYWANIE DANYCH
# ======================

def load_data(data_dir='data'):
    """Wczytuje wszystkie pliki CSV"""
    print(f"\n📂 Wczytywanie danych z: {data_dir}")
    
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"   ✅ train.csv: {len(train_df)} wierszy, {train_df['engine_id'].nunique()} silników (bez etykiet)")
    print(f"   ✅ val.csv: {len(val_df)} wierszy, {val_df['engine_id'].nunique()} silników (z etykietami)")
    print(f"   ✅ test.csv: {len(test_df)} wierszy, {test_df['engine_id'].nunique()} silników (do predykcji)")
    
    return train_df, val_df, test_df

# ======================
# 🔧 FEATURE ENGINEERING
# ======================

def interpolate_missing(df):
    """Interpoluje brakujące wartości"""
    spectrum_cols = [f'mV_{i}' for i in range(21)]
    df[spectrum_cols] = df[spectrum_cols].interpolate(method='linear', axis=1)
    df[spectrum_cols] = df[spectrum_cols].fillna(method='ffill', axis=1)
    df[spectrum_cols] = df[spectrum_cols].fillna(method='bfill', axis=1)
    return df

def extract_features(df, is_test=False):
    """Ekstrahuje 188 cech dla każdego cylindra"""
    print(f"\n🔧 Feature Engineering...")
    
    features_list = []
    spectrum_cols = [f'mV_{i}' for i in range(21)]
    
    # Grupuj po silniku do cech engine-relative
    if 'engine_id' in df.columns:
        grouped = df.groupby('engine_id')
    else:
        # Dla test - traktuj każdy wiersz osobno lub użyj dummy group
        df['engine_id'] = df.get('engine_id', ['dummy'] * len(df))
        grouped = df.groupby('engine_id')
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cechy"):
        features = {}
        
        # Podstawowe informacje
        features['engine_id'] = row['engine_id']
        features['cylinder'] = row['cylinder']
        features['n_cylinders'] = row.get('n_cylinders', 16)
        
        # Surowe widmo (21 cech)
        spectrum = row[spectrum_cols].values.astype(float)
        for i, val in enumerate(spectrum):
            features[f'mV_{i}'] = val
        
        # Statystyki widma
        features['mean'] = np.mean(spectrum)
        features['std'] = np.std(spectrum)
        features['min'] = np.min(spectrum)
        features['max'] = np.max(spectrum)
        features['median'] = np.median(spectrum)
        features['range'] = features['max'] - features['min']
        features['rms'] = np.sqrt(np.mean(spectrum**2))
        features['energy'] = np.sum(spectrum**2)
        
        if features['std'] > 0:
            features['skewness'] = ((spectrum - features['mean'])**3).mean() / (features['std']**3)
            features['kurtosis'] = ((spectrum - features['mean'])**4).mean() / (features['std']**4) - 3
        else:
            features['skewness'] = 0
            features['kurtosis'] = 0
        
        # Różnice i gradienty
        diffs = np.diff(spectrum)
        features['diff_mean'] = np.mean(diffs)
        features['diff_std'] = np.std(diffs)
        features['diff_max'] = np.max(np.abs(diffs))
        
        if len(diffs) > 1:
            second_diffs = np.diff(diffs)
            features['second_diff_mean'] = np.mean(second_diffs)
            features['second_diff_std'] = np.std(second_diffs)
        else:
            features['second_diff_mean'] = 0
            features['second_diff_std'] = 0
        
        # Piki
        peaks, properties = find_peaks(spectrum, height=features['mean'])
        features['n_peaks'] = len(peaks)
        if len(peaks) > 0:
            features['peak_freq_idx'] = peaks[np.argmax(spectrum[peaks])]
            features['peak_height'] = spectrum[features['peak_freq_idx']]
            if len(peaks) > 1:
                sorted_peaks = sorted(peaks, key=lambda p: spectrum[p], reverse=True)
                features['second_peak_height'] = spectrum[sorted_peaks[1]]
            else:
                features['second_peak_height'] = 0
        else:
            features['peak_freq_idx'] = 0
            features['peak_height'] = features['max']
            features['second_peak_height'] = 0
        
        # Energia w pasmach
        bands = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21)]
        for i, (start, end) in enumerate(bands):
            band_energy = np.sum(spectrum[start:end]**2)
            features[f'energy_band_{i}'] = band_energy
        
        # Cechy engine-relative (leave-one-out)
        engine_id = row['engine_id']
        try:
            engine_data = grouped.get_group(engine_id)
            other_cylinders = engine_data[engine_data['cylinder'] != row['cylinder']]
            
            if len(other_cylinders) > 0:
                other_spectra = other_cylinders[spectrum_cols].values.astype(float)
                median_other = np.median(other_spectra, axis=0)
                std_other = np.std(other_spectra, axis=0) + 1e-8
                
                # Odchylenie od mediany innych cylindrów
                for i in range(21):
                    features[f'engine_delta_{i}'] = spectrum[i] - median_other[i]
                    features[f'engine_zscore_{i}'] = (spectrum[i] - median_other[i]) / std_other[i]
                
                features['engine_max_delta'] = np.max(np.abs(spectrum - median_other))
                features['engine_mean_zscore'] = np.mean(np.abs((spectrum - median_other) / std_other))
                features['engine_distance'] = np.sqrt(np.sum((spectrum - median_other)**2))
            else:
                # Brak innych cylindrów - wypełnij zerami
                for i in range(21):
                    features[f'engine_delta_{i}'] = 0
                    features[f'engine_zscore_{i}'] = 0
                features['engine_max_delta'] = 0
                features['engine_mean_zscore'] = 0
                features['engine_distance'] = 0
        except KeyError:
            # Silnik nie znaleziony - wypełnij zerami
            for i in range(21):
                features[f'engine_delta_{i}'] = 0
                features[f'engine_zscore_{i}'] = 0
            features['engine_max_delta'] = 0
            features['engine_mean_zscore'] = 0
            features['engine_distance'] = 0
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Liczba cech
    feature_cols = [c for c in features_df.columns if c not in ['engine_id', 'cylinder', 'n_cylinders']]
    print(f"   ✅ Wygenerowano {len(feature_cols)} cech")
    
    return features_df

# ======================
# 🤖 MODELE
# ======================

def create_model(model_type, hyperparams):
    """Tworzy model na podstawie typu"""
    if model_type == 'xgboost':
        return xgb.XGBClassifier(**hyperparams)
    elif model_type == 'extratrees':
        return ExtraTreesClassifier(**hyperparams)
    elif model_type == 'randomforest':
        return RandomForestClassifier(**hyperparams)
    elif model_type == 'lightgbm':
        params = {k: v for k, v in hyperparams.items() if k != 'eval_metric'}
        return lgb.LGBMClassifier(**params)
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")

def compare_models(X, y, groups, n_folds=5):
    """Porównuje różne modele"""
    print(f"\n🏆 Porównywanie modeli ({n_folds}-fold CV)...")
    
    models = {
        'XGBoost': ('xgboost', HYPERPARAMS['xgboost']),
        'ExtraTrees': ('extratrees', HYPERPARAMS['extratrees']),
        'RandomForest': ('randomforest', HYPERPARAMS['randomforest']),
        'LightGBM': ('lightgbm', HYPERPARAMS['lightgbm'])
    }
    
    results = {}
    gkf = GroupKFold(n_splits=n_folds)
    
    for name, (model_type, params) in models.items():
        print(f"\n   Trening {name}...")
        model = create_model(model_type, params)
        
        scores = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring='f1_macro', n_jobs=-1)
        mean_score = scores.mean()
        std_score = scores.std()
        
        results[name] = {
            'model_type': model_type,
            'params': params,
            'f1_macro': mean_score,
            'std': std_score,
            'scores': scores
        }
        
        print(f"   ✅ {name}: F1 Macro = {mean_score:.4f} (+/- {std_score:.4f})")
    
    # Sortuj wyniki
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)
    
    print(f"\n📊 Ranking:")
    for rank, (name, res) in enumerate(sorted_results, 1):
        print(f"   {rank}. {name}: {res['f1_macro']:.4f}")
    
    return results, sorted_results[0][0]  # Zwróć najlepszy model

def train_ensemble(X, y, groups, results):
    """Trenuje ensemble najlepszych modeli"""
    print(f"\n🔗 Trenowanie ensemble...")
    
    # Wybierz 3 najlepsze modele
    top_models = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)[:3]
    
    ensemble_models = {}
    weights = {}
    
    for name, res in top_models:
        model_type = res['model_type']
        params = res['params']
        
        print(f"   Dodaję {name} do ensemble...")
        model = create_model(model_type, params)
        model.fit(X, y)
        ensemble_models[name] = model
        
        # Wagi z konfiguracji lub proporcjonalne do F1
        if name.lower() in HYPERPARAMS['ensemble_weights']:
            weights[name] = HYPERPARAMS['ensemble_weights'][name.lower()]
        else:
            weights[name] = res['f1_macro']
    
    # Normalizuj wagi
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    print(f"   ✅ Wagi ensemble: {weights}")
    
    return ensemble_models, weights

def predict_ensemble(models, weights, X):
    """Predykcja ensemble"""
    probs = None
    
    for name, model in models.items():
        model_prob = model.predict_proba(X)
        weight = weights[name]
        
        if probs is None:
            probs = model_prob * weight
        else:
            probs += model_prob * weight
    
    return probs

def detect_unknown(probs, X, threshold=0.35):
    """Detekcja klasy unknown"""
    print(f"\n🔍 Detekcja unknown (threshold={threshold})...")
    
    # Max prawdopodobieństwo
    max_probs = np.max(probs, axis=1)
    
    # Anomaly score z Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    anomaly_scores = -iso_forest.fit_predict(X)  # 1 = anomalia, 0 = normalne
    anomaly_scores = anomaly_scores.astype(float)
    
    # Połącz oba sygnały
    unknown_mask = (max_probs < threshold) | (anomaly_scores > 0.5)
    
    unknown_count = unknown_mask.sum()
    print(f"   ✅ Wykryto {unknown_count} unknown ({100*unknown_count/len(unknown_mask):.1f}%)")
    
    return unknown_mask, anomaly_scores

def train_severity_models(X, y_labels, y_severity, label_encoder):
    """Trenuje osobne modele severity dla każdej usterki"""
    print(f"\n📈 Trenowanie modeli severity...")
    
    # Tylko dla usterek (nie ok, nie unknown)
    fault_labels = ['zakoksowany', 'lejacy', 'pompa', 'iglica']
    severity_models = {}
    
    for label in fault_labels:
        label_idx = label_encoder.transform([label])[0]
        mask = (y_labels == label_idx)
        
        if mask.sum() < 10:
            print(f"   ⚠️ Za mało próbek dla {label} ({mask.sum()}), pomijam")
            continue
        
        X_fault = X[mask]
        y_sev = y_severity[mask]
        
        # Map severity na liczby
        sev_map = {'male': 0, 'srednie': 1, 'duze': 2}
        y_sev_num = np.array([sev_map.get(s, 0) for s in y_sev])
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_fault, y_sev_num)
        severity_models[label] = model
        
        print(f"   ✅ {label}: {mask.sum()} próbek")
    
    return severity_models

def predict_severity(X, y_labels_pred, severity_models, label_encoder):
    """Predykcja severity"""
    severity_map = {0: 'male', 1: 'srednie', 2: 'duze'}
    predictions = []
    
    inverse_label_encoder = {i: l for l, i in enumerate(label_encoder.classes_)}
    
    for i, label_idx in enumerate(y_labels_pred):
        label = inverse_label_encoder.get(label_idx, 'ok')
        
        if label in ['ok', 'unknown']:
            predictions.append('nie_dotyczy')
        elif label in severity_models:
            sev_idx = severity_models[label].predict([X[i]])[0]
            predictions.append(severity_map.get(sev_idx, 'male'))
        else:
            predictions.append('nie_dotyczy')
    
    return predictions

# ======================
# 📊 EWALUACJA
# ======================

def evaluate(y_true, y_pred, y_severity_true, y_severity_pred):
    """Oblicza metryki konkursowe"""
    f1_macro = f1_score(y_true, y_pred, average='macro')
    severity_acc = accuracy_score(y_severity_true, y_severity_pred)
    
    raw_score = 0.75 * f1_macro + 0.25 * severity_acc
    
    print(f"\n📊 Wyniki:")
    print(f"   F1 Macro: {f1_macro:.4f}")
    print(f"   Severity Accuracy: {severity_acc:.4f}")
    print(f"   RAW SCORE: {raw_score:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return f1_macro, severity_acc, raw_score

# ======================
# 💾 ZAPIS I ODCZYT
# ======================

def save_models(models, severity_models, label_encoder, scaler, filepath='models/engin_model.pkl'):
    """Zapisuje wszystkie modele"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data = {
        'models': models,
        'severity_models': severity_models,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'hyperparams': HYPERPARAMS
    }
    
    joblib.dump(data, filepath)
    print(f"\n💾 Modele zapisane do: {filepath}")

def load_models(filepath='models/engin_model.pkl'):
    """Wczytuje modele"""
    data = joblib.load(filepath)
    print(f"\n💾 Modele wczytane z: {filepath}")
    return data

# ======================
# 🎯 GŁÓWNY PIPELINE
# ======================

def main():
    print("="*60)
    print("🚀 ENGIN HACKATHON - KOMPLETNY PIPELINE")
    print("="*60)
    
    # 1. Wczytaj dane
    train_df, val_df, test_df = load_data('data')
    
    # 2. Interpoluj braki
    print("\n🔧 Interpolacja brakujących danych...")
    val_df = interpolate_missing(val_df)
    test_df = interpolate_missing(test_df)
    
    # 3. Feature Engineering
    val_features = extract_features(val_df)
    test_features = extract_features(test_df)
    
    # 4. Przygotuj dane do treningu
    feature_cols = [c for c in val_features.columns if c not in ['engine_id', 'cylinder', 'n_cylinders']]
    
    X = val_features[feature_cols].values
    y_labels = val_df['label'].values
    y_severity = val_df['severity'].values
    groups = val_features['engine_id'].values
    
    # Label encoder
    label_encoder = LabelEncoder()
    y_labels_encoded = label_encoder.fit_transform(y_labels)
    
    # Skalowanie
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n📊 Dane:")
    print(f"   Próbki: {len(X)}")
    print(f"   Cechy: {len(feature_cols)}")
    print(f"   Klasy: {len(label_encoder.classes_)}")
    print(f"   Klasy: {label_encoder.classes_}")
    
    # 5. Porównaj modele
    results, best_model_name = compare_models(X_scaled, y_labels_encoded, groups, HYPERPARAMS['n_folds'])
    
    # 6. Trenuj ensemble
    ensemble_models, ensemble_weights = train_ensemble(X_scaled, y_labels_encoded, groups, results)
    
    # 7. Trenuj modele severity
    severity_models = train_severity_models(X_scaled, y_labels_encoded, y_severity, label_encoder)
    
    # 8. Zapisz modele
    save_models(ensemble_models, severity_models, label_encoder, scaler)
    
    # 9. Predykcja na validation (ewaluacja)
    print(f"\n🎯 Predykcja na validation...")
    probs_val = predict_ensemble(ensemble_models, ensemble_weights, X_scaled)
    y_pred_val = np.argmax(probs_val, axis=1)
    
    # Detekcja unknown
    unknown_mask_val, anomaly_scores_val = detect_unknown(probs_val, X_scaled, HYPERPARAMS['unknown_threshold'])
    y_pred_final_val = y_pred_val.copy()
    y_pred_final_val[unknown_mask_val] = label_encoder.transform(['unknown'])[0]
    
    # Predykcja severity
    y_severity_pred_val = predict_severity(X_scaled, y_pred_final_val, severity_models, label_encoder)
    
    # Ewaluacja
    print("\n" + "="*60)
    print("📊 EWALUACJA NA VALIDATION")
    print("="*60)
    f1, sev_acc, raw = evaluate(y_labels_encoded, y_pred_final_val, y_severity, y_severity_pred_val)
    
    # 10. Predykcja na test
    print(f"\n🎯 Predykcja na test...")
    X_test = test_features[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    
    probs_test = predict_ensemble(ensemble_models, ensemble_weights, X_test_scaled)
    y_pred_test = np.argmax(probs_test, axis=1)
    
    # Detekcja unknown
    unknown_mask_test, anomaly_scores_test = detect_unknown(probs_test, X_test_scaled, HYPERPARAMS['unknown_threshold'])
    y_pred_final_test = y_pred_test.copy()
    y_pred_final_test[unknown_mask_test] = label_encoder.transform(['unknown'])[0]
    
    # Predykcja severity
    y_severity_pred_test = predict_severity(X_test_scaled, y_pred_final_test, severity_models, label_encoder)
    
    # Generuj predictions.csv
    predictions_df = pd.DataFrame({
        'engine_id': test_features['engine_id'],
        'cylinder': test_features['cylinder'],
        'label': label_encoder.inverse_transform(y_pred_final_test),
        'severity': y_severity_pred_test,
        'confidence': np.max(probs_test, axis=1),
        'anomaly_score': anomaly_scores_test
    })
    
    predictions_df.to_csv('reports/predictions.csv', index=False)
    print(f"\n✅ predictions.csv zapisany: {len(predictions_df)} predykcji")
    
    # Podsumowanie
    print("\n" + "="*60)
    print("🎉 PODSUMOWANIE")
    print("="*60)
    print(f"   Najlepszy model: {best_model_name}")
    print(f"   Ensemble: {list(ensemble_models.keys())}")
    print(f"   F1 Macro (val): {f1:.4f}")
    print(f"   Severity Acc (val): {sev_acc:.4f}")
    print(f"   RAW SCORE (val): {raw:.4f}")
    print(f"   Predykcje test: {len(predictions_df)}")
    print(f"   Unknown w test: {unknown_mask_test.sum()} ({100*unknown_mask_test.sum()/len(unknown_mask_test):.1f}%)")
    
    print("\n✅ PIPELINE ZAKOŃCZONY SUKCESEM!")

if __name__ == '__main__':
    main()
