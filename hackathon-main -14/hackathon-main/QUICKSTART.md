# 🚀 Quick Start - ENGIN Hackathon

## Najszybszy sposób na uruchomienie

### Krok 1: Instalacja zależności
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn tqdm joblib scipy
```

### Krok 2: Uruchomienie pipeline
```bash
python run_complete.py
```

**To wszystko!** Skrypt:
- ✅ Wczyta dane z `data/`
- ✅ Wygeneruje 92 cechy
- ✅ Porówna 4 modele (XGBoost, ExtraTrees, RandomForest, LightGBM)
- ✅ Wytrenuje ensemble
- ✅ Wykryje unknown
- ✅ Przewidzi severity
- ✅ Zapisze model do `models/engin_model.pkl`
- ✅ Wygeneruje `reports/predictions.csv`

### Krok 3 (opcjonalny): Aplikacja
```bash
streamlit run app/app.py -- --data-dir data --model-path models/engin_model.pkl
```

## 📊 Oczekiwane wyniki

```
🏆 Ranking modeli:
   1. XGBoost: F1 Macro = 0.9336
   2. LightGBM: F1 Macro = 0.8960
   3. ExtraTrees: F1 Macro = 0.8349
   4. RandomForest: F1 Macro = 0.8198

📊 EWALUACJA NA VALIDATION:
   F1 Macro: 0.8252
   Severity Accuracy: 0.8526
   RAW SCORE: 0.8389

✅ predictions.csv zapisany: 600 predykcji
```

## 📁 Struktura po uruchomieniu

```
/workspace/
├── run_complete.py          ← GŁÓWNY SKRYPT (samowystarczalny)
├── README.md                ← Pełna dokumentacja
├── QUICKSTART.md            ← Ten plik
├── requirements.txt         ← Zależności
├── data/                    ← Dane wejściowe
│   ├── train.csv           # 240 silników (bez etykiet)
│   ├── val.csv             # 40 silników (z etykietami)
│   └── test.csv            # 50 silników (do predykcji)
├── models/                  ← Wyjście
│   └── engin_model.pkl     # Wytrenowany model (5.2 MB)
├── reports/                 ← Wyjście
│   └── predictions.csv     # Predykcje (600 wierszy)
└── app/                     ← Aplikacja
    └── app.py              # Dashboard Streamlit
```

## 🔧 Rozwiązywanie problemów

### Brak plików CSV
```bash
ls data/
# Jeśli puste, pobierz z repozytorium GitHub
```

### Błąd importu
```bash
pip install -r requirements.txt
```

### Za mało pamięci
Zmniejsz liczbę foldów w `run_complete.py`:
```python
HYPERPARAMS = {
    ...
    'n_folds': 3  # zamiast 5
}
```

## 📈 Co dalej?

1. **Sprawdź predictions.csv**: `head reports/predictions.csv`
2. **Uruchom aplikację**: `streamlit run app/app.py ...`
3. **Eksperymentuj**: zmień hiperparametry w `run_complete.py`
4. **Dodaj semi-supervised**: wykorzystaj `train.csv` (240 silników bez etykiet)

Powodzenia! 🎉
