# Aesteel Engine Diagnostics

Panel diagnostyczny dla akustycznych pomiarów wtrysku silników Diesla. Lekki frontend HTML/CSS/JS + FastAPI + XGBoost, z osobnym sygnałem anomalii IsolationForest i wyjaśnieniem dla mechanika.

## Uczenie bez labelków

`train.csv` nie ma etykiet, `val.csv` ma etykiety, a `test.csv` jest bez etykiet. Produkcyjny pipeline nie używa `test.csv` do fitowania, strojenia ani wyboru progów.

1. Feature engineering i skalowanie fitowane są na `train.csv`.
2. `val.csv` jest tylko semantycznym bankiem prototypów do pseudo-labelowania train.
3. Analiza feature importance jest wykonywana przed redukcją cech.
4. Finalny XGBoost dla `label` uczy się wyłącznie na pseudo-labelach wierszy `train.csv`.
5. Drugi, niezależny XGBoost przewiduje `severity` tylko dla klas `zakoksowany`, `lejacy`, `pompa`, `iglica`.
6. Dla `ok` i `unknown` severity jest zawsze `nie_dotyczy`.
7. Zachowujemy 32 najmocniejsze cechy.
8. IsolationForest jest niezależnym detektorem anomalii fitowanym wyłącznie na train.
9. `test.csv` służy dopiero do finalnego inference.

## Model

Klasyfikacja `label`: XGBoost `hist` z konserwatywną regularizacją:

- `n_estimators=360`
- `max_depth=4`
- `learning_rate=0.035`
- `subsample=0.85`
- `colsample_bytree=0.80`
- `min_child_weight=5`
- `reg_alpha=0.20`
- `reg_lambda=3.0`
- `gamma=0.05`
- `max_bin=128`

`severity` ma osobny XGBoost. IsolationForest (`240` drzew, `contamination=auto`) jest niezależnym sygnałem anomalii i nie zastępuje klasyfikatorów.

## Analiza cech

Przed treningiem finalnego modelu wykonywana jest analiza importance na pełnym zestawie cech. Ranking zapisuje się w artefakcie pipeline i jest dostępny przez `GET /api/features`. Finalny model używa 32 najwyżej ocenionych cech.

## Explainability

Każdy wiersz predykcji zawiera confidence, anomaly score, anomalne pasmo `mV_i`, odchylenie względem pozostałych cylindrów tego silnika, top features oraz krótką wskazówkę inspekcyjną.

## Installation

Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uruchomienie — pełna instrukcja

**Terminal 1 — backend:**

```bash
python -m backend
```

Przy pierwszym starcie backend automatycznie zbuduje `models/xgboost_model.pkl`, jeżeli artefakt nie istnieje albo jest starszego typu. Budowanie czyta `train.csv` + `val.csv`, nigdy `test.csv`.

**Terminal 2 — frontend:**

```bash
python serve_frontend.py
```

Frontend automatycznie szuka wolnego portu, zaczynając od `8080` i próbując kolejnych 20 portów. Adres jest wypisywany w terminalu, np. `http://127.0.0.1:8081`. Możesz wymusić początkowy port przez `FRONTEND_PORT=9000`.

API działa domyślnie na `http://127.0.0.1:8000`.

### Notebook: trening + walidacja + test prediction

Uruchom Jupyter:

```bash
jupyter lab
```

Następnie otwórz:

`notebooks/train_validate_predict.ipynb`

Notebook:
- dzieli `val.csv` na semantic reference i niewidziany hold-out evaluation;
- analizuje feature importance przed redukcją;
- trenuje oba XGBoost-y oraz IsolationForest zgodnie z zasadami projektu;
- drukuje label accuracy, Macro-F1, severity accuracy i **RAW SCORE**;
- generuje predykcje dla `test.csv`;
- zapisuje `outputs/test_predictions.csv`;
- zapisuje artefakt `models/xgboost_model_notebook.pkl`.

### Alternatywnie: ręczne trenowanie

```bash
python train_semisupervised.py
```

Następnie:

```bash
python -m backend
```

### Test API

```bash
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/model
curl -X POST -F "file=@data/test.csv" http://127.0.0.1:8000/api/diagnose
```

`/api/diagnose` zwraca wszystkie wiersze, nie tylko pierwszy.

## Funkcje aplikacji

- batch prediction dla całego CSV;
- fault/severity/anomaly distribution;
- filtrowanie wyników;
- eksport JSON;
- feature importance;
- klikany wynik cylindra i panel wyjaśnienia;
- porównanie widma cylindra z pozostałymi cylindrami tego samego silnika;
- sygnał anomalii niezależny od klasyfikacji;
- lokalny inference CPU;
- automatyczny wybór wolnego portu frontendu.

## Performance

Frontend jest bez frameworka i bez ciężkich bibliotek wykresowych. Backend ładuje model raz i trzyma go w pamięci. XGBoost używa histogramowego tree buildera i wielowątkowości CPU, a inference odbywa się batchowo.

## Troubleshooting

**API offline:**
```bash
python -m backend
```

**Stary model po zmianie kodu:** usuń `models/xgboost_model.pkl` i uruchom backend ponownie. Zostanie odbudowany.

**Frontend:** nie zakłada już, że port `8080` jest wolny — `serve_frontend.py` automatycznie wybierze następny wolny port.

**Invalid CSV:** plik musi zawierać `engine_id`, `cylinder` oraz `mV_0` … `mV_20`. Braki widma są czyszczone przez `clean_spectrum`.

**Frontend nie widzi API:** upewnij się, że backend działa na `127.0.0.1:8000`. Frontend wyprowadza host API z aktualnego hosta strony i używa portu `8000`, chyba że ustawiono `window.AESTEEL_API`.

## Competition output

Do finalnego scoringu używaj wymaganego formatu `reports/predictions.csv`. Nie trenuj na hidden test labels.
