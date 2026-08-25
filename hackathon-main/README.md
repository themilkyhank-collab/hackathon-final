# Aesteel Engine Diagnostics

Lekki, lokalny system diagnostyczny dla przemysłowych silników Diesla. Aplikacja analizuje widmo pomiaru dla każdego cylindra i zwraca diagnozę (`label`), osobny stopień nasilenia (`severity`), sygnał anomalii oraz krótkie wyjaśnienie wyniku.

Frontend to HTML/CSS/JS, backend działa na FastAPI, a produkcyjne klasyfikatory korzystają z RBF SVC.

## Dane konkursowe

- `data/train.csv` — 240 silników, dane bez etykiet; jedyny zbiór, na którym fitowane są produkcyjne estymatory.
- `data/val.csv` — 40 silników z pełnymi etykietami; używany jako oznaczony bank referencyjny do konstrukcji pseudo-labeli/pseudo-severity oraz do walidacji strojenia.
- `data/test.csv` — 50 silników bez etykiet; nigdy nie jest używany do treningu ani tuningu.
- `predictions.csv` — plik wymagany przy oddaniu projektu; generuje go skrypt dopiero po jawnej komendzie `--output`.

Nie commitujemy ręcznie wygenerowanych danych testowych ani odpowiedzi referencyjnych.

## Model

### `label`

```text
features = first derivative (d1)
StandardScaler
RBF SVC
C = 1.0
gamma = 0.0175
class_weight = balanced
```

`d1` daje 20 cech. `StandardScaler` jest fitowany wyłącznie na `train.csv`.

### `severity`

Osobny RBF SVC dla czterech klas usterek (`zakoksowany`, `lejacy`, `pompa`, `iglica`) i trzech poziomów (`male`, `srednie`, `duze`). Dla `ok` i `unknown` wynik to `severity=nie_dotyczy`.

Aktualne hiperparametry:

```text
C = 10.0
gamma = 0.03
class_weight = balanced
```

Search używa `GroupKFold` po `engine_id`. Aktualny zapis strojenia znajduje się w `models/severity_hyperparams.json`.

### Anomaly detection

`IsolationForest` jest niezależnym sygnałem pomocniczym i nie wpływa na uczenie SVC.

## Protokół uczenia i leakage

- `train.csv` jest jedynym zbiorem, na którym fitowane są produkcyjne estymatory.
- `val.csv` jest używany jako oznaczony reference bank do nadawania pseudo-labeli/pseudo-severity rekordom treningowym oraz do strojenia/walidacji.
- `test.csv` nie jest ładowany przez trening ani tuner.
- `StandardScaler` jest fitowany na danych treningowych, a potem tylko transformuje kolejne zbiory.
- Grupowanie po `engine_id` jest używane w walidacji, aby cylindry jednego silnika nie trafiały do różnych foldów.

## Uruchomienie — Windows, jedno kliknięcie

1. Zainstaluj Python 3.11+.
2. Kliknij dwukrotnie **`start.bat`**.

Launcher nie tworzy virtual environment. Korzysta z istniejącego Pythona i:

- sprawdza wymagane importy;
- instaluje `requirements.txt` tylko wtedy, gdy czegoś brakuje;
- podczas instalacji używa `--no-cache-dir`;
- uruchamia API na `127.0.0.1:8000`;
- uruchamia lokalny frontend;
- wybiera wolny port od `8080`;
- otwiera przeglądarkę automatycznie.

## Testowanie modelu i Raw_Score

Do reprodukowalnego testowania służy:

```bash
python scripts/test_model.py data/test.csv
```

Jeśli wejściowy CSV zawiera `label`, skrypt dodatkowo liczy:

```text
Raw_Score = 0.75 * Macro_F1(label)
          + 0.25 * Accuracy(severity dla prawdziwych faultów)
```

oraz przelicza wynik na maksymalnie 40 punktów według zasad hackathonu.

Jeżeli CSV nie zawiera etykiet, skrypt **nie zgaduje wyniku** i tylko wykonuje predykcję. Nic nie jest zapisywane na dysku bez jawnego `--output`.

Przykład wygenerowania wymaganego pliku oddania:

```bash
python scripts/test_model.py data/test.csv --output predictions.csv
```

Wtedy `predictions.csv` ma dokładnie kolumny:

```text
engine_id,cylinder,label,severity
```

## Strojenie `severity`

```bash
python scripts/tune_severity.py
```

Tuner używa `d1 + StandardScaler + RBF SVC`, testuje `C`, `gamma` i `class_weight`, stosuje `GroupKFold` po `engine_id` i nie czyta `test.csv`.

Notebook do ręcznej reprodukcji searchu:

```text
notebooks/severity_hyperparameter_search.ipynb
```

## Ręczny trening

```bash
python train_semisupervised.py
```

Model produkcyjny zapisuje się jako `models/svc_model.pkl`.

## API

```bash
python -m backend
```

Health check: `http://127.0.0.1:8000/api/health`

Model: `http://127.0.0.1:8000/api/model`

Features: `http://127.0.0.1:8000/api/features`

Predykcja:

```bash
curl -X POST -F "file=@data/test.csv" http://127.0.0.1:8000/api/diagnose
```

## Format wejścia

Wymagane kolumny:

```text
engine_id
cylinder
mV_0 ... mV_20
```

## Testy

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Co jeszcze trzeba przygotować do finalnego oddania

1. Wygenerować `predictions.csv` **z `data/test.csv` przez `scripts/test_model.py --output predictions.csv`** i umieścić go w repo przed oddaniem.
2. Dodać prezentację PDF/slajdy do repozytorium — obecnie repo nie zawiera pliku prezentacji.
3. Przed oddaniem uruchomić `pytest -q` oraz test predykcji na pełnym `test.csv`.
4. Nie używać `test.csv` do wyboru hiperparametrów ani ręcznego poprawiania predykcji.

## Performance / CPU

RBF SVC pracuje na 20 cechach `d1`, frontend nie wymaga ciężkiego frameworka, a inference działa na CPU. Backend działa lokalnie i nie wymaga GPU.

## Struktura

```text
.
├── backend/                  # FastAPI
├── data/                     # train / val / test
├── frontend/                 # HTML/CSS/JS
├── models/                   # konfiguracja severity + lokalne artefakty
├── notebooks/                # eksperymenty i reprodukowalne analizy
├── scripts/                  # narzędzia offline
├── src/                      # pipeline ML
├── tests/                    # testy
├── requirements.txt
├── requirements-dev.txt
├── start.py
├── start.bat
├── serve_frontend.py
└── train_semisupervised.py
```

## Troubleshooting

**`Cache entry deserialization failed`** — launcher używa `pip --no-cache-dir`. Przy ręcznym `pip install` można jednorazowo wykonać `python -m pip cache purge`.

**Port 8080 zajęty** — launcher automatycznie znajdzie kolejny wolny port.

**Port 8000 zajęty** — jeśli działa tam Aesteel API, launcher może użyć istniejącego backendu; jeśli działa inna aplikacja, trzeba ją zatrzymać.
