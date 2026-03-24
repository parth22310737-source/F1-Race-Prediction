# F1 Race Prediction & Strategy Recommendation System

A production-ready machine-learning system that predicts Formula 1 race finish positions and recommends optimal pit-stop strategies, built with a complete MLOps stack: **DVC** for data versioning, **GitHub Actions** for CI/CD, and **Docker** for containerisation.

---

## Architecture

```
F1-Race-Prediction/
├── .github/
│   └── workflows/
│       ├── ci.yml          # Lint → Test → DVC pipeline
│       └── cd.yml          # Train → Docker build & push
├── src/
│   ├── data/
│   │   ├── fetch_data.py   # Fetch from Ergast API or generate synthetic data
│   │   └── preprocess.py   # Clean, encode and scale raw data
│   ├── features/
│   │   └── build_features.py  # Derive win_rate, podium_rate, dnf_flag
│   ├── models/
│   │   ├── train.py        # Train Random Forest classifier
│   │   └── predict.py      # RacePredictor inference wrapper
│   └── strategy/
│       └── recommend.py    # Pit-stop strategy simulator & recommender
├── api/
│   └── app.py              # FastAPI REST service
├── tests/                  # pytest test suite
├── data/
│   ├── raw/                # DVC-tracked raw CSVs
│   └── processed/          # DVC-tracked processed CSVs
├── models/                 # DVC-tracked model artifacts
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Hyperparameters & config
├── Dockerfile              # Multi-stage Docker build
└── docker-compose.yml      # Compose file (api + train services)
```

---

## ML Pipeline

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| `fetch_data` | `src/data/fetch_data.py` | Ergast API / synthetic | `data/raw/races.csv` |
| `preprocess` | `src/data/preprocess.py` | raw CSV | processed CSV + encoders |
| `build_features` | `src/features/build_features.py` | processed CSV | feature CSV |
| `train` | `src/models/train.py` | feature CSV | `models/model.pkl` + metrics |

### Model

A **Random Forest Classifier** predicts the discrete finish position (1–20) for each driver. Features include:

- `grid_position`, `qualifying_time_ms`
- `driver_points`, `constructor_points`, `driver_age`
- `laps_completed`
- `circuit_type`, `weather` (encoded)
- Derived: `win_rate`, `podium_rate`, `dnf_flag`

### Strategy Recommender

A lap-time simulation engine enumerates all valid 1-stop and 2-stop strategies (compound combinations obeying F1's mandatory two-compound rule) and returns the one with the minimum projected race time. Wet/mixed conditions trigger intermediate-tyre logic automatically.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full DVC pipeline

```bash
dvc repro
```

### 3. Start the API

```bash
uvicorn api.app:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs: `http://localhost:8000/docs`.

---

## API Endpoints

### `GET /health`
Liveness probe.

### `POST /predict`
Predict finish positions for a list of drivers.

```json
{
  "drivers": [
    {
      "driver": "Max Verstappen",
      "grid_position": 1,
      "driver_age": 26,
      "constructor_points": 454.0,
      "driver_points": 454.0,
      "laps_completed": 57,
      "qualifying_time_ms": 80123
    }
  ]
}
```

### `POST /strategy`
Recommend the optimal pit-stop strategy.

```json
{
  "total_laps": 57,
  "grid_position": 1,
  "weather": "dry",
  "circuit_type": "mixed"
}
```

### `GET /metrics`
Return last training run metrics.

---

## Docker

```bash
# Start the API
docker-compose up api

# Run training
docker-compose --profile train up train

# Build manually
docker build -t f1-race-prediction .
docker run -p 8000:8000 -v $(pwd)/models:/app/models f1-race-prediction
```

---

## DVC

```bash
dvc repro          # run / reproduce pipeline
dvc metrics show   # print metrics
dvc params diff    # diff params vs last commit
```

---

## CI/CD

### CI (`ci.yml`) — every push / PR
1. Lint (flake8, black, isort)
2. Test (pytest with coverage, Python 3.10 & 3.11)
3. DVC pipeline validation

### CD (`cd.yml`) — push to `main`
1. Train model via DVC
2. Build & push multi-stage Docker image to GHCR

---

## Tests

```bash
pytest tests/ -v --cov=src --cov=api
```

---

## Configuration (`params.yaml`)

```yaml
model:
  n_estimators: 200
  max_depth: 10

strategy:
  soft_tire_life: 25
  pit_stop_time_loss: 22
```

Change any value and run `dvc repro` — only affected stages re-run.
