# F1 Race Prediction & Strategy Recommendation System

A production-ready machine-learning system that predicts **Formula 1 race outcomes** and provides **pit-stop strategy recommendations**, complete with a REST API, DVC-managed data/model pipelines, and CI/CD automation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [DVC Pipeline](#dvc-pipeline)
- [API Reference](#api-reference)
- [Docker](#docker)
- [CI/CD Pipelines](#cicd-pipelines)
- [Configuration](#configuration)
- [Testing](#testing)

---

## Overview

| Feature | Details |
|---|---|
| **Race Outcome Prediction** | Gradient Boosting classifier – predicts probability of a driver finishing in the top 3 |
| **Position Regression** | Random Forest regressor – predicts expected finish position |
| **Strategy Recommendation** | Rule + ML hybrid – recommends 1-stop / 2-stop / undercut strategies |
| **Data Source** | [Ergast F1 API](http://ergast.com/mrd/) (historical seasons 2018–2023) |
| **Data Versioning** | [DVC](https://dvc.org/) – tracks datasets, feature files, and model artifacts |
| **REST API** | [FastAPI](https://fastapi.tiangolo.com/) with automatic OpenAPI docs at `/docs` |
| **Containerisation** | Multi-stage Docker build; `docker-compose` for local deployment |
| **CI/CD** | GitHub Actions – lint → test → build Docker image on every push |

---

## Architecture

```
Ergast API
    │
    ▼
data_ingestion.py ──► data/raw/
    │
    ▼
preprocessing.py  ──► data/processed/{train,val,test}.csv
    │
    ▼
feature_engineering.py ──► data/processed/{split}_features.csv
    │
    ├──► train.py ──► models/{race_predictor,strategy_recommender,scaler}.pkl
    │
    └──► evaluate.py ──► metrics/evaluation.json
                              │
                              ▼
                    FastAPI app (app/main.py)
                    /predict  /strategy  /health
```

---

## Project Structure

```
F1-Race-Prediction/
├── .github/
│   └── workflows/
│       ├── ci.yml          # Lint → Test → Build Docker
│       ├── cd.yml          # Push Docker image to GHCR on main/tags
│       └── dvc.yml         # Run DVC pipeline on data/code changes
├── app/
│   └── main.py             # FastAPI REST API
├── src/
│   ├── data_ingestion.py   # Fetch from Ergast API
│   ├── preprocessing.py    # Clean, merge, split data
│   ├── feature_engineering.py  # Feature creation
│   ├── train.py            # Model training
│   ├── evaluate.py         # Model evaluation & metrics
│   ├── predict.py          # Inference wrapper
│   └── strategy.py         # Strategy recommendation engine
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_predict.py
│   └── test_strategy.py
├── data/                   # Managed by DVC
│   ├── raw/
│   └── processed/
├── models/                 # Managed by DVC
├── metrics/                # DVC metrics output
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Hyperparameters & config
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the full DVC pipeline

```bash
dvc repro
```

This will:
1. Fetch F1 data from the Ergast API → `data/raw/`
2. Preprocess and split the data → `data/processed/`
3. Engineer features → `data/processed/*_features.csv`
4. Train models → `models/*.pkl`
5. Evaluate and write metrics → `metrics/evaluation.json`

### 3. Start the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open <http://localhost:8000/docs> for interactive API documentation.

---

## DVC Pipeline

The pipeline is defined in `dvc.yaml` with parameters in `params.yaml`.

```bash
# Reproduce the full pipeline
dvc repro

# Show metrics
dvc metrics show

# Show DAG
dvc dag

# Pull data/models from remote (after configuring a DVC remote)
dvc pull
```

### Configuring a DVC remote

```bash
# Example: S3
dvc remote add -d myremote s3://my-bucket/f1-dvc
dvc remote modify myremote region us-east-1
```

---

## API Reference

### `GET /health`
Returns API status.

### `POST /predict`
Predict race outcome for a single driver entry.

**Request body:**
```json
{
  "driver_id_enc": 5,
  "constructor_id_enc": 3,
  "circuit_id_enc": 12,
  "grid": 3,
  "quali_position": 3,
  "total_points": 320.0,
  "wins": 11.0,
  "podiums": 17.0,
  "total_points_constructor": 570.0,
  "wins_constructor": 13.0,
  "reliability": 0.95,
  "driver_rolling_position": 2.5,
  "driver_rolling_points": 22.0,
  "constructor_rolling_position": 2.0,
  "constructor_rolling_points": 24.0
}
```

**Response:**
```json
{
  "top3_probability": 0.8421,
  "predicted_position": 2.14,
  "is_top3": true
}
```

### `POST /strategy`
Get ranked pit-stop strategy recommendations.

**Request body:**
```json
{
  "total_laps": 57,
  "current_position": 4,
  "safety_car_probability": 0.2,
  "current_lap": 1,
  "leader_pit_lap": 22
}
```

**Response:** List of `StrategyRecommendation` objects with stints, pit-stop count, estimated time loss, and rationale.

---

## Docker

### Build and run locally

```bash
# Build
docker build -t f1-race-prediction:latest .

# Run (mount pre-trained models)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  f1-race-prediction:latest
```

### Using docker-compose

```bash
docker-compose up --build
```

---

## CI/CD Pipelines

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | Every push / PR | Lint (black + flake8) → Test (pytest + coverage) → Build Docker image |
| `cd.yml` | Push to `main` / version tag | Build & push Docker image to GitHub Container Registry |
| `dvc.yml` | Push to `main` when `src/` or `params.yaml` change | Run `dvc repro`, upload metrics |

---

## Configuration

All hyperparameters are in `params.yaml`:

```yaml
model:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05
  ...

strategy:
  soft_deg_rate: 0.12
  medium_deg_rate: 0.07
  hard_deg_rate: 0.04
  max_stint_soft: 25
  ...
```

---

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_strategy.py -v
```

Tests cover:
- **Preprocessing** – data cleaning, encoding, splits, rolling features
- **Feature Engineering** – interaction features, build pipeline
- **Prediction** – `RaceEntry` derived fields, `RacePredictor` with mocked models
- **Strategy** – degradation model, one-stop / two-stop / undercut, `StrategyRecommender`