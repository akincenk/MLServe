# MLServe

[![CI](https://github.com/akincenk/MLServe/actions/workflows/ci.yml/badge.svg)](https://github.com/akincenk/MLServe/actions/workflows/ci.yml)

A minimal example of serving a scikit-learn model with FastAPI: an Iris species classifier with validated input and a JSON prediction endpoint.

## What it does

- On startup, trains a `RandomForestClassifier` on scikit-learn's bundled Iris dataset (80/20 stratified split, fixed seed 42), so every run serves the same model. Held-out accuracy is reported at `GET /model`.
- `POST /predict` takes the four flower measurements (cm) as named fields and returns the predicted species plus per-class probabilities.
- Input is validated with Pydantic: all four fields required, numeric, in `(0, 30]`, no extra fields. Invalid input gets `422` with details.

Training happens in memory (well under a second) rather than loading a pickle, so the service never deserialises an untrusted model file.

## Stack

Python 3.11+, FastAPI, Pydantic 2, scikit-learn, NumPy. All code is in `main.py`; tests are in `tests/`.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs. No configuration or environment variables are needed.

### Docker

```bash
docker build -t mlserve .
docker run --rm -p 8000:8000 mlserve
```

The container runs as a non-root user.

## API

| Method | Path | Description |
| --- | --- | --- |
| POST | `/predict` | Predict species from four measurements |
| GET | `/model` | Algorithm, features, classes, held-out accuracy, seed |
| GET | `/health` | Liveness check |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
# {"class_id": 0, "species": "setosa", "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}}
```

## Tests

```bash
pip install -r requirements-dev.txt
ruff check . && pytest
```

Tests check that training is reproducible (identical probabilities across runs), held-out accuracy is at least 0.9, known samples are classified correctly, and malformed input (missing, negative, out-of-range, non-numeric, unknown fields) is rejected. CI runs them on Python 3.11 and 3.12.

## Limitations / next steps

- One fixed toy model; there is no model registry, versioning or hot reload.
- No batch endpoint, authentication or rate limiting.
- For a model that is expensive to train, save it with `skops` or ONNX at build time and load it at startup instead of retraining.

## License

MIT, see [LICENSE](LICENSE).
