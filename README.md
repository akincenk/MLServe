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

Tests check that training is reproducible (identical probabilities across runs), held-out accuracy is at least 0.85 (typical is 0.93 or higher; the margin keeps library upgrades from making the test flaky), nine canonical samples (textbook Iris rows and per-class means) are classified correctly through the API, and malformed input (missing, negative, out-of-range, non-numeric, unknown fields) is rejected. Any warning fails the test run. CI runs ruff, `pip-audit` and pytest on Python 3.11 and 3.12, and builds the Docker image. Dependabot opens weekly update PRs for pip packages and GitHub Actions.

## Limitations / next steps

- One fixed toy model; there is no model registry, versioning or hot reload.
- No batch endpoint, authentication or rate limiting (see Security).
- For a model that is expensive to train, save it with `skops` or ONNX at build time and load it at startup instead of retraining.

## Design decisions

- **Train at startup instead of loading a file.** Iris trains in well under a second, and not loading a pickle removes a code-execution risk and a build artifact to manage.
- **Fixed seed and stratified split.** Every process serves the same model and reports the same held-out accuracy, which makes the tests exact.
- **Named fields instead of a feature list.** `{"sepal_length": ...}` is self-documenting and lets Pydantic validate each value; `extra: forbid` catches typos.
- **Sync handler.** `/predict` is a plain `def`, so the scikit-learn call runs in FastAPI's thread pool rather than blocking the event loop.
- **Exact dependency pins.** NumPy is pinned per Python version because NumPy 2.5 dropped Python 3.11.

## Security

- The API only reads input and returns a prediction: it stores nothing, calls no paid service and holds no secrets, so it has no authentication. Put it behind a gateway with auth and rate limiting before exposing it publicly.
- Input is strictly validated (types, ranges, no extra fields); bad requests get `422` before reaching the model.
- No model file is deserialised, so there is no pickle attack surface.
- The Docker image runs as a non-root user.

## License

MIT, see [LICENSE](LICENSE).
