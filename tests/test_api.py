import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import train_model


@pytest.fixture(scope="module")
def client():
    from main import app

    with TestClient(app) as c:  # runs lifespan -> trains the model
        yield c


SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
VIRGINICA = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2}


def test_training_is_deterministic():
    m1, classes, acc1 = train_model()
    m2, _, acc2 = train_model()
    grid = np.random.default_rng(0).uniform(0.1, 8, size=(200, 4))
    assert classes == ["setosa", "versicolor", "virginica"]
    assert acc1 == acc2
    assert (m1.predict_proba(grid) == m2.predict_proba(grid)).all()


def test_model_is_accurate():
    _, _, acc = train_model()
    assert acc >= 0.9


@pytest.mark.parametrize(("sample", "species"), [(SETOSA, "setosa"), (VIRGINICA, "virginica")])
def test_predict_known_samples(client, sample, species):
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert body["species"] == species
    assert set(body["probabilities"]) == {"setosa", "versicolor", "virginica"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["probabilities"][species] == max(body["probabilities"].values())


@pytest.mark.parametrize(
    "payload",
    [
        {k: v for k, v in SETOSA.items() if k != "petal_width"},  # missing feature
        {**SETOSA, "petal_width": -1},  # negative
        {**SETOSA, "petal_width": 0},  # zero
        {**SETOSA, "petal_width": 1000},  # out of range
        {**SETOSA, "petal_width": "wide"},  # not a number
        {**SETOSA, "colour": 3},  # unknown field
        {"features": [5.1, 3.5, 1.4, 0.2]},  # old list format
    ],
)
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_model_info(client):
    body = client.get("/model").json()
    assert body["features"] == ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assert body["random_seed"] == 42
    assert body["test_accuracy"] >= 0.9


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
