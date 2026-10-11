import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import train_model


@pytest.fixture(scope="module")
def client():
    from main import app

    with TestClient(app) as c:  # runs lifespan -> trains the model
        yield c


def sample(sl, sw, pl, pw):
    return dict(zip(["sepal_length", "sepal_width", "petal_length", "petal_width"], [sl, sw, pl, pw], strict=True))


SETOSA = sample(5.1, 3.5, 1.4, 0.2)
VIRGINICA = sample(6.7, 3.3, 5.7, 2.1)

# Textbook Iris rows (Fisher 1936, as shipped by scikit-learn) and per-class means, all far from
# the class boundaries, so any reasonable model must get them right regardless of random seed.
CANONICAL = [
    (SETOSA, "setosa"),  # row 0
    (sample(4.9, 3.0, 1.4, 0.2), "setosa"),  # row 1
    (sample(5.0, 3.4, 1.5, 0.2), "setosa"),  # class mean
    (sample(7.0, 3.2, 4.7, 1.4), "versicolor"),  # row 50
    (sample(5.7, 2.8, 4.1, 1.3), "versicolor"),  # row 99
    (sample(5.9, 2.8, 4.3, 1.3), "versicolor"),  # class mean
    (sample(6.3, 3.3, 6.0, 2.5), "virginica"),  # row 100
    (VIRGINICA, "virginica"),  # row 124
    (sample(6.6, 3.0, 5.6, 2.0), "virginica"),  # class mean
]


def test_training_is_deterministic():
    m1, classes, acc1 = train_model()
    m2, _, acc2 = train_model()
    grid = np.random.default_rng(0).uniform(0.1, 8, size=(200, 4))
    assert classes == ["setosa", "versicolor", "virginica"]
    assert acc1 == acc2
    assert (m1.predict_proba(grid) == m2.predict_proba(grid)).all()


def test_model_is_accurate():
    # Iris is easy (typical held-out accuracy 0.93-1.0); 0.85 leaves margin so library
    # upgrades that shift the forest slightly do not make the test flaky.
    _, _, acc = train_model()
    assert acc >= 0.85


@pytest.mark.parametrize(("features", "species"), CANONICAL)
def test_predict_canonical_samples(client, features, species):
    r = client.post("/predict", json=features)
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
    assert body["test_accuracy"] >= 0.85


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
