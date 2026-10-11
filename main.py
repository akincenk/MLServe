"""MLServe: serve an Iris species classifier over a FastAPI REST API.

The model is a scikit-learn RandomForest trained at startup on the bundled
Iris dataset with a fixed random seed, so every process serves an identical
model. Training takes well under a second, so nothing is pickled to disk
(and no untrusted pickle is ever loaded).
"""

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SEED = 42
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def train_model() -> tuple[RandomForestClassifier, list[str], float]:
    """Train deterministically; return (model, class names, held-out accuracy)."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=SEED, stratify=iris.target
    )
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))
    return model, [str(name) for name in iris.target_names], accuracy


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.classes, app.state.accuracy = train_model()
    yield


app = FastAPI(title="MLServe", description="Iris species classifier served with FastAPI", lifespan=lifespan)

# Measurements in centimetres. Bounds reject nonsense (negative, absurdly large) input.
_cm = Field(gt=0, le=30)


class IrisFeatures(BaseModel):
    sepal_length: float = _cm
    sepal_width: float = _cm
    petal_length: float = _cm
    petal_width: float = _cm

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]
        },
    }


class Prediction(BaseModel):
    class_id: int
    species: str
    probabilities: dict[str, float]


class ModelInfo(BaseModel):
    algorithm: str
    features: list[str]
    classes: list[str]
    test_accuracy: float
    random_seed: int


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures, request: Request):
    state = request.app.state
    x = np.array([[getattr(features, f) for f in FEATURES]])
    probs = state.model.predict_proba(x)[0]
    class_id = int(np.argmax(probs))
    return Prediction(
        class_id=class_id,
        species=state.classes[class_id],
        probabilities={name: round(float(p), 4) for name, p in zip(state.classes, probs, strict=True)},
    )


@app.get("/model", response_model=ModelInfo)
def model_info(request: Request):
    state = request.app.state
    return ModelInfo(
        algorithm="RandomForestClassifier(n_estimators=100)",
        features=FEATURES,
        classes=state.classes,
        test_accuracy=round(state.accuracy, 4),
        random_seed=SEED,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
