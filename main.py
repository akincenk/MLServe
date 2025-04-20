# MLServe - Machine Learning Model Deployment API

"""
A production-ready API for serving machine learning models.
Users can send feature data via a POST request and receive predictions in real-time.
Includes model training, saving, loading, and an interactive /predict endpoint.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Load model at startup
MODEL_PATH = "model.pkl"
if not os.path.exists(MODEL_PATH):
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier()
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)

model = joblib.load(MODEL_PATH)

# Define input schema
class PredictRequest(BaseModel):
    features: list[float]  # Must match model input shape

class PredictResponse(BaseModel):
    prediction: int

app = FastAPI(title="MLServe API", description="Predict using deployed ML model", version="1.0")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        X = np.array(request.features).reshape(1, -1)
        pred = model.predict(X)[0]
        return {"prediction": int(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to MLServe API. Use /predict to make predictions."}
