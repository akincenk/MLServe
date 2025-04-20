# MLServe

MLServe is a lightweight FastAPI app that allows you to serve and interact with a machine learning model using REST endpoints. It loads (or trains) a RandomForest model on the Iris dataset and makes predictions via a `/predict` API.

## Features

- Loads or trains an ML model (RandomForestClassifier)
- Accepts feature input as JSON
- Returns prediction from deployed model
- Includes Swagger UI

## Stack

- FastAPI
- scikit-learn
- joblib
- Docker (optional)

## How to Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Or with Docker

```bash
docker build -t mlserve .
docker run -p 8000:8000 mlserve
```

## Example API Call

```http
POST /predict
Content-Type: application/json

{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Returns:

```json
{
  "prediction": 0
}
```

---

© 2025
