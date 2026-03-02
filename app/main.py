from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from typing import Optional
from pathlib import Path
import pickle

import mlflow.pyfunc
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def load_production_model():
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_PATH = BASE_DIR / "prod_model" / "prod_model.pkl"

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# NOTE: We would theoretically use this function to load the "best model from production" in a real deployment where we can access remote MLflow, 
# but for simplicity in the code above we will just load a specific model.
# def load_production_model(model_name: str):
#     # Ensure tracking URI points to local mlflow.db in project root
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#     db_path = os.path.join(project_root, "mlflow.db")
#     mlflow.set_tracking_uri(f"sqlite:///{db_path}")

#     model_uri = f"models:/{model_name}/Production"
#     logger.info("Loading MLflow model from %s with tracking uri %s", model_uri, mlflow.get_tracking_uri())
    
#     # Use MlflowClient to verify a Production version exists
#     client = MlflowClient()
#     versions = client.search_model_versions(f"name='{model_name}'")
#     prod_versions = [v for v in versions if v.current_stage == "Production"]
#     if not prod_versions:
#         raise RuntimeError(f"No Production model registered for '{model_name}' in tracking store {mlflow.get_tracking_uri()}")

#     return mlflow.pyfunc.load_model(model_uri)


@app.on_event("startup")
def startup_event():
    try:
        app.state.model = load_production_model()
        logger.info(f"Model loaded into app.state.model")
    except Exception as e:
        app.state.model = None
        logger.warning("Could not load model at startup: %s", e)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health():
    loaded = getattr(app.state, "model", None) is not None
    return {"status": "ok", "model_loaded": loaded }


@app.post("/predict")
async def predict(input: IrisInput):
    model = getattr(app.state, "model", None)

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if input is None:
        raise HTTPException(status_code=400, detail="Input is empty")
    
    if not all([input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]):
        raise HTTPException(status_code=400, detail="All input features must be provided and non-zero")
    
    if any([input.sepal_length < 0, input.sepal_width < 0, input.petal_length < 0, input.petal_width < 0]):
        raise HTTPException(status_code=400, detail="Input features must be non-negative")

    df = pd.DataFrame([
        {
            "sepal_length": input.sepal_length,
            "sepal_width": input.sepal_width,
            "petal_length": input.petal_length,
            "petal_width": input.petal_width,
        }
    ])

    try:
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Ensure result is JSON serializable
    result = preds.tolist() if hasattr(preds, "tolist") else preds

    return {"prediction": result}