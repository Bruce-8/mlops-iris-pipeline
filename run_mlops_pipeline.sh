#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLFLOW_DB="${PROJECT_ROOT}/mlflow.db"

# 0. Initialize MLflow database (will be created automatically on first use, but ensure directory exists)
echo "Initializing MLflow database at ${MLFLOW_DB}..."
mkdir -p "$(dirname "${MLFLOW_DB}")"
# SQLite will auto-create the database when MLflow first connects, so we just ensure the directory exists
echo "MLflow database will be created at: ${MLFLOW_DB}"

# 1. Run preprocessing notebook
echo "Running preprocessing (iris_preprocess.ipynb)..."
jupyter nbconvert --to notebook --execute --inplace src/preprocess/iris_preprocess.ipynb
if [ $? -ne 0 ]; then
    echo "Preprocessing notebook failed."
    exit 1
fi

# 2. Run training notebook
echo "Running training (train_logistic_regression.ipynb)..."
jupyter nbconvert --to notebook --execute --inplace src/training/train_logistic_regression.ipynb
if [ $? -ne 0 ]; then
    echo "Training notebook failed."
    exit 1
fi

# 3. Evaluate models and promote best to Production
echo "Evaluating models and promoting best to Production..."
python src/model_evaluation/evaluate_models.py
if [ $? -ne 0 ]; then
    echo "Model evaluation failed."
    exit 1
fi

echo "Pipeline finished successfully!"
echo ""
echo "To view MLflow UI, run:"
echo "  mlflow server --backend-store-uri sqlite:///mlflow.db"
echo "Then open http://127.0.0.1:5000 in your browser"