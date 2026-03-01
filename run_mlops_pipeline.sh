#!/bin/bash

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

echo "Pipeline finished successfully!"