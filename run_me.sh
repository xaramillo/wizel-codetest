#!/usr/bin/env bash
# github.com/xaramillo
#
echo "# ============================================== #"
echo "MULTIVARIATE REGRESSION BASH PIPELINE FOR WIZELINE TEST CODING"
echo "BY XARAMILLO"
# Train the model, ensure that you have loaded the files for training_data and blind_test_data
docker run  -v $(pwd)/data:/data \
            -v $(pwd)/models:/models \
            regression-model \
            python train_model.py
# Predict and store the results in results folder
docker run  -v $(pwd)/data:/data \
            -v $(pwd)/models:/models \
            -v $(pwd)/results:/results \
            regression-model \
            python predict.py

echo "# ============================================== #"