# wizel-codetest (Regression Model Deployment)

Pipeline for training and deploying multivariate regression model into codespace.

## Prerequisites
- Docker installed
- Python 3.9-slim dependencies (managed via Docker)
- `data/`,  `results/` and `models/` directories in project root

## Prepare Data

1. Place your training_data.csv and blind_test_data.csv in the data/ directory
2. Ensure CSV files contain:
  - Training data: Features + target column (default name 'target')
  - Test data: Same features as training data

## Pipeline Run

3. Build Docker Image

    ```docker build -t regression-model .```

4. Train Model

    ```docker run -v $(pwd)/data:/data -v $(pwd)/models:/models regression-model python train_model.py``` 

5. Run Predictions

     ```docker run -v $(pwd)/data:/data -v $(pwd)/models:/models -v $(pwd)/results:/results regression-model python predict.py```

## Results

6. Results are saved into  `results/`  folder

## Run Me script

7. For ease, you can use the ` run_me.sh` script once the image is built
