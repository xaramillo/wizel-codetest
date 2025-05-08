# WIZELINE TEST CODE FOR MULTIVARIATE REGRESSION MODEL TRAINING
# BY XARAMILLO
# 05/08/25
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import numpy as np

# === GLOBAL VARIABLES ====
DATA_PATH = '/data/training_data.csv' # You can change the 
TARGET_COLUMN = 'target'         
TEST_SIZE = 0.2
MODEL_SAVE_PATH = '/models/model.pkl'
# ======================== #

def main():
    # Load data
    d = pd.read_csv(DATA_PATH)
    
    # Separate features and target
    X = d.drop(columns=[TARGET_COLUMN])
    y = d[TARGET_COLUMN]
    
    # data splitting into training and val set
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = TEST_SIZE, random_state = 23)
    
    # Train model algorithm, for this implementation we will provide a Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_val)
    
    # Calculate metrics for regression
    metrics = {
        'mae': mean_absolute_error(y_val, y_pred),
        'mse': mean_squared_error(y_val, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred))),
        'r2': r2_score(y_val, y_pred)
    }
    
    print(f"Validation Metrics: {metrics}") # we can save the output to a log file as well
    
    # Saving our model to the models path, at this point we could versionize the model name, to iterate or to make experimentation
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {os.path.abspath(MODEL_SAVE_PATH)}")

if __name__ == '__main__':
    main()