import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# == CONFIG===
MODEL_PATH = '/models/model.pkl'    # Path to trained model
TEST_DATA_PATH = '/data/blind_test_data.csv' # test data
TARGET_COLUMN = 'target'
PREDICTIONS_SAVE_PATH = '/results/predictions.csv'
# =================

def main():
    # Load model and data witht a try/except just to make sure that the user trained the model
    try:    
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Model not found at {MODEL_PATH}!!!")
        print("Train the model first with train_model.py")
        sys.exit(1)
    
    d = pd.read_csv(TEST_DATA_PATH)
    
    # I put this clause in case that we could be changing the blind_test_data file to a subset from training_data file
    if TARGET_COLUMN in d.columns:
        X = d.drop(columns=[TARGET_COLUMN])
        y = d[TARGET_COLUMN]
    else:
        X = d
    
    # Predict
    predictions = model.predict(X)

    results = pd.DataFrame({
            'target_pred': predictions
        })
    results.to_csv(PREDICTIONS_SAVE_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_SAVE_PATH}")
    print(f"Total predictions length: {len(results)}")

if __name__ == '__main__':
    main()