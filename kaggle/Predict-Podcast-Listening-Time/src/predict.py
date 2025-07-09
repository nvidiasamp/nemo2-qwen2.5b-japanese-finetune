# /home/cho/workspace/kaggle/Predict-Podcast-Listening-Time/src/predict.py
import pandas as pd
import joblib
import os

from utils import load_data, get_data_path
from feature_engineering import apply_feature_engineering
from model_training import MODEL_OUTPUT_DIR, MODEL_FILE_NAME, TARGET_VARIABLE

SUBMISSION_OUTPUT_DIR = '../submissions' # Relative to src directory
SUBMISSION_FILE_NAME = 'submission.csv'

def load_model(model_dir: str, model_file: str):
    """Loads the trained model from a file."""
    model_path = os.path.join(model_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    return model

def generate_predictions(model, X_test: pd.DataFrame) -> pd.Series:
    """Generates predictions on the test data."""
    print("Generating predictions...")
    predictions = model.predict(X_test)
    # Ensure predictions are non-negative if required by the problem
    predictions[predictions < 0] = 0
    return predictions

def create_submission_file(test_ids: pd.Series, predictions: pd.Series, output_dir: str, file_name: str):
    """Creates the submission file in the specified format."""
    submission_df = pd.DataFrame({
        'ID': test_ids, # Assuming the test set has an 'ID' column
        TARGET_VARIABLE: predictions
    })
    os.makedirs(output_dir, exist_ok=True)
    submission_path = os.path.join(output_dir, file_name)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file created at {submission_path}")

if __name__ == '__main__':
    DATA_DIR = get_data_path()
    TEST_FILE = 'test.csv'

    # 1. Load Test Data
    print(f"Loading test data from: {DATA_DIR}")
    df_test = load_data(DATA_DIR, TEST_FILE)
    test_ids = df_test['ID'] # Store IDs for submission file

    # 2. Apply Feature Engineering
    # Use is_train_data=False to apply transformations fitted on training data
    # TODO: Ensure scalers/encoders are loaded/used correctly
    print("Applying feature engineering to test data...")
    X_test = apply_feature_engineering(df_test, is_train_data=False)

    # Ensure correct feature alignment
    # X_test = X_test.fillna(X_test.median()) # Example imputation
    # TODO: Make sure columns match the training data features used by the model

    # 3. Load Model
    model = load_model(MODEL_OUTPUT_DIR, MODEL_FILE_NAME)

    # 4. Generate Predictions
    predictions = generate_predictions(model, X_test)

    # 5. Create Submission File
    create_submission_file(test_ids, predictions, SUBMISSION_OUTPUT_DIR, SUBMISSION_FILE_NAME)

    print("\nPrediction script finished.") 