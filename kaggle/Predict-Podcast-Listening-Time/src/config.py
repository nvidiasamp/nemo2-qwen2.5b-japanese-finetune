# /home/cho/workspace/kaggle/Predict-Podcast-Listening-Time/src/config.py
import os
import numpy as np

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project Root: Predict-Podcast-Listening-Time
# BASE_DIR = "/home/cho/workspace/kaggle/Predict-Podcast-Listening-Time" # Alternative if above fails

INPUT_DIR = os.path.join(BASE_DIR, 'input', 'playground-series-s5e4') # Adjusted to standard Kaggle structure if applicable
# INPUT_DIR = "/home/cho/workspace/kaggle/input/playground-series-s5e4" # Absolute path if needed

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'models')
SUBMISSION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'submissions')
OOF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'oof')
FEATURE_DIR = os.path.join(OUTPUT_DIR, 'features') # For saving processed features
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots') # For saving EDA plots
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs') # For saving logs

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_OUTPUT_DIR, exist_ok=True)
os.makedirs(OOF_OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# --- Data Files ---
TRAIN_FILE = os.path.join(INPUT_DIR, 'train.csv')
TEST_FILE = os.path.join(INPUT_DIR, 'test.csv')
SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, 'sample_submission.csv')

# --- Submission File ---
SUBMISSION_FILE = os.path.join(SUBMISSION_OUTPUT_DIR, 'submission.csv')

# --- Target Variable ---
TARGET_VARIABLE = 'Listening_Time_minutes'
ID_COLUMN = 'ID' # Assuming 'ID' is the identifier column

# --- Cross-Validation ---
N_SPLITS = 5
RANDOM_SEED = 42

# --- Features ---
# Define lists for feature types (populate after EDA)
CATEGORICAL_FEATURES = [
    # Example: 'Podcast_Category', 'Device_Type'
]
NUMERICAL_FEATURES = [
    # Example: 'Listener_Age', 'Previous_Listens'
]
# Features to drop (if any)
DROP_FEATURES = []


# --- Model ---
# Example model parameters (adjust for your chosen model)
MODEL_PARAMS = {
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    # Add other specific model parameters here
    # e.g., for RandomForest: 'n_estimators': 100, 'max_depth': 10
}

# --- Feature Engineering Objects ---
ENCODER_FILE = os.path.join(MODEL_OUTPUT_DIR, 'categorical_encoder.joblib')
SCALER_FILE = os.path.join(MODEL_OUTPUT_DIR, 'numerical_scaler.joblib')

# --- Final Model File ---
MODEL_FILE_NAME = 'podcast_listener_model.joblib'
FINAL_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILE_NAME) 