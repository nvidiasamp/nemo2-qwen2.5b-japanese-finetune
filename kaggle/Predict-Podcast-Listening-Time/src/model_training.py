# /home/cho/workspace/kaggle/Predict-Podcast-Listening-Time/src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor # Example model
from sklearn.metrics import mean_squared_error
import joblib # For saving the model
import os

from utils import load_data, get_data_path
from feature_engineering import apply_feature_engineering

TARGET_VARIABLE = 'Listening_Time_minutes'
MODEL_OUTPUT_DIR = '../models' # Relative to src directory
MODEL_FILE_NAME = 'podcast_listener_model.joblib'

def train_model(X: pd.DataFrame, y: pd.Series):
    """Trains a model and returns the trained model object."""
    # TODO: Implement model selection, hyperparameter tuning (e.g., GridSearchCV)
    print("Training model...")
    # Example: Basic RandomForest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=10)
    model.fit(X, y)
    print("Model training complete.")
    return model

def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Evaluates the model on validation data using RMSE."""
    print("Evaluating model...")
    predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, predictions))
    print(f"Validation RMSE: {rmse:.4f}")
    return rmse

def save_model(model, output_dir: str, file_name: str):
    """Saves the trained model to a file."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, file_name)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    DATA_DIR = get_data_path()
    TRAIN_FILE = 'train.csv'

    # 1. Load Data
    print(f"Loading training data from: {DATA_DIR}")
    df_train = load_data(DATA_DIR, TRAIN_FILE)

    # 2. Feature Engineering
    # TODO: Need to handle potential data leakage if fitting scalers/encoders here
    # It might be better to integrate feature engineering within cross-validation folds
    # or split data first, then apply FE fit on train, transform on train/val.
    print("Applying feature engineering...")
    df_train_processed = apply_feature_engineering(df_train.drop(columns=[TARGET_VARIABLE]), is_train_data=True)
    y = df_train[TARGET_VARIABLE]
    X = df_train_processed

    # Ensure correct feature alignment if not done in FE
    # X = X.fillna(X.median()) # Example simple imputation after FE

    # 3. Split Data (Simple Train-Validation Split for now)
    # TODO: Implement robust cross-validation (e.g., KFold)
    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Model
    model = train_model(X_train, y_train)

    # 5. Evaluate Model
    rmse = evaluate_model(model, X_val, y_val)

    # 6. Save Model (Optionally train on full data before saving)
    print("\nTraining final model on full training data...")
    final_model = train_model(X, y) # Retrain on all data
    save_model(final_model, MODEL_OUTPUT_DIR, MODEL_FILE_NAME)
    print("\nModel training script finished.")

    # TODO: Add cross-validation implementation
    # Example KFold structure:
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # fold_rmses = []
    # for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    #     print(f"--- Fold {fold+1} ---")
    #     X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    #     y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
    #     # Apply FE fitting *only* on X_train_fold, then transform both
    #     model_fold = train_model(X_train_fold, y_train_fold)
    #     rmse_fold = evaluate_model(model_fold, X_val_fold, y_val_fold)
    #     fold_rmses.append(rmse_fold)
    # print(f"\nAverage Cross-Validation RMSE: {np.mean(fold_rmses):.4f}") 