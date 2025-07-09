# /home/cho/workspace/kaggle/Predict-Podcast-Listening-Time/src/utils.py
import pandas as pd
import os
import logging
import sys
from pathlib import Path
import joblib # For saving/loading other objects if needed

# Import configuration
import config

def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a specified CSV file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def save_dataframe(df: pd.DataFrame, file_path: str):
    """Saves a DataFrame to a Parquet file."""
    try:
        # Ensure the directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path, index=False)
        logging.info(f"DataFrame saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving DataFrame to {file_path}: {e}")
        raise

def load_dataframe(file_path: str) -> pd.DataFrame:
    """Loads a DataFrame from a Parquet file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"DataFrame file not found at: {file_path}")
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"DataFrame loaded from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading DataFrame from {file_path}: {e}")
        raise

def save_object(obj: object, file_path: str):
    """Saves a Python object using joblib."""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise

def load_object(file_path: str) -> object:
    """Loads a Python object using joblib."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Object file not found at: {file_path}")
    try:
        obj = joblib.load(file_path)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise

def setup_logger(log_file_name: str = 'training.log') -> logging.Logger:
    """Sets up the logger to output to console and file."""
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(log_formatter)
    logger.addHandler(stdout_handler)

    # File Handler
    log_file_path = os.path.join(config.LOG_DIR, log_file_name)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    logging.info(f"Logger initialized. Log file: {log_file_path}")
    return logger

# Example usage of logger (can be called in other scripts):
# logger = setup_logger()
# logger.info("This is an info message.")
# logger.error("This is an error message.")

# Add other utility functions as needed, e.g., for logging, saving results, etc. 