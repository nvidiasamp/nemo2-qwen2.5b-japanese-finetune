# /home/cho/workspace/kaggle/Predict-Podcast-Listening-Time/src/feature_engineering.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer # Example imputer

# Import project modules
import config
from utils import save_object, load_object, setup_logger

# Setup logger
logger = setup_logger(log_file_name='feature_engineering.log')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies basic preprocessing steps like handling missing values and type conversion."""
    logger.info("Starting data preprocessing...")
    processed_df = df.copy()

    # TODO: Implement more sophisticated missing value handling based on EDA
    # Example: Impute numerical features with median
    num_imputer = SimpleImputer(strategy='median')
    if config.NUMERICAL_FEATURES: # Check if list is populated
        logger.info(f"Imputing numerical features: {config.NUMERICAL_FEATURES}")
        processed_df[config.NUMERICAL_FEATURES] = num_imputer.fit_transform(processed_df[config.NUMERICAL_FEATURES])
    else:
        logger.warning("NUMERICAL_FEATURES list in config is empty. Skipping numerical imputation.")

    # Example: Impute categorical features with most frequent
    cat_imputer = SimpleImputer(strategy='most_frequent')
    if config.CATEGORICAL_FEATURES: # Check if list is populated
        logger.info(f"Imputing categorical features: {config.CATEGORICAL_FEATURES}")
        processed_df[config.CATEGORICAL_FEATURES] = cat_imputer.fit_transform(processed_df[config.CATEGORICAL_FEATURES])
    else:
        logger.warning("CATEGORICAL_FEATURES list in config is empty. Skipping categorical imputation.")

    # TODO: Add data type conversions if necessary (e.g., datetime)

    logger.info("Data preprocessing finished.")
    return processed_df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates new features from existing ones."""
    logger.info("Starting feature generation...")
    featured_df = df.copy()

    # TODO: Implement actual feature creation logic based on EDA
    # Examples:
    # - Extract time-based features (hour of day, day of week) from datetime columns
    # - Create interaction features between numerical/categorical variables
    # - Apply text feature extraction if applicable (e.g., TF-IDF from podcast descriptions)
    # Example: featured_df['new_feature'] = featured_df['col_a'] * featured_df['col_b']

    logger.info("Feature generation finished.")
    return featured_df

def fit_transformers(df_train: pd.DataFrame) -> Tuple[StandardScaler, OneHotEncoder]:
    """Fits StandardScaler and OneHotEncoder on training data and saves them."""
    logger.info("Fitting transformers (Scaler and Encoder)...")

    # Fit Scaler
    scaler = StandardScaler()
    if config.NUMERICAL_FEATURES:
        logger.info(f"Fitting StandardScaler on: {config.NUMERICAL_FEATURES}")
        scaler.fit(df_train[config.NUMERICAL_FEATURES])
        save_object(scaler, config.SCALER_FILE)
        logger.info(f"Scaler saved to {config.SCALER_FILE}")
    else:
        logger.warning("NUMERICAL_FEATURES list in config is empty. Skipping Scaler fitting.")
        scaler = None # Indicate scaler wasn't fitted/saved

    # Fit Encoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    if config.CATEGORICAL_FEATURES:
        logger.info(f"Fitting OneHotEncoder on: {config.CATEGORICAL_FEATURES}")
        encoder.fit(df_train[config.CATEGORICAL_FEATURES])
        save_object(encoder, config.ENCODER_FILE)
        logger.info(f"Encoder saved to {config.ENCODER_FILE}")
    else:
        logger.warning("CATEGORICAL_FEATURES list in config is empty. Skipping Encoder fitting.")
        encoder = None # Indicate encoder wasn't fitted/saved

    logger.info("Transformer fitting complete.")
    return scaler, encoder

def apply_transformations(df: pd.DataFrame, scaler: StandardScaler, encoder: OneHotEncoder) -> pd.DataFrame:
    """Applies fitted StandardScaler and OneHotEncoder to the data."""
    logger.info("Applying fitted transformations...")
    transformed_df = df.copy()

    # Apply Scaler
    if scaler and config.NUMERICAL_FEATURES:
        logger.info(f"Applying StandardScaler to: {config.NUMERICAL_FEATURES}")
        transformed_df[config.NUMERICAL_FEATURES] = scaler.transform(transformed_df[config.NUMERICAL_FEATURES])
    elif not scaler and config.NUMERICAL_FEATURES:
        logger.warning("Scaler was not provided or not fitted. Skipping numerical scaling.")
    elif not config.NUMERICAL_FEATURES:
        logger.info("No numerical features defined in config. Skipping scaling.")

    # Apply Encoder
    if encoder and config.CATEGORICAL_FEATURES:
        logger.info(f"Applying OneHotEncoder to: {config.CATEGORICAL_FEATURES}")
        encoded_data = encoder.transform(transformed_df[config.CATEGORICAL_FEATURES])
        # Get feature names after one-hot encoding
        encoded_feature_names = encoder.get_feature_names_out(config.CATEGORICAL_FEATURES)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=transformed_df.index)

        # Drop original categorical columns and concatenate encoded ones
        transformed_df = transformed_df.drop(columns=config.CATEGORICAL_FEATURES)
        transformed_df = pd.concat([transformed_df, encoded_df], axis=1)
        logger.info(f"Added {len(encoded_feature_names)} encoded features.")
    elif not encoder and config.CATEGORICAL_FEATURES:
        logger.warning("Encoder was not provided or not fitted. Skipping categorical encoding.")
    elif not config.CATEGORICAL_FEATURES:
        logger.info("No categorical features defined in config. Skipping encoding.")

    logger.info("Transformations applied successfully.")
    return transformed_df

# --- Removed Old Functions ---
# encode_categorical_features, scale_numerical_features, apply_feature_engineering
# The logic is now split into fit_transformers and apply_transformations

# --- Removed Example Usage ---
# The feature engineering steps will be called from model_training.py and predict.py 