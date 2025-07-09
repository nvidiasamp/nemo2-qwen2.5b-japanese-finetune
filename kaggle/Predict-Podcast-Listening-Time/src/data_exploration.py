# /home/cho/workspace/kaggle/Predict-Podcast-Listening-Time/src/data_exploration.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Import project modules
import config
from utils import load_data, setup_logger

# Setup logger
logger = setup_logger(log_file_name='data_exploration.log')

def explore_data(df: pd.DataFrame, df_name: str = "DataFrame", save_plots: bool = False):
    """Performs basic data exploration and optionally saves plots."""
    logger.info(f"--- Exploring {df_name} ---")
    logger.info(f"Data Shape: {df.shape}")
    logger.info(f"\nData Types:\n{df.dtypes}")
    logger.info(f"\nMissing Values:\n{df.isnull().sum()}")
    logger.info(f"\nBasic Statistics:\n{df.describe(include='all')}")

    # Explore target variable distribution (if it exists)
    if config.TARGET_VARIABLE in df.columns:
        logger.info(f"\nTarget Variable Distribution ({config.TARGET_VARIABLE}):")
        logger.info(f"\n{df[config.TARGET_VARIABLE].describe()}")

        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[config.TARGET_VARIABLE], kde=True, bins=50)
            plt.title(f'Distribution of {config.TARGET_VARIABLE} in {df_name}')
            plt.xlabel(config.TARGET_VARIABLE)
            plt.ylabel('Frequency')

            if save_plots:
                plot_filename = f"{df_name.lower()}_{config.TARGET_VARIABLE}_distribution.png"
                plot_path = os.path.join(config.PLOT_DIR, plot_filename)
                plt.savefig(plot_path)
                logger.info(f"Target distribution plot saved to: {plot_path}")
                plt.close() # Close the plot figure after saving
            else:
                plt.show() # Show plot if not saving
        except Exception as e:
            logger.error(f"Error generating/saving target distribution plot: {e}")
            plt.close() # Ensure plot is closed in case of error

    elif df_name.lower().startswith('train'): # Only warn if target is missing in train data
        logger.warning(f"Target variable '{config.TARGET_VARIABLE}' not found in {df_name}.")

    logger.info(f"--- Finished Exploring {df_name} ---")
    # TODO: Add more exploration steps (e.g., correlation analysis, categorical feature analysis)

if __name__ == "__main__":

    logger.info("Starting data exploration script...")

    # Load Data using config paths
    logger.info(f"Loading training data from: {config.TRAIN_FILE}")
    df_train = load_data(config.TRAIN_FILE)

    logger.info(f"Loading test data from: {config.TEST_FILE}")
    df_test = load_data(config.TEST_FILE)

    # Explore Data (and save plots)
    explore_data(df_train, df_name="Training Data", save_plots=True)
    explore_data(df_test, df_name="Test Data", save_plots=False) # Don't plot target for test

    logger.info("Data exploration script finished.") 