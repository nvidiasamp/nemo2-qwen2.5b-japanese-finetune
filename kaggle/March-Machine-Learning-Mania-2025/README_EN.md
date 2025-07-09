# NCAA Basketball Tournament Prediction System

[English](README_EN.md) | [中文](README_CN.md) | [日本語](README_JP.md)

## Introduction

The NCAA Basketball Tournament Prediction System is a state-of-the-art machine learning solution designed to predict the outcomes of NCAA basketball tournament games with high accuracy. This system implements a sophisticated prediction pipeline that processes historical basketball data, engineers relevant features, trains optimized XGBoost models, and generates calibrated win probability predictions for tournament matchups.

This system is specifically designed for the [March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) Kaggle competition, which challenges participants to predict the outcomes of the NCAA basketball tournaments.

### Key Improvements in Version 5.0

- **GPU Acceleration**: Added CUDA support via cudf and cupy for dramatically improved performance on compatible hardware
- **Memory Optimization**: Enhanced memory management with adaptive batch processing and precision reduction
- **Error Resilience**: Improved validation, graceful fallbacks, and error recovery throughout the pipeline
- **Expanded Visualization**: Comprehensive visual analytics including calibration curves and comparative gender analysis
- **Multi-language Documentation**: Full documentation in English, Chinese, and Japanese

### Previous Version Improvements

- **Dual-gender prediction support**: Support for both men's and women's NCAA basketball tournaments
- **Performance optimizations**: Improved parallel processing and vectorized operations for faster data processing
- **Memory efficiency**: Better memory usage and caching strategies for handling large datasets
- **Robust error handling**: Improved validation and error recovery throughout the pipeline

## System Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- XGBoost
- joblib
- tqdm
- psutil (for memory monitoring)
- concurrent.futures (for parallel processing)
- cupy and cudf (optional, for GPU acceleration)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ncaa-prediction-system.git
cd ncaa-prediction-system

# Create a virtual environment (optional but recommended)
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install GPU dependencies (optional)
pip install cupy-cuda11x cudf-cuda11x
```

## System Architecture

The system follows a modular architecture designed for flexibility, reproducibility, and performance:

```
NCAA Prediction System
├── Data Acquisition Layer
│   ├── Historical Game Data Loading
│   ├── Team Information Processing
│   └── Tournament Structure Analysis
├── Feature Engineering Layer
│   ├── Team Performance Statistics
│   ├── Tournament Progression Modeling
│   ├── Matchup History Analysis
│   └── Seed-based Feature Generation
├── Model Training Layer
│   ├── Gender-specific Model Training
│   ├── Hyperparameter Optimization
│   ├── Cross-validation Framework
│   └── GPU-accelerated Learning
├── Prediction & Evaluation Layer
│   ├── Calibration Curve Analysis
│   ├── Brier Score Optimization
│   ├── Prediction Distribution Analysis
│   └── Risk-adjusted Strategy
└── Visualization & Reporting Layer
    ├── Interactive Performance Charts
    ├── Gender Comparison Analytics
    ├── Feature Importance Visualization
    └── Prediction Confidence Analysis
```

## Code Structure

The project is organized into several modules, each handling a specific aspect of the prediction pipeline:

- **main.py**: Orchestrates the entire workflow and provides command-line interface
- **data_preprocessing.py**: Handles data loading, exploration, and train-validation splitting
- **feature_engineering.py**: Creates features from raw data (team stats, seeds, matchups)
- **train_model.py**: Implements XGBoost model training with gender-specific models
- **submission.py**: Generates tournament predictions for submission
- **evaluate.py**: Contains evaluation metrics and visualization tools
- **utils.py**: Provides utility functions including GPU acceleration support

## Usage

### Basic Usage

```bash
python main.py --data_path ./data --output_path ./output --target_year 2025
```

### Advanced Options

```bash
python main.py --data_path ./data \
               --output_path ./output \
               --train_start_year 2016 \
               --train_end_year 2024 \
               --target_year 2025 \
               --explore \
               --random_seed 42 \
               --n_cores 8 \
               --use_gpu \
               --generate_predictions
```

### Command-line Arguments

- `--data_path`: Path to the data directory (default: '../input')
- `--output_path`: Path for output files (default: '../output')
- `--explore`: Enable data exploration and visualization (default: False)
- `--train_start_year`: Start year for training data (default: 2016)
- `--train_end_year`: End year for training data (default: 2024)
- `--target_year`: Target year for predictions (default: 2025)
- `--random_seed`: Random seed for reproducibility (default: 42)
- `--n_cores`: Number of CPU cores for parallel processing (default: auto-detect)
- `--use_cache`: Use cached data to speed up processing (default: False)
- `--use_gpu`: Enable GPU acceleration for compatible operations (default: False)
- `--xgb_trees`: Number of trees for XGBoost model (default: 500)
- `--xgb_depth`: Maximum tree depth for XGBoost model (default: 6)
- `--xgb_lr`: Learning rate for XGBoost model (default: 0.05)
- `--generate_predictions`: Generate predictions for all possible matchups (default: False)
- `--output_file`: Output file name for predictions (default: timestamp-based)
- `--load_models`: Load pre-trained models instead of training new ones (default: False)
- `--men_model`: Path to men's model file (default: None)
- `--women_model`: Path to women's model file (default: None)
- `--men_features`: Path to men's features file (default: None)
- `--women_features`: Path to women's features file (default: None)

## Data Requirements

The system expects the following CSV files in the data directory:

- **MTeams.csv**: Men's teams information
- **WTeams.csv**: Women's teams information
- **MRegularSeasonCompactResults.csv**: Men's regular season results
- **WRegularSeasonCompactResults.csv**: Women's regular season results
- **MNCAATourneyCompactResults.csv**: Men's tournament results
- **WNCAATourneyCompactResults.csv**: Women's tournament results
- **MRegularSeasonDetailedResults.csv**: Men's regular season detailed stats
- **WRegularSeasonDetailedResults.csv**: Women's regular season detailed stats
- **MNCAATourneySeeds.csv**: Men's tournament seeds
- **WNCAATourneySeeds.csv**: Women's tournament seeds
- **SampleSubmissionStage1.csv**: Sample submission format

## Key Features

### GPU Acceleration

- CUDA-based acceleration via cupy and cudf libraries
- Adaptive GPU memory management with fallback mechanisms
- Optimized tensor operations for feature engineering and model training
- Automatic hardware detection with graceful degradation to CPU

### Dual-Gender Prediction

- Separate models trained for men's and women's tournaments
- Gender-specific feature engineering tailored to each tournament's characteristics
- Combined prediction outputs for comprehensive tournament coverage
- Comparative analysis of prediction patterns between genders

### Advanced Feature Engineering

- Team performance statistics calculation
- Seed information processing
- Historical matchup analysis
- Tournament progression probability estimation
- Favorite-longshot bias correction
- Gender-specific feature adjustments

### Performance Optimization

- Multi-core parallel processing for compute-intensive operations
- GPU acceleration for compatible operations
- Memory caching to avoid redundant calculations
- Vectorized operations for improved efficiency
- Memory usage monitoring and optimization
- Time-aware function decorators for performance tracking

### Robust Evaluation

- Multiple metrics (Brier score, log loss, accuracy, ROC AUC)
- Calibration curve analysis
- Visual prediction distributions by gender
- Risk-optimized submission strategy based on Brier score properties
- Comparison analytics between men's and women's prediction models

## Prediction Pipeline

1. **Data Loading**: Load and preprocess historical basketball data for both genders
2. **Feature Engineering**: Create predictive features from raw data with gender-specific considerations
3. **Model Training**: Train separate XGBoost models for men's and women's tournaments
4. **Evaluation**: Evaluate model performance using multiple metrics
5. **Prediction Generation**: Create predictions for all possible tournament matchups
6. **Risk Strategy Application**: Apply optimal risk strategy for Brier score
7. **Submission Creation**: Format predictions for competition submission

## Theoretical Insights

The system implements several theoretical insights to improve prediction accuracy:

- **Brier Score Optimization**: For predictions with approximately 33.3% win probability, a strategic risk adjustment is applied to optimize the expected Brier score.
- **Favorite-Longshot Bias Correction**: The system corrects for the systematic underestimation of strong teams (low seeds) and overestimation of weak teams (high seeds).
- **Time-Aware Validation**: Validation is performed using more recent seasons to better reflect the temporal nature of basketball predictions.
- **Gender-Specific Modeling**: Separate models capture the unique characteristics of men's and women's basketball tournaments.
- **Calibration Theory**: Implements probability calibration techniques to ensure predicted probabilities accurately reflect true win likelihoods.

## Example Results

The system generates several output files:

- Trained model files for both men's and women's tournaments (men_model.pkl, women_model.pkl)
- Feature cache files (men_features.pkl, women_features.pkl)
- Prediction submission file (submission_YYYYMMDD_HHMMSS.csv)
- Model evaluation metrics and visualizations
- Comparative analysis between men's and women's predictions

## Advanced Usage

### GPU Acceleration

```python
from utils import gpu_context, to_gpu, to_cpu

# Check if GPU is available
with gpu_context(use_gpu=True) as gpu_available:
    if gpu_available:
        print("GPU acceleration enabled")
        # Move data to GPU
        X_gpu = to_gpu(X_train)
        y_gpu = to_gpu(y_train)
        
        # Process on GPU
        # ... processing steps ...
        
        # Move results back to CPU
        X_processed = to_cpu(X_gpu)
        y_processed = to_cpu(y_gpu)
    else:
        print("GPU not available, using CPU")
        X_processed = X_train
        y_processed = y_train
```

### Training Gender-Specific Models

```python
from train_model import train_gender_specific_models
from utils import save_features

# Prepare features for both genders
m_features, m_targets = merge_features(m_train_data, m_team_stats, m_seed_features, m_matchup_history)
w_features, w_targets = merge_features(w_train_data, w_team_stats, w_seed_features, w_matchup_history)

# Train gender-specific models
models = train_gender_specific_models(
    m_features, m_targets, w_features, w_targets,
    m_tourney_train, w_tourney_train,
    random_seed=42, save_models_dir='./models'
)

# Access individual models
men_model = models['men']['model']
women_model = models['women']['model']
```

### Generating Combined Predictions

```python
from submission import prepare_all_predictions, create_submission

# Generate predictions for both genders
all_predictions = prepare_all_predictions(
    model, features_dict, data_dict, 
    model_columns=model_columns,
    year=2025, 
    gender='both'  # Process both men's and women's matchups
)

# Create submission file
submission = create_submission(all_predictions, sample_submission, 'submission_2025.csv')
```

## Performance Notes

- Feature engineering is the most time-consuming part of the pipeline; use the `--use_cache` flag to reuse previously calculated features.
- GPU acceleration significantly improves performance but requires compatible hardware and drivers.
- For extremely large datasets, adjust the `n_cores` parameter to balance speed and memory usage.
- The system includes automatic batch size optimization to manage memory usage effectively.

## Visualization

The system generates several visualizations to help understand model performance:

- Prediction distribution charts for both men's and women's tournaments
- Calibration curves showing predicted vs. actual win probabilities
- Feature importance plots highlighting the most predictive factors
- Comparison plots showing differences between men's and women's predictions
- Memory and performance profiling charts

## References

- March Machine Learning Mania 2025: [https://www.kaggle.com/competitions/march-machine-learning-mania-2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)
- XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- Brier Score: [https://en.wikipedia.org/wiki/Brier_score](https://en.wikipedia.org/wiki/Brier_score)
- NCAA Tournament: [https://www.ncaa.com/march-madness](https://www.ncaa.com/march-madness)
- RAPIDS cuDF: [https://docs.rapids.ai/api/cudf/stable/](https://docs.rapids.ai/api/cudf/stable/)
- CuPy: [https://cupy.dev/](https://cupy.dev/)

## Author

Junming Zhao

## License

MIT License

---

This README provides a comprehensive overview of the NCAA Basketball Tournament Prediction System, including setup instructions, usage examples, and key technical details. For questions or contributions, please open an issue on the repository.