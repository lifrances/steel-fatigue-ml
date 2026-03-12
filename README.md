## Project Overview
This project predicts steel fatigue using a combination of physics-informed feature engineering and machine learning. We compare multiple models, including linear regression, Ridge regression, and XGBoost, to evaluate prediction accuracy and feature importance.

## Dataset
Place the dataset files in the repository root directory:

- original_data.csv
- preprocessed_data_no_CE.csv
- preprocessed_data.csv
- preprocessed_data_no_HJP.csv
- preprocessed_data_no_logC.csv
- preprocessed_data_no_sixthC.csv

No preprocessing is required beyond what is already provided in these files.

## Requirements
Install the required Python packages using:

pip install -r requirements.txt

## Running the Analysis

Run the main analysis script:

[NAME OF THE SCRIPT]

This script performs:
- Data splitting (train/test)
- Model training (Linear, Ridge, XGBoost)
- Hyperparameter tuning with Optuna
- Evaluation metrics (R², RMSE, MAE)
- Visualization of predicted vs. actual values, residuals, and feature importance
- Ablation study results
