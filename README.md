# Project Overview
This project applies machine learning methods to predict fatigue strength in steels using metallurgically informed features. We evaluate multiple regression models, including linear regression, ridge regression, and XGBoost, and investigate the impact of physics-informed feature engineering on predictive performance.

# Data
The dataset used in this project originates from the NIMS fatigue database.
Download the dataset from:

[https://www.kaggle.com/datasets/chaozhuang/steel-fatigue-strength-prediction/data]

Place the dataset in the following directory:

data/steel_fatigue_dataset.csv

# Requirements
Install dependencies with:

pip install -r requirements.txt

# Running the Analysis

To reproduce the results in the paper, run:
python main.py

This script performs:
- Data loading and preprocessing
- Feature engineering
- Model training (Linear, Ridge, XGBoost)
- Hyperparameter optimization using Optuna
- Model evaluation and comparison
