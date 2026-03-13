# Predicting Steel Fatigue Strength Using Modern Data-Driven Models

## Overview
Fatigue failure accounts for a large proportion of structural failures in engineering systems. Accurately predicting fatigue strength is therefore critical for designing reliable steel components. However, experimental fatigue testing is costly and time-consuming, motivating the development of data-driven prediction models. 

This project investigates whether physics-informed feature engineering combined with machine learning models can improve fatigue strength prediction for steels. We compare three regression models: 

- Linear Regression (OLS)
- Ridge Regression
- XGBoost

In addition to the original dataset features, metallurgical-based engineered features were introduced, including: 

- Hollomon-Jaffe Parameter (HJP)
- Carbon Equivalent (CE)
- Murakami defect-based transformations

Models are trained and evaluated on both the original dataset and a preprocessed dataset to assess the impact of these engineered features on prediction accuracy.

## Repository Structure

```
steel-fatigue-ml/

main.py                                # Main script that runs the full modeling pipeline
requirements.txt                       # Python dependencies
README.md                              # Project documentation

data/                                  # Input datasets
│
├── original_data.csv                  
├── preprocessed_data.csv              
├── preprocessed_data_no_CE.csv
├── preprocessed_data_no_HJP.csv
├── preprocessed_data_no_logC.csv
└── preprocessed_data_no_sixthC.csv

src/                                  # Source code modules
│
├── __init__.py
├── data_loader.py                    # Loads and prepares datasets
├── model.py                          # Model training and evaluation
├── optimization.py                   # Hyperparameter optimization (Ridge + XGBoost)
└── plotting.py                       # Diagnostic plots and result visualization

results/                              # Generated outputs (plots and evaulation metrics)
```

> Note: The `results/` folder is automatically created when `main.py` is executed.

## Data Preparation

All required datasets are stored in the `data/` directory:

- `original_data.csv`
- `preprocessed_data.csv`
- `preprocessed_data_no_CE.csv`
- `preprocessed_data_no_HJP.csv`
- `preprocessed_data_no_logC.csv`
- `preprocessed_data_no_sixthC.csv`

The project automatically loads all datasets and performs an 80/20 train-tet split during execution. 

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lifrances/steel-fatigue-ml.git
cd steel-fatigue-ml
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

Required packages include:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `optuna`
- `matplotlob`

## Usage

Run the full modeling pipeline with:
```bash
python main.py
```

The script will: 
1. Load datasets from `data/`
2. Train and evaluate Linear Regression (OLS)
3. Tune Ridge Regression hyperparameters
4. Perform XGBoost hyperparameter optimization using Optuna
5. Evaluate the final XGBoost model
6. Generate diagnostic plots

All outputs, including evaluation metrics and plots, are saved in the `results/` directory.

## Notes

- Ensure all dataset files remain in the `data/` folder.
- The pipeline is fully automated; no additional preprocessing is required.
