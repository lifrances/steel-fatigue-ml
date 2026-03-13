"""
Main execution script for Steel Fatigue Strength Prediction.
Outputs:
1. Traditional Linear Regression (OLS) evaluation and plot
2. Ridge Regression tuning and evaluation and plot
3. XGBoost Optuna tuning, evaluation, and plot
"""

import os
import pandas as pd
from src.data_loader import load_all_datasets
from src.optimization import find_best_alpha, run_xgboost_optimization
from src.model import train_linear_regression, train_final_xgboost, calculate_metrics
from src.plotting import plot_results

def main():
    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. Load datasets
    all_datasets = load_all_datasets('data/')
    target_datasets = ['Original', 'Preprocessed']

    for name in target_datasets:
        if name not in all_datasets:
            continue

        print(f"\n{'='*60}")
        print(f" Full Evaluation: {name}")
        print(f"{'='*60}")

        X_train, X_test, y_train, y_test = all_datasets[name]

        # Traditional Linear Regression (OLS)
        print(f"[{name}] Running Traditional Linear Regression...")
        ols_model = train_linear_regression(X_train, y_train)
        ols_pred = ols_model.predict(X_test)
        plot_results(y_test, ols_pred, f"Ridge_{name}", save_path=f"results/{name}_ols_eval.png")
        
        ols_metrics = calculate_metrics(y_test, ols_pred)
        print(f"OLS Results: R2={ols_metrics['R2']:.4f}, RMSE={ols_metrics['RMSE']:.2f}, MAE={ols_metrics['MAE']:.4f}")

        # Ridge Regression
        print(f"[{name}] Running Ridge Analysis...")
        best_alpha = find_best_alpha(X_train, y_train, name)
        print(f"Best Ridge Alpha: {best_alpha:.4f}")
        
        ridge_model = train_final_ridge(X_train, y_train, best_alpha)
        ridge_pred = ridge_model.predict(X_test)
        plot_results(y_test, ridge_pred, f"Ridge_{name}", save_path=f"results/{name}_ridge_eval.png")
        
        ridge_metrics = calculate_metrics(y_test, ridge_pred)
        print(f"Ridge Results: R2={ridge_metrics['R2']:.4f}, RMSE={ridge_metrics['RMSE']:.2f}, MAE={ridge_metrics['MAE']:.4f}")

        # XGBoost
        print(f"[{name}] Optimizing XGBoost...")
        best_params, study = run_xgboost_optimization(X_train, y_train, name, n_trials=100)
        print(f"Best XGBoost Params: {best_params}")
        
        final_xgb = train_final_xgboost(X_train, y_train, best_params)
        xgb_pred = final_xgb.predict(X_test)
        plot_results(y_test, xgb_pred, f"Ridge_{name}", save_path=f"results/{name}_xgb_eval.png")
        
        xgb_metrics = calculate_metrics(y_test, xgb_pred)
        print(f"XGBoost Results: R2={xgb_metrics['R2']:.4f}, RMSE={xgb_metrics['RMSE']:.2f}, MAE={xgb_metrics['MAE']:.4f}")

    print(" All evaluations complete. Results saved in 'results/' folder.")

if __name__ == "__main__":

    main()

