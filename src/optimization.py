import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score


# Ridge parameter tuning visualization

def find_best_alpha(X_train, y_train, name):
    """
    Performs alpha search for Ridge Regression and saves the coefficient impact plot.
    """
    alphas = np.logspace(-3, 3, 100)
    best_alpha = None
    best_cv_r2 = -np.inf
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Search for the best alpha
    for alpha in alphas:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha))
        ])
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2')
        if cv_scores.mean() > best_cv_r2:
            best_cv_r2 = cv_scores.mean()
            best_alpha = alpha

    # Train a final model with best_alpha to extract coefficients
    final_ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=best_alpha))
    ])
    final_ridge.fit(X_train, y_train)

    # Plot Ridge Coefficients
    coefs = final_ridge.named_steps['ridge'].coef_
    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': coefs
    }).sort_values(by='Coefficient', ascending=True)

    plt.figure(figsize=(10, 8))
    colors = ['#d73027' if x < 0 else '#4575b4' for x in coef_df['Coefficient']]
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.8, edgecolor='black')
    plt.axvline(0, color='black', linestyle='-', linewidth=1.5)
    plt.title(f"[{name}] Ridge Standardized Coefficients")
    plt.xlabel("Coefficient Value (Standardized Weight)")
    plt.tight_layout()
    plt.savefig(f"results/{name}_ridge_coefficients.png", dpi=300)
    plt.close()

    return best_alpha


# XGBoost optuna tuning & visualization

def run_xgboost_optimization(X_train, y_train, name, n_trials=100):
    """
    Performs Bayesian optimization for XGBoost and saves plots.
    """

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            "random_state": 42
        }
        model = XGBRegressor(**params)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Plot Parameter Importance
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title(f"Parameter Importance")
    plt.tight_layout()
    plt.savefig(f"results/{name}_xgb_param_importance.png", dpi=300)
    plt.close()

    return study.best_params, study
