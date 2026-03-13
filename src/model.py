import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

def calculate_metrics(y_true, y_pred):
    """Calculates R2, RMSE, and MAE for model evaluation."""
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

def train_linear_regression(X_train, y_train):
    """
    Trains a basic OLS Linear Regression model.
    Used as the baseline (Traditional Linear) for your report.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_final_ridge(X_train, y_train, alpha):
    """Trains the final Ridge model using the optimized alpha."""
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(X_train, y_train)
    return model

def train_final_xgboost(X_train, y_train, best_params):
    """Trains the final XGBoost model using the best parameters from Optuna."""
    final_params = {
        **best_params,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    model = XGBRegressor(**final_params)
    model.fit(X_train, y_train)
    return model
