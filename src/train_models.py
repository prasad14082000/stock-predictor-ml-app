### src/train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.visuals import plot_residuals, plot_feature_importance
import os

def adjusted_r2_score(r2: float, n: int, k: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def select_features_by_correlation(df: pd.DataFrame, threshold: float = 0.75) -> list:
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [col for col in df.columns if col not in to_drop]

def train_multiple_models(df: pd.DataFrame, stock_name: str):
    all_features = df.drop(columns=['Close', 'Date']).select_dtypes(include=[np.number])
    filtered_features = select_features_by_correlation(all_features, threshold=0.75)

    X = df[filtered_features]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.9),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='linear', C=10, epsilon=1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = []
    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        adj_r2 = adjusted_r2_score(r2, X_test.shape[0], X_test.shape[1])

        # Save visuals
        plot_residuals(y_test, y_pred, name.replace(" ", "_").lower(), stock_name)
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, X.columns, name.replace(" ", "_").lower(), stock_name)
            
        results.append((name, rmse, r2, adj_r2))
        model_path = f"C://GITHUB CODES//stock-predictor-ml//models/{stock_name}_{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(pipeline, model_path)

    print("\nModel Comparison Results:")
    for name, rmse, r2, adj_r2 in sorted(results, key=lambda x: x[1]):
        print(f"{name}: RMSE = {rmse:.2f}, R² = {r2:.4f}, Adjusted R² = {adj_r2:.4f}")

