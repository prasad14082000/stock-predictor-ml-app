import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_residuals(y_true, y_pred, model_name, stock_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot - {model_name}")
    plt.tight_layout()
    path = f"C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_{model_name}_residuals.png"
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved residual plot: {path}")

def plot_feature_importance(model, feature_names, model_name, stock_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=df)
        plt.title(f"Feature Importance - {model_name}")
        plt.tight_layout()
        path = f"C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_{model_name}_feature_importance.png"
        plt.savefig(path)
        plt.close()
        print(f"✅ Saved feature importance chart: {path}")
