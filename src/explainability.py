import shap
import matplotlib.pyplot as plt
import pandas as pd
import os

def explain_model_shap(model, X_train, stock_name, model_name):
    # Pick explainer based on model type
    if "xgboost" in str(type(model)).lower():
        explainer = shap.Explainer(model)  # XGBoost 1.3+ supports this
    elif "randomforest" in str(type(model)).lower():
        explainer = shap.TreeExplainer(model)
    elif "elasticnet" in str(type(model)).lower() or "linearregression" in str(type(model)).lower():
        explainer = shap.Explainer(model, X_train)
    else:
        explainer = shap.Explainer(model, X_train)

    shap_values = explainer(X_train)

    try:
        # Summary plot 
        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False)
        path_bar = f"C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_{model_name}_shap_bar.png"
        plt.savefig(path_bar, bbox_inches='tight')
        plt.close()
        print(f"SHAP bar plot saved at: {path_bar}")

    except Exception as e:
        print(f"SHAP explanation failed for {model_name}: {e}")
