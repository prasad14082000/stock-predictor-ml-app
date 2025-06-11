import shap
import matplotlib.pyplot as plt
import pandas as pd
import os

def explain_model_shap(model, X_train, stock_name, model_name):
    explainer = shap.Explainer(model.predict, X_train)
    shap_values = explainer(X_train)

    # Summary plot (bar)
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    path_bar = f"C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_{model_name}_shap_bar.png"
    plt.savefig(path_bar)
    plt.close()
    print(f"✅ SHAP bar plot saved at: {path_bar}")

    # Summary plot (beeswarm)
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    path_swarm = f"C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_{model_name}_shap_beeswarm.png"
    plt.savefig(path_swarm)
    plt.close()
    print(f"✅ SHAP beeswarm plot saved at: {path_swarm}")
