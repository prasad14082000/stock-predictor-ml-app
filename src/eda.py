##src/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def eda_summary(df: pd.DataFrame, stock_name: str):
    print("\nData Summary:\n", df.describe())
    print("\nMissing Values:\n", df.isnull().sum())
    
    corr = df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Correlation Heatmap for {stock_name}")
    plt.tight_layout()
    plt.savefig(f"C://GITHUB CODES//stock-predictor-ml//reports/{stock_name}_correlation_heatmap.png")
    print(f"Correlation heatmap saved to reports/{stock_name}_correlation_heatmap.png")


def select_features_by_correlation(df: pd.DataFrame, threshold: float = 0.75, always_keep: list = None) -> list:
    if always_keep is None:
        always_keep = []

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = []
    for column in upper.columns:
        if any(upper[column] > threshold) and column not in always_keep:
            to_drop.append(column)

    selected = [col for col in df.columns if col not in to_drop]
    
    # Ensure essential features are preserved even if they were marked for removal
    for col in always_keep:
        if col not in selected and col in df.columns:
            selected.append(col)

    return selected
def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    X = X.select_dtypes(include=[np.number]).dropna()
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) if X.values[:, i].std() > 0 else np.inf for i in range(X.shape[1])]
    return vif_data[vif_data['VIF'] < 10]['Feature'].tolist()