import os

def create_project_structure():
    folders = ['src', 'data', 'notebooks', 'models', 'reports']
    files = ['README.md', '.gitignore', 'requirements.txt']

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    with open("README.md", "w") as f:
        f.write("# Stock Predictor ML App\n\nThis repository contains a stock price prediction app using ML models.\n")

    with open(".gitignore", "w") as f:
        f.write("venv/\n__pycache__/\n*.pyc\n.DS_Store\n.env\n")

    with open("requirements.txt", "w") as f:
        f.write("yfinance\nnumpy\npandas\nmatplotlib\nscikit-learn\n")

    print("✅ Project setup completed in current directory.")

if __name__ == "__main__":
    create_project_structure()
