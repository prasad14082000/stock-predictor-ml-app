📈 Stock Price Predictor + Black-Scholes Option Pricing (ML + Quant Finance)
Welcome!
This repo combines two powerful financial tools:

A Machine Learning-powered Stock Price Predictor (LSTM, ElasticNet, and more)

A Black-Scholes Option Pricing & Heatmap Dashboard
Both built as a practical deep dive into Quantitative Finance and AI, aimed at Indian (NSE) stocks but easily extended.
Beginner friendly, modular, and ready to play with!

🚀 What This Project Does
Stock Forecasting

Download Indian NSE (or any Yahoo Finance) stock data

Engineer technical indicators: moving averages, RSI, MACD, volatility, etc.

Train 7 ML models: Linear, Ridge, Lasso, ElasticNet, SVR, Random Forest, XGBoost

Run LSTM deep learning forecasting

Multi-step forecasts: predict several future days, not just one

Easy-to-use Streamlit UI for input and outputs

Options Pricing

Interactive Black-Scholes calculator: call & put prices, plus Delta/Gamma Greeks

Dynamic heatmaps: see how prices change with spot price & volatility

Plug in ML-predicted or actual spot prices (or enter manually)

Clear, responsive Streamlit UI for quick experimentation

🧠 Why I Built This
I’ve invested in Indian stocks and mutual funds for 7+ years, weathered COVID, wars, bear/bull markets, FII cycles, and built an 18%+ XIRR portfolio.
But Futures & Options (F&O) were always the “next frontier”.
So I took my background in statistics & probabilities, and built this project — to learn how ML and classic quant models (like Black-Scholes) actually work together, and what it would take to approach finance like a quant.
This project is my “learn in public” notebook — and maybe yours too!

✨ Features
Module	Description
📉 ML Forecasting	Linear, Ridge, Lasso, ElasticNet, SVR, RF, XGB
🔮 LSTM Forecast	Deep learning, sequence-aware stock forecasting
📊 Model Evaluation	RMSE, R², residual plots
🔁 Multi-step	Predict multiple future days, not just one
📋 CSV/Plot Output	Forecasts saved as CSV and plots for later use
🧮 Option Pricing	Black-Scholes with call/put/Greeks
🌈 Option Heatmaps	Visualize option price sensitivity (spot, vol)
🔗 Unified Interface	One Streamlit app for both stocks & options
🧩 Modular	Codebase is clean, commented, easy to extend

📂 Project Structure
bash
Copy
Edit
stock-predictor-ml/
│
├── unified_app.py             # Streamlit unified dashboard
├── run_pipeline.py            # ML workflow runner
├── src/
│   ├── options/
│   │   └── black_scholes.py   # Black-Scholes pricing class
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── evaluate_models.py
│   ├── train_models.py
│   ├── lstm_model.py
│   ├── forecast.py
├── data/                      # Processed data (pickle/csv)
├── models/                    # Saved model files
├── reports/                   # Plots, CSVs
├── requirements.txt
└── README.md
🛠️ How to Run (Step-by-Step)
Clone this repo

bash
Copy
Edit
git clone https://github.com/prasad14082000/stock-predictor-ml.git
cd stock-predictor-ml
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Launch the unified app

bash
Copy
Edit
streamlit run unified_app.py
In the browser:

Use the Stock Forecast tab for ML-powered forecasts (select stock, set dates, etc.)

Use the Option Pricing tab for Black-Scholes pricing (choose spot price from ML, actual, or manual input)

🖥️ Demo
Forecast Tab Example	Options Tab Example
	

Replace image links with your own generated images if needed.

💡 What’s Inside / What I Learned
Feature engineering matters as much as models.

Classical models can still outperform DL in some regimes.

Deploying with Streamlit (or Gradio) makes ML results “real”.

Options pricing intuition comes alive when you can tweak and see — not just read theory!

“Learn by building” works — even with complex topics.

🚀 Future Scope / Next Steps
Integrate live data feeds (NSE/yfinance auto-refresh)

Auto-calculate volatility (σ) using historical data or implied vol

Add more Greeks (Theta, Vega, Rho) for richer risk analysis

Plug forecasted prices directly into real option chains from NSE

Add popular option strategies: spreads, straddles, etc

Batch backtesting: see how the models would have predicted in the past

Invite feedback and collaboration from fellow quants, statisticians, and finance nerds!

🤝 Contributing
Open to PRs, feedback, and new feature ideas!
Feel free to fork, raise an issue, or submit your own experiments.

📬 Contact / Social
📧 Email: prasadsonsale10@gmail.com

💼 LinkedIn: Prasad Sonsale

⭐ Star on GitHub if you find it useful!

📜 License
MIT

Built by Prasad Sonsale — a stats nerd learning in public!

Last updated: 2025-06-25