# 📈 Stock Forecasting & Options Pricing App (NSE + Black-Scholes)

A unified, interactive platform for **Machine Learning-based stock price forecasting** (LSTM, ElasticNet, and more) and **Black-Scholes Option Pricing & Heatmaps** for Indian equities.  
Ideal for beginners and tinkerers in Quantitative Finance, Data Science, or anyone curious about ML, markets, or F&O!

---

## 🚀 Features

- **Stock Prediction (NSE/any Yahoo Finance symbol)**
  - Downloads stock price data and computes 20+ technical indicators (RSI, MACD, moving averages, volatility, etc.)
  - Trains **7+ models** (Linear, Ridge, Lasso, ElasticNet, SVR, Random Forest, XGBoost)
  - Deep Learning (LSTM) forecasting
  - Multi-step forward prediction
  - Visualizations: LSTM and multi-step plots, result tables, CSV export

- **Option Pricing**
  - Black-Scholes call/put option pricing calculator (with Greeks: Delta, Gamma)
  - Plug in ML-predicted price, actual closing price, or any manual spot price
  - Interactive **heatmaps** for call/put prices across spot price & volatility

- **Unified, Modular UI**
  - All-in-one Streamlit app: seamless tabbed navigation between forecasting & options
  - Clean, beginner-friendly, and extensible Python code

---

## 📚 Why This Project?

I’ve been investing in Indian equities for 7+ years and learning statistics for 5. After living through market crashes, booms, COVID, elections, and earning an 18%+ XIRR, I wanted to finally explore the world of **derivatives** and **quant finance** hands-on.

I started by mixing:
- **Statistics/Probability** (my comfort zone)
- **ML models & Python**
- **Options theory** (Black-Scholes)
- And lots of “learn as you go” with OpenAI’s ChatGPT!

This project is the practical outcome: a bridge between data, code, markets, and “unknown waters” of F&O, now open to collaboration and feedback.

---

## 🖥️ Live Demo (Run Locally)

```bash
git clone https://github.com/prasad14082000/stock-predictor-ml.git
cd stock-predictor-ml
pip install -r requirements.txt
streamlit run unified_app.py


stock-predictor-ml/
│
├── unified_app.py              # Streamlit unified dashboard
├── run_pipeline.py             # ML pipeline runner
├── src/
│   ├── options/
│   │   └── black_scholes.py    # Black-Scholes class
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── evaluate_models.py
│   ├── train_models.py
│   ├── lstm_model.py
│   ├── forecast.py
├── data/                       # Processed data (pickle/csv)
├── models/                     # Saved model files
├── reports/                    # Plots, forecast CSVs, etc.
├── requirements.txt
└── README.md

📝 How to Use
Stock Forecast Tab

Select a stock (e.g. TITAN)

Enter start/end dates, forecast days

Click Run Forecast

See LSTM/multi-step plots, preview table, and summary

Option Pricing Tab

Choose to use the ML-predicted spot price, the actual last price, or enter a custom spot price

Set strike, maturity, volatility, and risk-free rate

Click Calculate Options

See call/put prices, Greeks, and dynamic heatmaps


💡 What’s Next?
Pull in live option chains and implied volatility

Auto-calculate volatility (σ) from historical prices

Add more Greeks: Theta, Vega, Rho

Strategy builder for common F&O spreads

Batch backtesting and walk-forward evaluation

Invite more collaboration from the quant/statistics/finance/data science community!

🤝 Contribute or Connect
Suggestions, PRs, and critical reviews are welcome!

prasadsonsale10@gmail.com | www.linkedin.com/in/prasad-sonsale/

⭐ Star the repo if you find it useful!

📜 License
MIT

Made by Prasad Sonsale — statistics nerd, investor, and aspiring quant. Built with ❤️, curiosity, and lots of Python.

