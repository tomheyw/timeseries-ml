# timeseries-ml

Financial time series analysis and ML pipeline for US tech stocks.

**Features:**
- Downloads & caches 2 years of daily prices (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA)
- Log returns, data checks, visualizations
- Univariate & multivariate stats, cointegration, PCA
- Lasso feature selection
- XGBoost & LSTM (PyTorch) for next-day AAPL return prediction

**Quickstart:**
1. Clone: `git clone https://github.com/tomheyw/timeseries-ml.git`
2. Install: `pip install -r requirements.txt`
3. Run: Open `timeseries.ipynb` and run all cells

MIT License
