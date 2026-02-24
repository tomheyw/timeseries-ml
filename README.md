# timeseries-ml

A complete, reproducible pipeline for financial time series analysis and machine learning prediction of US tech stock returns. Built for research, education, and rapid prototyping of quantitative trading ideas.

## Features
- **Data Acquisition & Caching:**
  - Downloads 2 years of daily close prices for AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA using yfinance
  - Caches data in SQLite for fast reloads
- **Data Quality Checks:**
  - Nulls, negatives, duplicates, index periodicity
- **Return Calculation:**
  - Computes log returns for all stocks
- **Visualization:**
  - Cumulative returns, box plots, correlation heatmaps, PCA scree plots
- **Univariate Analysis:**
  - Normality (Jarque-Bera), stationarity (ADF), autocorrelation (ACF, Ljung-Box)
- **Signal Quality:**
  - Information Coefficient (IC) of each stock's return vs next-day AAPL return
- **Multivariate Analysis:**
  - Correlation matrix, Engle-Granger cointegration, PCA
- **Mean Reversion:**
  - AR(1) model and half-life estimation for each ticker
- **Feature Selection:**
  - Lasso regression with cross-validation
- **Machine Learning Models:**
  - XGBoost and PyTorch LSTM for next-day AAPL return prediction
  - Proper time series train/test split (no shuffling)
  - StandardScaler for features/target
  - Evaluation: R², RMSE, MAE

## Quickstart
1. Clone the repo:
   ```sh
   git clone https://github.com/tomheyw/timeseries-ml.git
   cd timeseries-ml
   ```
2. Install dependencies (recommended: Python 3.12+):
   ```sh
   pip install -r requirements.txt
   # or
   pip install uv && uv pip install --system .
   ```
3. Run the notebook:
   - Open `timeseries.ipynb` in VS Code or Jupyter
   - Run all cells

## Project Structure
- `timeseries.ipynb` — Main analysis notebook
- `requirements.txt` — Full dependency list
- `pyproject.toml` — Modern Python project metadata
- `data/market_data.db` — SQLite cache (auto-created)

## Notes
- GPU acceleration for PyTorch is supported (CUDA 12.1 wheels in `pyproject.toml`)
- All code is designed for clarity and reproducibility
- For research/educational use only — not investment advice

## License
MIT License
