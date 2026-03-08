# timeseries-ml

Financial time series machine learning analysis.

## Tech Stocks (`tech_timeseries.ipynb`)
- Daily OHLC (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA)
- Returns, cointegration, PCA, Lasso
- XGBoost & LSTM for next-day return prediction

## Crypto (`crypto_data.py`)
- Binance monthly trade tick data
- Lazy Polars DataFrames with disk-backed parquet
- Multi-threaded parallel downloads with caching
