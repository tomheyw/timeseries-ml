"""
Download Binance monthly trade tick zip files and convert to parquet on disk. 
Return dict of polars lazy frames for queried date range.
"""

import io
import zipfile
import polars as pl
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BINANCE_VISION_URL = "https://data.binance.vision/data/spot/monthly/trades"
DATA_DIR = Path(__file__).parent / "data" / "crypto"


def download_month_to_parquet(symbol: str, year: int, month: int) -> None:
    """
    Download monthly trade data from Binance Vision, convert to parquet on disk.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        year: Year (e.g., 2026)
        month: Month (1-12)
    
    Returns:
        None - writes directly to disk
    """
    month_str = f"{year}-{month:02d}"
    url = f"{BINANCE_VISION_URL}/{symbol}/{symbol}-trades-{month_str}.zip"
    
    logger.info(f"Downloading {symbol} {month_str}")
    
    # Download zip with progress bar
    response = requests.get(url, timeout=10_000, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    content = io.BytesIO()
    
    with tqdm(
        total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
        desc=f"{symbol} {month_str}", leave=False, dynamic_ncols=True, 
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content.write(chunk)
                pbar.update(len(chunk))
    
    # Extract csv from zip (reset buffer position to read from start)
    content.seek(0)
    with zipfile.ZipFile(content) as zf:
        csv_data = zf.read(zf.namelist()[0])
    
    # Parse csv to Polars DataFrame 
    schema = {
        'trade_id': pl.UInt64,
        'price': pl.Float64,
        'quantity': pl.Float64,
        'quote_quantity': pl.Float64,
        'time': pl.Int64,
        'is_buyer_maker': pl.Boolean,
        'is_best_match': pl.Boolean,
    }
    
    df = pl.read_csv(
        csv_data,
        has_header=False,
        new_columns=list(schema.keys()),  # no headers in Binance csvs
        schema=schema
    )
    
    logger.info(f"Parsed {len(df):,} trades")
    
    # Add symbol and datetime columns
    df = df.with_columns([
        pl.lit(symbol).alias('symbol'),
        pl.from_epoch(pl.col('time'), time_unit='ms').alias('datetime'),
    ])
    
    # Convert to parquet in memory and save to disk
    symbol_dir = DATA_DIR / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = symbol_dir / f"{month_str}.parquet"
    df.write_parquet(str(filepath), compression='zstd')
    
    logger.info(f"Saved parquet to {filepath}")


def load_month(symbol: str, year: int, month: int, download_missing: bool) -> pl.LazyFrame | None:
    """Load or download a single month of data as a lazy DataFrame."""
    month_str = f"{year}-{month:02d}"
    symbol_dir = DATA_DIR / symbol
    filepath = symbol_dir / f"{month_str}.parquet"
    
    if filepath.exists():
        logger.info(f"Loading {symbol} {month_str} from cache")
        return pl.scan_parquet(str(filepath))
    else:
        if download_missing:
            logger.info(f"{symbol} {month_str} not found, downloading...")
            try:
                # Download writes to disk, then return lazy frame from parquet
                download_month_to_parquet(symbol, year, month)
                return pl.scan_parquet(str(filepath))
            except Exception as e:
                logger.error(f"Failed to download {symbol} {month_str}: {e}")
                return None
        else:
            logger.warning(f"Skipping {symbol} {month_str}, file not found")
            return None


def query_data(
        symbols: list, 
        start_date: str,
        end_date: str,
        download_missing: bool = False,
        num_threads: int = 3
) -> dict[str, pl.LazyFrame]:
    """
    Query trade data from parquet files with multithreading, checking local cache first.
    
    Args:
        symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
        start_date: Start date as YYYY-MM-DD string
        end_date: End date as YYYY-MM-DD string
        download_missing: If True, download missing months; if False, skip missing
        num_threads: Number of threads to use for parallel loading/downloading
    
    Returns:
        Dictionary mapping symbol to LazyFrame with trade data
    """
    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Cap end_date at end of previous month (last available dataset)
    end = min(end, datetime.now().replace(day=1) - timedelta(days=1))
    
    start_year, start_month = start.year, start.month
    end_year, end_month = end.year, end.month
    
    months = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    # Submit all symbol/month combinations to executor for parallel processing
    results_by_symbol = {symbol: [] for symbol in symbols}
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for symbol in symbols:
            for year, month in months:
                future = executor.submit(load_month, symbol, year, month, download_missing)
                futures[future] = symbol
        
        # Collect results as they complete
        for future in as_completed(futures):
            symbol = futures[future]
            df = future.result()
            if df is not None:
                results_by_symbol[symbol].append(df)
    
    # Concatenate results for each symbol after all threads complete (lazy concat)
    results = {}
    for symbol in symbols:
        if results_by_symbol[symbol]:
            results[symbol] = pl.concat(results_by_symbol[symbol])
            logger.info(f"Queued {len(results_by_symbol[symbol])} months for {symbol}")
        else:
            logger.warning(f"No data found for {symbol}")
            results[symbol] = pl.LazyFrame()
    
    return results


if __name__ == "__main__":
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    query_data(symbols, "2025-08-01", "2025-12-01", download_missing=True)
