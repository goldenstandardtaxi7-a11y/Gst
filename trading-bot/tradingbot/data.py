"""Historical price loading for backtests."""
import os

import numpy as np
import pandas as pd


def load_csv_dir(path, symbols) -> pd.DataFrame:
    """Read <path>/<SYMBOL>.csv files with a Date column and an 'Adj Close' or 'Close' column."""
    series = {}
    for sym in symbols:
        file = os.path.join(path, f"{sym}.csv")
        if not os.path.exists(file):
            continue
        df = pd.read_csv(file)
        date_col = next(c for c in df.columns if c.lower() in ("date", "timestamp", "datetime"))
        price_col = next((c for c in df.columns if c.lower() in ("adj close", "adj_close")),
                         next(c for c in df.columns if c.lower() == "close"))
        series[sym] = pd.Series(df[price_col].values,
                                index=pd.to_datetime(df[date_col]).dt.tz_localize(None))
    if not series:
        raise FileNotFoundError(f"No CSV files for {symbols} in {path}")
    return pd.DataFrame(series).sort_index()


def synthetic_closes(symbols, days=2520, seed=7) -> pd.DataFrame:
    """Random-walk prices with regime shifts. Only for demos and tests, not for judging a strategy."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=days)
    out = {}
    for sym in symbols:
        drift = rng.choice([-0.0004, 0.0002, 0.0006], size=days // 250 + 1).repeat(250)[:days]
        vol = rng.uniform(0.006, 0.018)
        out[sym] = 100 * np.exp(np.cumsum(drift + rng.normal(0, vol, days)))
    return pd.DataFrame(out, index=index)
