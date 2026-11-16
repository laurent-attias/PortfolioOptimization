import json
from pathlib import Path

import pandas as pd
import yfinance as yf


def get_tickers():
    with open(Path('.')/"sp500_data.json", 'r') as file:
        dataset = json.load(file)
    tickers = [data["ticker"] for data in dataset]
    return tickers


def fetch_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    df_prices: pd.DataFrame = yf.download(tickers, start=start, end=end)
    df_prices = df_prices["Close"].dropna(
        axis=1, thresh=int(0.95 * len(df_prices))).dropna()
    df_returns = df_prices.pct_change().dropna()
    return df_returns


def compute_empirical_moments(df_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    df_expected_returns = df_returns.mean()
    df_covariance = df_returns.cov()
    df_covariance = 0.5 * (df_covariance + df_covariance.T)
    return df_expected_returns, df_covariance


def get_empirical_moments(start: pd.Timestamp, end: pd.Timestamp):
    return compute_empirical_moments(
        df_returns=fetch_returns(
            tickers=get_tickers(),
            start=start,
            end=end
        )
    )