import json
from pathlib import Path

import numpy as np
import pandas as pd


def to_numpy_reg(df_covariance: pd.DataFrame, eps: float = 1e-8) -> np.ndarray:
    Sigma = df_covariance.to_numpy()
    d = Sigma.shape[0]
    Sigma = Sigma + eps * np.eye(d)
    return Sigma


def get_tickers():
    with open(Path(".") / "sp500_data.json", "r") as file:
        dataset = json.load(file)
    tickers = [data["ticker"] for data in dataset]
    return tickers
