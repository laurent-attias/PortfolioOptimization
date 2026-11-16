import numpy as np
import pandas as pd

from portfolio_optimization.data_loader import compute_empirical_moments
from portfolio_optimization.optimizer import solve_global_min_variance
from portfolio_optimization.utils import to_numpy_reg


def isoweighted_allocator(
    df_returns: pd.DataFrame,
) -> tuple[np.ndarray, float] :
    n_assets = df_returns.shape[1]
    forecasted_volatility = df_returns.mean(axis=1).std(ddof=1)
    allocation = np.full(
        shape=n_assets,
        fill_value=1.0 / n_assets
    )
    return allocation, forecasted_volatility 


def global_min_variance_allocator(
    df_returns: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    _, df_covariance_sample = compute_empirical_moments(
        df_returns=df_returns,
    )
    Sigma = to_numpy_reg(df_covariance=df_covariance_sample)
    allocation = solve_global_min_variance(
        Sigma=Sigma,
    )
    forecasted_volatility = np.sqrt(
        np.dot(
            allocation,
            Sigma @ allocation
        )
    )
    return allocation, forecasted_volatility
