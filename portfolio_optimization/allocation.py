import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from portfolio_optimization.data_loader import compute_empirical_moments
from portfolio_optimization.optimizer import solve_global_min_variance
from portfolio_optimization.utils import to_numpy_reg
from spectral_analysis.denoising import denoise_covariance


def GMV_allocate(
    covariance_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    allocation = solve_global_min_variance(
        Sigma=covariance_matrix,
    )
    forecasted_volatility = np.sqrt(
        np.dot(
            allocation,
            covariance_matrix @ allocation
        )
    )
    return allocation, forecasted_volatility


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


def GMV_allocator(
    df_returns: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    _, df_covariance = compute_empirical_moments(
        df_returns=df_returns,
    )
    covariance_matrix = to_numpy_reg(df_covariance=df_covariance)

    return GMV_allocate(
        covariance_matrix=covariance_matrix,
    )


def denoised_GMV_allocator(
    df_returns: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    _, df_covariance = compute_empirical_moments(df_returns=df_returns)
    covariance_matrix = to_numpy_reg(
        df_covariance=df_covariance
    )
    concentration_ratio = df_returns.shape[1]/df_returns.shape[0]
    denoised_covariance_matrix = denoise_covariance(
        covariance_matrix=covariance_matrix,
        concentration_ratio=concentration_ratio
    )
    return GMV_allocate(
        covariance_matrix=denoised_covariance_matrix,
    )


def ledoit_wolf_GMV_allocator(
    df_returns: pd.DataFrame,
) -> tuple[float, float]:
    T = df_returns.shape[0]
    lw = LedoitWolf().fit(df_returns.to_numpy())
    shrunk_covariance_matrix= lw.covariance_ * T / (T-1)  # LedoitWolf uses biased covariance matrix

    return GMV_allocate(
        covariance_matrix=shrunk_covariance_matrix
    )
