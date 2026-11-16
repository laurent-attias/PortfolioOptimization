from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_optimization.data_loader import compute_empirical_moments
from portfolio_optimization.optimizer import solve_global_min_variance
from portfolio_optimization.utils import to_numpy_reg


@dataclass
class Evaluation:
    rebalancing_dates: pd.DatetimeIndex
    optimal_allocations: np.ndarray
    forecasted_volatility: np.ndarray
    turnover: np.ndarray
    returns_out_of_sample: np.ndarray
    realized_volatility: np.ndarray


def out_of_sample_evaluation(
    df_returns: pd.DataFrame,
    delta_days: int,
    sample_window_size: int,
):
    rebalancing_indices = range(
        sample_window_size, len(df_returns) - delta_days, delta_days
    )

    n_steps = len(rebalancing_indices)
    n_assets = df_returns.shape[1]

    optimal_allocations = np.empty((n_steps, n_assets))
    forecasted_volatility = np.empty(n_steps)
    realized_volatility = np.empty(n_steps)
    turnover = np.empty(n_steps)
    returns_out_of_sample = np.empty(n_steps * delta_days)

    weights = None
    turnover[0] = np.nan
    for k, rebalancing_day_idx in enumerate(rebalancing_indices):
        # estimation de la covariance in-sample
        df_returns_sample = df_returns.iloc[
            rebalancing_day_idx - sample_window_size : rebalancing_day_idx
        ]
        _, df_covariance_sample = compute_empirical_moments(
            df_returns=df_returns_sample
        )
        Sigma = to_numpy_reg(df_covariance=df_covariance_sample)

        # rebalancing du portefeuille
        optimal_allocation_min_variance = solve_global_min_variance(
            Sigma=Sigma,
        )

        optimal_allocations[k] = optimal_allocation_min_variance
        forecasted_volatility[k] = np.sqrt(
            np.dot(
                optimal_allocation_min_variance, Sigma @ optimal_allocation_min_variance
            )
        )

        # calcul du turnover
        if weights is not None:
            previous_weights = weights.copy()
            turnover[k] = 0.5 * np.sum(
                np.abs(optimal_allocation_min_variance - previous_weights)
            )

        # actualisation du portefeuille
        # rendements out-of-sample
        df_returns_out_of_sample: pd.DataFrame = df_returns.iloc[
            rebalancing_day_idx : rebalancing_day_idx + delta_days
        ]
        # poids effectifs
        weights = optimal_allocation_min_variance.copy()
        for day_idx, returns in enumerate(df_returns_out_of_sample.to_numpy()):
            daily_return = np.dot(weights, returns)
            returns_out_of_sample[k * delta_days + day_idx] = daily_return
            weights = weights * (1 + returns) / (1 + daily_return)
        # volatilité réalisée out-of-sample
        realized_volatility[k] = np.std(
            returns_out_of_sample[k * delta_days : (k + 1) * delta_days], ddof=1
        )

    return Evaluation(
        rebalancing_dates=df_returns.index[rebalancing_indices],
        optimal_allocations=optimal_allocations,
        forecasted_volatility=forecasted_volatility,
        turnover=turnover,
        returns_out_of_sample=returns_out_of_sample,
        realized_volatility=realized_volatility,
    )
