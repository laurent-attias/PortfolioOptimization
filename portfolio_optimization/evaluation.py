from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from portfolio_optimization.utils import TRADING_DAYS_PER_YEAR


@dataclass
class Evaluation:
    rebalancing_dates: pd.DatetimeIndex
    allocations: np.ndarray
    forecasted_volatility: np.ndarray
    turnover: np.ndarray
    returns_out_of_sample: np.ndarray
    realized_volatility: np.ndarray

    def summarize(self, display=False) -> dict[str, float] | None:
        annualized_return = TRADING_DAYS_PER_YEAR * self.returns_out_of_sample.mean()
        annualized_volatility = np.sqrt(
            TRADING_DAYS_PER_YEAR) * self.returns_out_of_sample.std(ddof=1)
        annualized_sharpe = annualized_return/annualized_volatility
        risk_bias_ratio = self.realized_volatility.mean()/self.forecasted_volatility.mean()
        mean_turnover = np.nanmean(self.turnover)
        summary = {
            "arithmetic_annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "annualized_sharpe": annualized_sharpe,
            "risk_bias_ratio": risk_bias_ratio,
            "mean_turnover": mean_turnover,
        }
        if display:
            for metric_name, metric in summary.items():
                print(f"{metric_name} : {metric:.1%}")
        else:
            return summary


def out_of_sample_evaluation(
    df_returns: pd.DataFrame,
    allocator: Callable[[pd.DataFrame], tuple[np.ndarray, float]],
    delta_days: int,
    sample_window_size: int,
):
    rebalancing_indices = range(
        sample_window_size, len(df_returns) - delta_days, delta_days
    )

    n_steps = len(rebalancing_indices)
    n_assets = df_returns.shape[1]

    allocations = np.empty((n_steps, n_assets))
    forecasted_volatility = np.empty(n_steps)
    realized_volatility = np.empty(n_steps)
    turnover = np.empty(n_steps)
    returns_out_of_sample = np.empty(n_steps * delta_days)

    weights = None
    turnover[0] = np.nan
    for k, rebalancing_day_idx in enumerate(rebalancing_indices):
        # estimation de la covariance in-sample
        df_returns_sample = df_returns.iloc[
            rebalancing_day_idx - sample_window_size: rebalancing_day_idx
        ]

        # rebalancing du portefeuille
        allocations[k], forecasted_volatility[k] = allocator(df_returns_sample)

        # calcul du turnover
        if weights is not None:
            previous_weights = weights.copy()
            turnover[k] = 0.5 * np.sum(
                np.abs(allocations[k] - previous_weights)
            )

        # actualisation du portefeuille
        # rendements out-of-sample
        df_returns_out_of_sample: pd.DataFrame = df_returns.iloc[
            rebalancing_day_idx: rebalancing_day_idx + delta_days
        ]
        # poids effectifs
        weights = allocations[k].copy()
        for day_idx, returns in enumerate(df_returns_out_of_sample.to_numpy()):
            daily_return = np.dot(weights, returns)
            returns_out_of_sample[k * delta_days + day_idx] = daily_return
            weights = weights * (1 + returns) / (1 + daily_return)
        # volatilité réalisée out-of-sample
        realized_volatility[k] = np.std(
            returns_out_of_sample[k * delta_days: (k + 1) * delta_days], ddof=1
        )

    return Evaluation(
        rebalancing_dates=df_returns.index[rebalancing_indices],
        allocations=allocations,
        forecasted_volatility=forecasted_volatility,
        turnover=turnover,
        returns_out_of_sample=returns_out_of_sample,
        realized_volatility=realized_volatility,
    )
