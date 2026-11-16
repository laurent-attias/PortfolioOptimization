import cvxpy as cp
import numpy as np
import pandas as pd


def to_numpy_reg(
    df_covariance: pd.DataFrame,
    eps: float
) -> np.ndarray:
    Sigma = df_covariance.to_numpy()
    d = Sigma.shape[0]
    Sigma = Sigma + eps * np.eye(d)
    return Sigma

def solver_wrapper(
    variable: cp.Variable,
    objective: cp.Minimize,
    constraints: list[cp.Constraint],
):
    prob = cp.Problem(
        objective=objective,
        constraints=constraints
    )
    prob.solve(solver=cp.OSQP)
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError
    return variable.value


def solve_global_min_variance(
    df_covariance: pd.DataFrame,
    eps: float = 1e-8,
) -> np.ndarray:

    Sigma = df_covariance.to_numpy_reg()
    d = Sigma.shape[0]

    pi = cp.Variable(d)
    value = solver_wrapper(
        variable=pi,
        objective=cp.Minimize(cp.quad_form(pi, Sigma)),
        constraints=[
            cp.sum(pi) == 1.0,
            pi >= 0.0
        ]
    )
    return value


def solve_efficient_border(
    df_expected_returns: pd.DataFrame,
    df_covariance : pd.DataFrame,
    target_return: float,
    eps: float = 1e-8
) -> np.ndarray:
    
    Sigma = df_covariance.to_numpy_reg()
    d = Sigma.shape[0]
    M = df_expected_returns.to_numpy()

    pi = cp.Variable(d)
    value = solver_wrapper(
        variable=pi,
        objective=cp.Minimize(cp.quad_form(pi, Sigma)),
        constraints=[cp.sum(pi) == 1.0, pi >= 0.0, pi.T @ M >= target_return]        
    )
    
    return value


    

