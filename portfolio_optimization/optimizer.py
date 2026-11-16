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
        raise ValueError(f"Erreur lors de la résolution du problème d'optimisation : {prob.status}")
    return variable.value


def solve_global_min_variance(
    df_covariance: pd.DataFrame,
    eps: float,
) -> np.ndarray:

    Sigma = to_numpy_reg(df_covariance=df_covariance, eps=eps)
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


def solve_efficient_border_with_minimal_expected_return(
    df_expected_returns: pd.Series,
    df_covariance: pd.DataFrame,
    minimal_expected_return: float,
    eps: float,
) -> np.ndarray:
    
    Sigma = to_numpy_reg(df_covariance=df_covariance, eps=eps)
    d = Sigma.shape[0]
    M = df_expected_returns.to_numpy()

    pi = cp.Variable(d)
    value = solver_wrapper(
        variable=pi,
        objective=cp.Minimize(cp.quad_form(pi, Sigma)),
        constraints=[cp.sum(pi) == 1.0, pi >= 0.0, pi @ M >= minimal_expected_return]        
    )

    return value

def solve_efficient_border(
    df_expected_returns: pd.Series,
    df_covariance: pd.DataFrame,
    eps: float = 1.e-8,
    N: int = 10**3 + 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    Sigma = to_numpy_reg(df_covariance=df_covariance, eps=eps)    
    M = df_expected_returns.to_numpy()

    pi_min = solve_global_min_variance(
        df_covariance=df_covariance,
        eps=eps
    )
    min_expected_return = np.dot(pi_min, M)
    max_expected_return = df_expected_returns.max()
    minimal_expected_returns = np.linspace(min_expected_return, max_expected_return, N)[:-1]
    optimal_affectations = np.array(
        [
            solve_efficient_border_with_minimal_expected_return(
                df_expected_returns=df_expected_returns,
                df_covariance=df_covariance,
                minimal_expected_return=minimal_expected_return,
                eps=eps,
            ) 
            for minimal_expected_return in minimal_expected_returns
        ]
    )
    efficient_border_expected_returns = np.array(
        [
            np.dot(optimal_affectation, M)
            for optimal_affectation in optimal_affectations
        ]
    )
    efficient_border_sigma = np.array(
        [
            np.sqrt(np.dot(optimal_affectation, Sigma @ optimal_affectation))
            for optimal_affectation in optimal_affectations
        ]
    )
    return optimal_affectations, efficient_border_expected_returns, efficient_border_sigma
    



