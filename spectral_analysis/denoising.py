from statistics import correlation
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde


def estimate_noise_variance(
    eigenvalues: np.ndarray,
    concentration_ratio: float,
    bandwidth: float = 0.05,
    nb_grid=1000,
) -> float:
    kde = gaussian_kde(eigenvalues, bw_method=bandwidth)
    grid = np.linspace(eigenvalues.min(), eigenvalues.max(), nb_grid)
    empirical_pdf = kde(grid)

    def loss(noise_variance: float) -> float:
        theorical_pdf = marchenko_pastur_pdf(
            x=grid,
            concentration_ratio=concentration_ratio,
            noise_variance=noise_variance,
            bulk_bounds=compute_bulk_bounds(
                concentration_ratio, noise_variance)
        )
        return np.sum((empirical_pdf - theorical_pdf) ** 2)

    res = minimize_scalar(loss, bounds=(1e-3, 1.0), method="bounded")
    return float(res.x)


def compute_bulk_bounds(
    concentration_ratio: float,
    noise_variance: float
) -> tuple[float, float]:
    lambda_minus = noise_variance * (1 - np.sqrt(concentration_ratio)) ** 2
    lambda_plus = noise_variance * (1 + np.sqrt(concentration_ratio)) ** 2
    return lambda_minus, lambda_plus


def marchenko_pastur_pdf(
    x: np.ndarray,
    concentration_ratio: float,
    noise_variance: float,
    bulk_bounds: tuple[float, float]
) -> np.ndarray:
    lambda_minus, lambda_plus = bulk_bounds
    pdf = np.zeros_like(x, dtype=float)
    mask = (lambda_minus <= x) & (x <= lambda_plus)
    pdf[mask] = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / \
        (2 * np.pi * concentration_ratio * noise_variance * x[mask])
    return pdf


def denoise_covariance(
    covariance_matrix: np.ndarray,
    concentration_ratio: float,
) -> np.ndarray:
    # 1. passage en corrélation
    volatility = np.sqrt(covariance_matrix.diagonal())
    inv_volatility = 1.0 / volatility
    correlation_matrix = covariance_matrix * np.outer(inv_volatility, inv_volatility)

    # 2. décomposition spectrale
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)

    # 3. estimation de la variance du bruit
    noise_variance = estimate_noise_variance(
        eigenvalues=eigenvalues,
        concentration_ratio=concentration_ratio
    )

    # 4. calcul des bornes du bulk
    _, lambda_plus = compute_bulk_bounds(
        concentration_ratio=concentration_ratio,
        noise_variance=noise_variance,
    )

    # 5. eigenvalue clipping
    eigenvalues[eigenvalues <= lambda_plus] =  eigenvalues[eigenvalues <= lambda_plus].mean()

    # 6. calcul de la matrice de corrélation clippée
    ## reconstruction
    clipped_correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    ## renormalisation
    inv_diag_sqrt = 1.0 / np.sqrt(clipped_correlation_matrix.diagonal())
    clipped_correlation_matrix = clipped_correlation_matrix * np.outer(inv_diag_sqrt, inv_diag_sqrt)

    # 7. calcul de la matrice de covariance clippée
    clipped_covariance_matrix = clipped_correlation_matrix * np.outer(volatility, volatility)

    return clipped_covariance_matrix
