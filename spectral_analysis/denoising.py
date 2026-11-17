import numpy as np


def estimate_noise_variance(
    eigenvalues: np.ndarray,
    concentration_ratio: float,
    max_iter: int = 20,
    tol: float = 1e-8,
) -> float:
    d = len(eigenvalues)
    noise_variance = 1.0

    for _ in range(max_iter):
        lambda_plus = noise_variance * (1 + np.sqrt(concentration_ratio)) ** 2
        new_noise_variance = 1.0 - eigenvalues[eigenvalues > lambda_plus].sum() / d

        if abs(new_noise_variance - noise_variance) < tol:
            noise_variance = new_noise_variance
            break

        noise_variance = new_noise_variance
    
    return noise_variance


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
    pdf[mask] = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) / (2 * np.pi * concentration_ratio * noise_variance * x[mask])
    return pdf