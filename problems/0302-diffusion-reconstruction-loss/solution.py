import numpy as np

def diffusion_loss(x_0: np.ndarray, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: np.ndarray, predicted_noise: np.ndarray) -> float:
    """
    Compute the reconstruction loss for diffusion model training.
    
    Args:
        x_0: Original input data (numpy array)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: True noise array (same shape as x_0)
        predicted_noise: Model's predicted noise (same shape as x_0)
    
    Returns:
        Mean squared error loss (float)
    """
    T = num_timesteps
    beta = np.array([beta_start + (beta_end - beta_start) * (i - 1) / (T - 1) for i in range(1,T+1)])
    alpha = 1- beta
    alpha_t = np.prod(alpha[:t])
    xt = x_0 * alpha_t ** 0.5 + (1 - alpha_t) ** 0.5 * noise
    x0_hat = (xt - (1-alpha_t) ** 0.5 * predicted_noise) / alpha_t ** 0.5
    return np.mean((x0_hat-x_0)**2)