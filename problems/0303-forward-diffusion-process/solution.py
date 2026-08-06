import numpy as np

def forward_diffusion(x_0: np.ndarray, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: np.ndarray) -> np.ndarray:
    """
    Apply forward diffusion process to add noise to input data.
    
    Args:
        x_0: Original input data (numpy array)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: Noise array (same shape as x_0)
    
    Returns:
        Noisy sample x_t as numpy array
    """
    T = num_timesteps
    beta = np.array([beta_start + (i - 1) / (T - 1) * (beta_end - beta_start) for i in range(1,T+1)])
    alphas =1 - beta
    alpha = np.prod(alphas[:t])
    return x_0 * alpha ** 0.5 + noise * (1-alpha) ** 0.5