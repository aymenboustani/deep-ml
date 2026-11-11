def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> list[float]:
    """
    Compute learning rate schedule with linear warmup and cosine decay.
    
    Args:
        T: Total number of training steps
        W: Number of warmup steps
        lr_max: Maximum learning rate (reached after warmup)
        lr_min: Minimum learning rate (reached at end of training)
    
    Returns:
        List of learning rates for each step
    """
    # Your code here
    import math
    lr = []
    for i in range(T):
        if i < W:
            lr.append(i * lr_max / W)
        else:
            lr.append(
                lr_min + 0.5 * (lr_max - lr_min) * (
                    1 + math.cos((i-W)/ (T-W) * math.pi)
                )
                )
    return lr