import numpy as np

class MixedPrecision:
    def __init__(self, loss_scale=1024.0):
        # Initialize loss scaling factor
        self.scale = loss_scale
    
    def forward(self, weights, inputs, targets):
        # Perform forward pass with float16, return scaled loss as a Python float
        weights, inputs, targets = map(lambda x: x.astype(np.float16), (weights,inputs,targets))
        preds = inputs @ weights
        loss = np.mean((targets - preds) ** 2)
        return loss.item() * self.scale
    
    def backward(self, gradients):
        # Unscale gradients and check for overflow, return as float32
        if np.any(np.isnan(gradients)) or np.any(np.isinf(gradients)):
            return np.zeros_like(gradients, dtype = np.float32)
        else:
            return gradients.astype(np.float32) / self.scale

        