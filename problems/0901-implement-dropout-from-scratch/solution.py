import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    mask = (torch.rand(x.size()) >= p).float()
    return x * mask / (1 - p) if training else x