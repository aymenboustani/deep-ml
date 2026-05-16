import torch
import torch.nn as nn

def build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    # TODO: return a Sequential of Linear -> ReLU -> Linear
    lin1 = nn.Linear(in_dim, hidden_dim)
    relu = nn.ReLU()
    lin2 = nn.Linear(hidden_dim, out_dim)
    return nn.Sequential(
        lin1,
        relu,
        lin2
    )
