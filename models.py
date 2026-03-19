import torch
import torch.nn as nn
import numpy as np

class Siren(nn.Module):
    """SIREN network with sinusoidal activations.

    Reference: Implicit Neural Representations with Periodic Activation Functions
    https://arxiv.org/abs/2006.09661
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self.omega_0 = config.omega_0
        dims = config.MLP_dims
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            # SIREN initialization
            with torch.no_grad():
                if i == 0:
                    # First layer
                    layer.weight.uniform_(-1 / dims[i], 1 / dims[i])
                elif i < len(dims) - 2:
                    # Hidden layers
                    layer.weight.uniform_(-np.sqrt(6 / dims[i]) / self.omega_0, np.sqrt(6 / dims[i]) / self.omega_0)
                else:
                    # Final layer: use default initialization (outermost_linear=True in SIREN paper)
                    pass
            self.layers.append(layer)

    def forward(self, input):
        y = input
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i < len(self.layers) - 1:
                y = torch.sin(self.omega_0 * y)
        return y

class KdV_pinn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        dims = config.MLP_dims
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)

    def forward(self, input):
        y = input
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i < len(self.layers) - 1:
                y = torch.tanh(y)
        return y

#model wrapper for analytic results that can be differentiated
class AnalyticField(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func  # func(input) -> (N, 2) or (N, 1)

    def forward(self, input):
        return self.func(input)