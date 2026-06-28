"""Neural network architectures for KdV PINN."""
import torch
import torch.nn as nn
import numpy as np


class Siren(nn.Module):
    """SIREN network with sinusoidal activations.

    Uses periodic activation functions which can better represent solutions
    to PDEs with oscillatory behavior.

    Reference: Sitzmann et al., "Implicit Neural Representations with
               Periodic Activation Functions" (NeurIPS 2020)
               https://arxiv.org/abs/2006.09661
    """
    def __init__(self, config):
        super().__init__()
        self.omega_0 = config.omega_0
        dims = config.MLP_dims

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])

            # SIREN-specific initialization
            with torch.no_grad():
                if i == 0:
                    layer.weight.uniform_(-1 / dims[i], 1 / dims[i])
                elif i < len(dims) - 2:
                    bound = np.sqrt(6 / dims[i]) / self.omega_0
                    layer.weight.uniform_(-bound, bound)
                # Final layer uses default initialization

            self.layers.append(layer)

    def forward(self, input):
        y = input
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i < len(self.layers) - 1:
                y = torch.sin(self.omega_0 * y)
        return y


class KdV_pinn(nn.Module):
    """MLP with tanh activation for KdV equation.

    The tanh activation is well-suited for soliton solutions since
    sech^2 solitons are effectively derivatives of tanh.
    """
    def __init__(self, config):
        super().__init__()
        dims = config.MLP_dims

        self.layers = nn.ModuleList()
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


class AnalyticField(nn.Module):
    """Wrapper for analytic solutions to enable gradient computation.

    Wraps an analytic function in a nn.Module to allow it to be used
    interchangeably with learned models in the training pipeline.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, input):
        return self.func(input)