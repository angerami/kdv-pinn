"""Configuration management for KdV PINN training."""
from types import SimpleNamespace
import json

kdv_config = SimpleNamespace(
    # Training parameters
    num_epochs=150,
    num_pretrain_epochs=200,
    num_samp_bulk=100,
    num_samp_eval=128,
    seed=123,
    lr=0.001,
    eta_min=1e-5,
    plot_interval=50,
    # Model architecture
    MLP_dims=[2, 128, 128, 128, 1],  # (t,x) -> u
    omega_0=30.0,  # SIREN frequency parameter
    # Domain parameters
    T=1.0,   # Time domain: [0, T]
    L=10.0,  # Spatial domain: [-L, L]
    Lmax=10.0,
    # Plotting parameters
    vmin=0,
    vmax=3,
    equation_file='equations.md',
    # Loss weights
    loss_types=['KDV', 'IC', 'BC'],
    lambda_KDV=1,
    lambda_IC=1,
    lambda_BC=1,
    # Soliton parameters for initial/boundary conditions
    kappas=[1.0],   # Wave numbers [κ₁, κ₂, ...]
    x0s=[1.4142],   # Initial positions
    m=0.5,          # Elliptic modulus for cnoidal waves
    x0=0.0,         # Position offset
)


def save_config(config, path):
    """Save configuration to JSON file."""
    with open(path, 'w') as f:
        json.dump(vars(config), f, indent=2)

def load_config(path):
    """Load configuration from JSON file."""
    with open(path) as f:
        return SimpleNamespace(**json.load(f))

def config_to_dict(config):
    """Convert configuration to dictionary."""
    return vars(config)

def dict_to_config(config_dict):
    """Convert dictionary to configuration."""
    return SimpleNamespace(**config_dict)