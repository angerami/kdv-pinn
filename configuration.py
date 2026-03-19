from types import SimpleNamespace
import json

kdv_config = SimpleNamespace(
    num_epochs=150,
    num_samp_bulk=100,
    num_samp_eval=128,
    seed=123,
    lr=0.001,
    eta_min=1e-5,
    omega_0=30.0,
    MLP_dims=[2, 128, 128, 128, 1],  # KDV is scalar: (t,x) -> u
    # Domain parameters
    T=1.0,                # Time domain: [0, T]
    L=10.0,               # Spatial domain: [-L, L]
    Lmax=10.0,
    vmin=-3,              # Colorbar min for plotting
    vmax=0,               # Colorbar max for plotting
    # Loss and their weights
    loss_types=['kdv', 'BC']
    lambda_kdv = 1
    lambda_BC = 1
    # Soliton parameters
    soliton_type='sech',  # 'sech' or 'cn' for cnoidal wave
    kappa=1.0,            # Wave number parameter κ
    m=0.5,                # Elliptic modulus m ∈ [0,1] for cnoidal waves
    x0=0.0,               # Initial position offset
    plot_interval=50,
)

# Save
def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(vars(config), f, indent=2)

# Load
def load_config(path):
    with open(path) as f:
        return SimpleNamespace(**json.load(f))