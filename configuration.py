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
    vmin=0,              # Colorbar min for plotting
    vmax=3,               # Colorbar max for plotting
    # Loss and their weights
    loss_types=['kdv', 'BC'],
    lambda_kdv=1,
    lambda_BC=1,
    # Soliton parameters
    soliton_type='sech',  # 'sech', 'cn' (cnoidal), 'gauss', or 'scattering'
    kappa=1.0,            # Wave number parameter κ
    m=0.5,                # Elliptic modulus m ∈ [0,1] for cnoidal waves
    x0=0.0,               # Initial position offset
    # Scattering data parameters (for multi-soliton solutions)
    kappas=[1.0],         # List of wave numbers [κ₁, κ₂, ...] for scattering
    x0s=[1.4142],  # Norming constants [c₁(0), c₂(0), ...], default ~ x_0 = 0, sqrt(2kappa) ~ sqrt(2)
    plot_interval=50,
    equation_file='notes.md'
)

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(vars(config), f, indent=2)

def load_config(path):
    with open(path) as f:
        return SimpleNamespace(**json.load(f))

def config_to_dict(config):
    return vars(config)

def dict_to_config(config_dict):
    return SimpleNamespace(**config_dict)