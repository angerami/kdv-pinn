import torch
from functools import partial

def gradient(y, x):
    """Compute gradient of y with respect to x, handling edge cases."""
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    # Handle the case where y doesn't depend on x (returns None)
    if grad is None:
        grad = torch.zeros_like(x)
    #  Ensure gradient maintains requires_grad for further differentiation
    # This is needed when grad is a constant (e.g., all zeros)
    if not grad.requires_grad and x.requires_grad:
        # Create a new tensor with gradient tracking by adding a zero term that depends on x
        grad = grad + 0.0 * x.sum() * torch.ones_like(grad)
    return grad

def kdv(u, input):
    grad_u = gradient(u, input)
    u_t = grad_u[:, 0:1]
    u_x = grad_u[:, 1:2]

    grad_u_x = gradient(u_x, input)
    u_xx = grad_u_x[:, 1:2]

    grad_u_xx = gradient(u_xx, input)
    u_xxx = grad_u_xx[:, 1:2]

    # Conserved densities (free)
    rho_1 = 0.5 * u**2 #momentum
    rho_2 = -u**3 + 0.5 * u_x**2 #energy
    rho_3 = 2.5 * u**4 + 5 * u * u_x**2 + 0.5 * u_xx**2 #H_32

    # Conservation form flux
    J_0 = 3 * u**2 + u_xx
    J_1 = -4.5*u**4 - 3*u**2*u_xx + 6*u*u_x**2 + u_x*u_xxx - 0.5*u_xx**2

    J_0_x = gradient(J_0, input)[:, 1:2]
    J_1_x = gradient(J_1, input)[:, 1:2]
    rho_2_t = gradient(rho_2,input)[:,0:1]

    res_KDV = u_t + 6 * u * u_x + u_xxx
    res_H0 = u_t + J_0_x
    res_H1 = rho_2_t + J_1_x

    results = {}
    results['u'] = u
    results['u_t'] = u_t
    results['u_x'] = u_x
    results['u_xx'] = u_xx
    results['u_xxx'] = u_xxx
    results['rho_1'] = rho_1
    results['rho_2'] = rho_2
    results['rho_3'] = rho_3
    results['J_0'] = J_0
    results['J_1'] = J_1
    results['res_KDV'] = res_KDV
    results['res_H0'] = res_H0
    results['res_H1'] = res_H1

    return results

def init_metrics():
    base_keys = ['u', 'u_t', 'u_x', 'u_xx', 'u_xxx', 'rho_1', 'rho_2', 'rho_3', 'J_0', 'J_1', 'res_KDV', 'res_H0', 'res_H1']
    return {f'mean_{k}': [] for k in base_keys}


def compute_conserved_integrals(model, t_values, config, num_x_points=500, device='cpu'):
    import numpy as np

    model.eval()
    model.to(device)

    x = np.linspace(-config.L, config.L, num_x_points)
    dx = x[1] - x[0]

    H_0, H_1, H_2 = [], [], []

    for t in t_values:
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        t_tensor = torch.ones_like(x_tensor) * t
        input_tensor = torch.stack([t_tensor, x_tensor], dim=1)
        input_tensor.requires_grad_(True)

        u = model(input_tensor)
        results = kdv(u, input_tensor)

        # Integrate using trapezoidal rule
        rho_0 = results['u'].detach().cpu().numpy().flatten()
        rho_1 = results['rho_1'].detach().cpu().numpy().flatten()
        rho_2 = results['rho_2'].detach().cpu().numpy().flatten()

        H_0.append(np.trapz(rho_0, dx=dx))
        H_1.append(np.trapz(rho_1, dx=dx))
        H_2.append(np.trapz(rho_2, dx=dx))

    return {
        't_values': np.array(t_values),
        'H_0': np.array(H_0),
        'H_1': np.array(H_1),
        'H_2': np.array(H_2)
    }
  

## KDV Soliton Solutions
def kdv_soliton_sech(input, config):
    t = input[:, 0:1]
    x = input[:, 1:2]

    kappa = config.kappa
    x0 = config.x0

    # Velocity of soliton: v = 4κ²
    velocity = 4 * kappa**2

    # Argument: κ(x - vt - x₀)
    arg = kappa * (x - velocity * t - x0)

    # Amplitude: 2κ²sech²(arg)
    u = 2 * kappa**2 / (torch.cosh(arg)**2)

    return u


def kdv_cnoidal_wave(input, config):
    try:
        from scipy.special import ellipj
    except ImportError:
        raise ImportError("scipy is required for cnoidal wave computation")

    t = input[:, 0:1]
    x = input[:, 1:2]

    kappa = config.kappa
    m = config.m
    x0 = config.x0

    # Velocity: v = (4κ²/3)(2m - 1)
    velocity = (4 * kappa**2 / 3) * (2 * m - 1)

    # Argument: κ(x - vt - x₀)
    arg = kappa * (x - velocity * t - x0)

    # Compute Jacobi elliptic function cn(arg | m)
    # ellipj returns (sn, cn, dn, ph)
    arg_np = arg.detach().cpu().numpy()
    _, cn_vals, _, _ = ellipj(arg_np, m)

    # Convert back to torch tensor
    cn_tensor = torch.tensor(cn_vals, dtype=input.dtype, device=input.device)

    # Amplitude: -2κ²m·cn²(arg | m)
    u = -2 * kappa**2 * m * cn_tensor**2

    # Ensure output shape is (N, 1)
    if u.dim() == 1:
        u = u.unsqueeze(-1)

    return u

def gaussian_blob(input, config):
    t = input[:, 0:1]
    x = input[:, 1:2]

    kappa = config.kappa
    velocity = 4 * kappa**2
    x0 = config.x0
    # Argument: κ(x - vt - x₀)
    arg = kappa * (x - velocity * t - x0)
    u = 3 * torch.exp(-arg**2)
    return u

soliton_library = {
    'sech' : kdv_soliton_sech,
    'cn' : kdv_cnoidal_wave,
    'gauss' : gaussian_blob
    }