"""Physics operators and conserved quantities for the KdV equation."""
import torch


def gradient(y, x):
    """Compute gradient of y with respect to x using autograd.

    Handles edge cases where y doesn't depend on x and ensures
    gradient tracking is maintained for higher-order derivatives.

    Args:
        y: Output tensor
        x: Input tensor with requires_grad=True

    Returns:
        Gradient dy/dx with requires_grad=True
    """
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]

    if grad is None:
        grad = torch.zeros_like(x)

    # Ensure gradient maintains requires_grad for further differentiation
    if not grad.requires_grad and x.requires_grad:
        grad = grad + 0.0 * x.sum() * torch.ones_like(grad)

    return grad


def kdv(u, input):
    """Compute KdV operator, derivatives, and conserved quantities.

    Computes the KdV residual u_t + 6u*u_x + u_xxx and all conserved
    densities and fluxes for the KdV hierarchy.

    Args:
        u: Field values, shape (N, 1)
        input: Input coordinates (t, x), shape (N, 2) with requires_grad=True

    Returns:
        Dictionary containing:
            - Derivatives: u_t, u_x, u_xx, u_xxx
            - Conserved densities: rho_1 (momentum), rho_2 (energy), rho_3
            - Fluxes: J_0, J_1, J_2
            - Residuals: res_KDV, res_H0, res_H1, res_H2
    """
    # Compute spatial and temporal derivatives
    grad_u = gradient(u, input)
    u_t = grad_u[:, 0:1]
    u_x = grad_u[:, 1:2]

    grad_u_x = gradient(u_x, input)
    u_xx = grad_u_x[:, 1:2]

    grad_u_xx = gradient(u_xx, input)
    u_xxx = grad_u_xx[:, 1:2]

    # Conserved densities
    rho_1 = 0.5 * u**2  # Momentum
    rho_2 = -u**3 + 0.5 * u_x**2  # Energy
    rho_3 = 2.5 * u**4 + 5 * u * u_x**2 + 0.5 * u_xx**2  # H_3/2

    # Conservation law fluxes
    J_0 = 3 * u**2 + u_xx
    J_1 = 2 * u**3 + u * u_xx - u_x**2 / 2
    J_2 = -4.5 * u**4 - 3 * u**2 * u_xx + 6 * u * u_x**2 + u_x * u_xxx - 0.5 * u_xx**2

    # Flux derivatives
    J_0_x = gradient(J_0, input)[:, 1:2]
    J_1_x = gradient(J_1, input)[:, 1:2]
    J_2_x = gradient(J_2, input)[:, 1:2]
    rho_1_t = gradient(rho_1, input)[:, 0:1]
    rho_2_t = gradient(rho_2, input)[:, 0:1]

    # PDE residuals
    res_KDV = u_t + 6 * u * u_x + u_xxx  # KdV equation
    res_H0 = u_t + J_0_x  # Mass conservation
    res_H1 = rho_1_t + J_1_x  # Mass conservation
    res_H2 = rho_2_t + J_2_x  # Energy conservation

    return {
        'u': u,
        'u_t': u_t,
        'u_x': u_x,
        'u_xx': u_xx,
        'u_xxx': u_xxx,
        'rho_1': rho_1,
        'rho_2': rho_2,
        'rho_3': rho_3,
        'J_0': J_0,
        'J_1': J_1,
        'J_2': J_2,
        'res_KDV': res_KDV,
        'res_H0': res_H0,
        'res_H1': res_H1,
        'res_H2': res_H2,
    }


def init_metrics():
    """Initialize dictionary for tracking field statistics during training."""
    base_keys = ['u', 'u_t', 'u_x', 'u_xx', 'u_xxx', 'rho_1', 'rho_2',
                 'rho_3', 'J_0', 'J_1', 'J_2', 'res_KDV', 'res_H0', 'res_H1','res_H2']
    return {f'mean_{k}': [] for k in base_keys}


def compute_conserved_integrals(model, t_values, config, num_x_points=500, device='cpu'):
    """Compute spatial integrals of conserved quantities over time.

    Integrates the conserved densities over the spatial domain at multiple
    time points to verify conservation laws.

    Args:
        model: Neural network model
        t_values: Time values at which to evaluate
        config: Configuration object with domain parameters
        num_x_points: Number of spatial points for integration
        device: PyTorch device

    Returns:
        Dictionary with time values and integrated quantities H_0, H_1, H_2
    """
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


# Analytic KdV soliton solutions

def kdv_soliton_sech(input, config):
    """Single soliton solution: u = 2κ² sech²(κ(x - 4κ²t - x₀))."""
    t = input[:, 0:1]
    x = input[:, 1:2]

    kappa = config.kappa
    x0 = config.x0
    velocity = 4 * kappa**2

    arg = kappa * (x - velocity * t - x0)
    u = 2 * kappa**2 / (torch.cosh(arg)**2)

    return u


def kdv_cnoidal_wave(input, config):
    """Cnoidal wave solution using Jacobi elliptic function.

    Periodic traveling wave solution parameterized by elliptic modulus m.
    Requires scipy for elliptic function evaluation.
    """
    try:
        from scipy.special import ellipj
    except ImportError:
        raise ImportError("scipy is required for cnoidal wave computation")

    t = input[:, 0:1]
    x = input[:, 1:2]

    kappa = config.kappa
    m = config.m
    x0 = config.x0
    velocity = (4 * kappa**2 / 3) * (2 * m - 1)

    arg = kappa * (x - velocity * t - x0)
    arg_np = arg.detach().cpu().numpy()
    _, cn_vals, _, _ = ellipj(arg_np, m)

    cn_tensor = torch.tensor(cn_vals, dtype=input.dtype, device=input.device)
    u = -2 * kappa**2 * m * cn_tensor**2

    if u.dim() == 1:
        u = u.unsqueeze(-1)

    return u


def gaussian_blob(input, config):
    """Gaussian initial condition (not an exact solution, for testing)."""
    t = input[:, 0:1]
    x = input[:, 1:2]

    kappa = config.kappa
    x0 = config.x0
    velocity = 4 * kappa**2

    arg = kappa * (x - velocity * t - x0)
    u = 3 * torch.exp(-arg**2)

    return u


soliton_library = {
    'sech': kdv_soliton_sech,
    'cn': kdv_cnoidal_wave,
    'gauss': gaussian_blob
}