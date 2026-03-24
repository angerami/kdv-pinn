import torch

def sample_coordinate(config, resample=False, num_samp=-1, time=False):
    if num_samp < 0:
        num_samp = config.num_samp_bulk

    if time:
        T = config.T
        coord = torch.linspace(0, T, num_samp)
        if resample:
            coord = coord + (torch.rand(num_samp) * 2 - 1) * T / num_samp
            coord = torch.clamp(coord, 0, T)
    else:
        L = config.L
        coord = torch.linspace(-L, L, num_samp)
        if resample:
            coord = coord + (torch.rand(num_samp) * 2 - 1) * L / num_samp

    return coord

def sample_bulk(config, resample=False, num_samp=-1):
    x = sample_coordinate(config, resample, num_samp, time=False)
    t = sample_coordinate(config, resample, num_samp, time=True)

    # Create meshgrid: tt[i,j] = t[i], xx[i,j] = x[j]
    tt, xx = torch.meshgrid(t, x, indexing='ij')
    return torch.stack([tt.flatten(), xx.flatten()], dim=1)  # N² x 2


def sample_slice(zval, config, resample=False, num_samp=-1, time=True):
    if num_samp < 0:
        num_samp = config.num_samp_bulk
    if time:
        x = sample_coordinate(config, resample, num_samp, time=False)
        t = zval * torch.ones_like(x)
    else:
        t = sample_coordinate(config, resample, num_samp, time=True)
        x = zval * torch.ones_like(t)
    return torch.stack([t, x], dim=1)  # N x 2

def sample_boundary(config, resample=False, num_samp=-1, ic_only=False, bc_only=False):
    L = config.L
    input_0 = sample_slice(0, config, resample, num_samp, time=True)
    input_left = sample_slice(-L, config, resample, num_samp, time=False)
    input_right = sample_slice(L, config, resample, num_samp, time=False)

    if ic_only:
        return input_0
    elif bc_only:
        return torch.cat([input_left, input_right], dim=0)
    else:
        return torch.cat([input_0, input_left, input_right], dim=0)

def extract_time_slices(u, input_tensor, t_values, config, num_x_points=300):
    import numpy as np

    x = np.linspace(-config.L, config.L, num_x_points)
    slices = {'t_values': t_values, 'x': x, 'u_slices': {}}

    # Extract t from input_tensor
    t = input_tensor[:, 0].detach().cpu().numpy()
    u_np = u.detach().cpu().numpy().flatten()

    for t_val in t_values:
        # Find indices where t is close to t_val
        mask = np.isclose(t, t_val, atol=1e-6)
        slices['u_slices'][t_val] = u_np[mask]

    return slices

def extract_density_slices(results, input_tensor, t_values, config, num_x_points=300):
    import numpy as np

    x = np.linspace(-config.L, config.L, num_x_points)
    slices = {
        't_values': t_values, 'x': x,
        'rho_0_slices': {}, 'rho_1_slices': {}, 'rho_2_slices': {}
    }

    # Extract t from input_tensor
    t = input_tensor[:, 0].detach().cpu().numpy()

    for t_val in t_values:
        # Find indices where t is close to t_val
        mask = np.isclose(t, t_val, atol=1e-6)

        slices['rho_0_slices'][t_val] = results['u'].detach().cpu().numpy().flatten()[mask]
        slices['rho_1_slices'][t_val] = results['rho_1'].detach().cpu().numpy().flatten()[mask]
        slices['rho_2_slices'][t_val] = results['rho_2'].detach().cpu().numpy().flatten()[mask]

    return slices
