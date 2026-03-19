import torch
## Sampling functions
def sample_x(config, resample=False, num_samp=-1):
    L = config.L
    if num_samp < 0:
        num_samp = config.num_samp_bulk
    x = torch.linspace(-L, L, num_samp)
    if resample:
        x = x + (torch.rand(num_samp) * 2 - 1) * L / num_samp
    return x

def sample_t(config, resample=False, num_samp=-1):
    T = config.T
    # Time domain: [0, T]
    t = torch.linspace(0, T, num_samp)
    if resample:
        t = t + (torch.rand(num_samp) * 2 - 1) * T / num_samp
        t = torch.clamp(t, 0, T)  # Keep within [0, T]
    return T

def sample_bulk(config, resample=False, num_samp=-1):
    x = sample_x(config, resample, num_samp)
    t = sample_t(config, resample, num_samp)

    # Create meshgrid: tt[i,j] = t[i], xx[i,j] = x[j]
    tt, xx = torch.meshgrid(t, x, indexing='ij')
    return torch.stack([tt.flatten(), xx.flatten()], dim=1)  # N² x 2


def sample_slice(zval, config, resample=False, num_samp=-1, time=True):
    #Clunky 
    if time:
        x = sample_x(config, resample, num_samp)
        t = zval * torch.ones(num_samp)
    else:
        x = zval * torch.ones(num_samp)
        t = sample_t(config, resample, num_samp)
    return torch.stack([t, x], dim=1)  # N x 2

def sample_boundary(config, resample=False, num_samp=-1):
    L = config.L
    input_0 = sample_slice(0, config, resample, num_samp,time=True)
    input_left = sample_slice(-L, config, resample, num_samp,time=False)
    input_right = sample_slice(L, config, resample, num_samp,time=False)
    return torch.cat([input_0, input_left, input_right], dim=0)
    #T = config.T
    #input_T = sample_slice(T, config, resample, num_samp,time=True)
    #return torch.cat([input_0, input_T, input_left, input_right], dim=0)
