"""Sampling utilities for generating training and evaluation points."""
import torch


def sample_coordinate(config, resample=False, num_samp=-1, time=False):
    """Sample 1D coordinates with optional jittering.

    Args:
        config: Configuration with domain parameters (T, L)
        resample: If True, add random jitter to points
        num_samp: Number of samples (default: config.num_samp_bulk)
        time: If True, sample time domain [0, T], else space domain [-L, L]

    Returns:
        1D tensor of coordinates
    """
    if num_samp < 0:
        num_samp = config.num_samp_bulk

    if time:
        T = config.T
        coord = torch.linspace(0, T, num_samp)
        if resample:
            jitter = (torch.rand(num_samp) * 2 - 1) * T / num_samp
            coord = torch.clamp(coord + jitter, 0, T)
    else:
        L = config.L
        coord = torch.linspace(-L, L, num_samp)
        if resample:
            jitter = (torch.rand(num_samp) * 2 - 1) * L / num_samp
            coord = coord + jitter

    return coord


def sample_bulk(config, resample=False, num_samp=-1):
    """Sample spacetime bulk domain as (t, x) meshgrid.

    Args:
        config: Configuration with domain parameters
        resample: If True, add random jitter to grid points
        num_samp: Number of samples per dimension

    Returns:
        Tensor of shape (num_samp², 2) with columns [t, x]
    """
    t = sample_coordinate(config, resample, num_samp, time=True)
    x = sample_coordinate(config, resample, num_samp, time=False)

    tt, xx = torch.meshgrid(t, x, indexing='ij')
    return torch.stack([tt.flatten(), xx.flatten()], dim=1)


def sample_slice(zval, config, resample=False, num_samp=-1, time=True):
    """Sample a time or space slice.

    Args:
        zval: Constant value for the sliced dimension
        config: Configuration with domain parameters
        resample: If True, add random jitter to points
        num_samp: Number of samples
        time: If True, slice at constant time t=zval, else constant space x=zval

    Returns:
        Tensor of shape (num_samp, 2) with columns [t, x]
    """
    if num_samp < 0:
        num_samp = config.num_samp_bulk

    if time:
        x = sample_coordinate(config, resample, num_samp, time=False)
        t = zval * torch.ones_like(x)
    else:
        t = sample_coordinate(config, resample, num_samp, time=True)
        x = zval * torch.ones_like(t)

    return torch.stack([t, x], dim=1)


def sample_boundary(config, resample=False, num_samp=-1, ic_only=False, bc_only=False):
    """Sample initial and boundary conditions.

    Args:
        config: Configuration with domain parameters
        resample: If True, add random jitter to points
        num_samp: Number of samples per boundary
        ic_only: If True, return only initial condition (t=0)
        bc_only: If True, return only spatial boundaries (x=±L)

    Returns:
        Tensor of boundary points with shape (N, 2)
    """
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