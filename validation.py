"""Validation script for training and evaluating KdV PINN."""
import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import KdV_pinn
from scattering import ScatteringData, SchrodingerSolver
from train import train_pinn, pretrain
from sampling import sample_bulk
from configuration import kdv_config, config_to_dict
from plot_utils import plot_field_visualization, plot_results_2panel
from physics import kdv


def _fresh_eval_grid(config, device):
    """Create a fresh evaluation grid with gradients enabled.

    Returns a new tensor each time, so previous computational graphs
    are not kept alive through a shared input tensor.
    """
    inp = sample_bulk(config, resample=False, num_samp=config.num_samp_eval).to(device)
    inp.requires_grad_(True)
    return inp


def _free_graph():
    """Force-free computational graph memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def plot_scattering_validation(sd, kappa_rec, eigenvector_stack, eigenvalue_stack, tvals, dx, output_dir, squared=True):
    """Plot eigenfunction evolution and recovered wave numbers.

    Args:
        sd: ScatteringData object
        kappa_rec: Recovered kappa values over time
        eigenvector_stack: Time series of eigenfunctions
        eigenvalue_stack: Time series of eigenvalues
        tvals: Time values
        dx: Spatial grid spacing
        output_dir: Directory for saving plots
        squared: If True, plot |ψ|², else plot ψ
    """
    vmin, vmax = (0, 0.1) if squared else (-0.3, 0.3)
    label_suffix = '^2' if squared else ''

    # Plot eigenfunctions
    fig = plt.figure(figsize=(12, 3))
    axes = []

    for ev_idx in range(sd.Ns):
        ax = fig.add_subplot(1, sd.Ns, ev_idx + 1)
        axes.append(ax)
        eigenvector_timeseries = eigenvector_stack[:, :, ev_idx]
        if squared:
            eigenvector_timeseries = eigenvector_timeseries**2
        im = ax.imshow(eigenvector_timeseries,
                       vmin=vmin, vmax=vmax,
                       origin='lower', cmap='coolwarm')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_title(f'$\\psi_{{{ev_idx}}}{label_suffix}(x,t)$')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    suffix = '_squared' if squared else ''
    plt.savefig(f'{output_dir}/eigenvectors{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Single-panel kappa recovery plot
    _, ax = plt.subplots(figsize=(8, 4))
    for idx in range(sd.Ns):
        line, = ax.plot(tvals[1:], kappa_rec[:, idx], label=f'$\\kappa_{{{idx}}}$ (recovered)')
        ax.axhline(sd.kappas[idx], color=line.get_color(), linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(sd.kappas) * 1.3)
    ax.set_ylabel('$\\kappa$')
    ax.set_xlabel('$t$')
    ax.legend()
    ax.set_title('Recovered Wave Numbers')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kappa_recovery.png', dpi=150, bbox_inches='tight')
    plt.close()


def compute_metrics(sd, kappa_rec, results, results_ana):
    """Compute validation metrics comparing PINN to analytic solution.

    Args:
        sd: ScatteringData object
        kappa_rec: Recovered kappa values
        results: PINN field results (detached)
        results_ana: Analytic field results (detached)

    Returns:
        Dictionary of metrics
    """
    # Input parameters
    input_params = {
        'n_solitons': sd.Ns,
    }
    for i in range(sd.Ns):
        input_params[f'kappa_{i}_input'] = sd.kappas[i]
        input_params[f'x0_{i}_input'] = sd.x0[i]
        input_params[f'c0_{i}_input'] = sd.c0[i]

    # Kappa recovery metrics
    kappa_metrics = {}
    for i in range(sd.Ns):
        kappa_mean = kappa_rec[:, i].mean()
        kappa_std = kappa_rec[:, i].std()
        kappa_true = sd.kappas[i]
        kappa_metrics[f'kappa_{i}_mean'] = kappa_mean
        kappa_metrics[f'kappa_{i}_std'] = kappa_std
        kappa_metrics[f'kappa_{i}_error'] = np.abs(kappa_mean - kappa_true)

    # Field errors
    field_errors = {}
    for key in ['u', 'u_t', 'u_x']:
        if key in results and key in results_ana:
            res_val = results[key].detach() if results[key].requires_grad else results[key]
            ana_val = results_ana[key].detach() if results_ana[key].requires_grad else results_ana[key]
            diff = (res_val - ana_val).cpu().numpy()
            field_errors[f'{key}_mae'] = np.abs(diff).mean()
            field_errors[f'{key}_rmse'] = np.sqrt((diff**2).mean())

    return {**input_params, **kappa_metrics, **field_errors}


def validate_run(output_dir='validation_output'):
    """Run complete validation: pretrain, train, and verify scattering data.

    Performs a full training run on a multi-soliton problem, then validates
    that the learned solution preserves scattering data (isospectrality).

    Memory management: each evaluation block (analytic, pretrained, trained)
    uses a fresh input tensor and explicitly frees the computational graph
    afterward, preventing memory accumulation from stacked autograd graphs.

    Args:
        output_dir: Directory for saving plots and results

    Returns:
        Dictionary of validation metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device.")

    training_config = kdv_config
    training_config.num_epochs = 5000
    training_config.num_pretrain_epochs = 2500
    training_config.num_samp_bulk = 96
    training_config.num_samp_eval = 256
    training_config.plot_interval = 100
    training_config.MLP = [2, 128, 128, 128, 1]
    training_config.kappas = [1.8, 1.3]
    training_config.x0s = [8, 7]
    training_config.lr = 1e-3
    training_config.T = 1
    training_config.L = 10
    training_config.Lmax = 15
    training_config.lambda_BC = 1
    training_config.lambda_kdv = 1
    training_config.vmin = 0
    training_config.vmax = 3
    torch.manual_seed(training_config.seed)
    print(f"Lmax = {training_config.Lmax}")

    print("=== Analytic Solution ===")
    sd = ScatteringData(training_config.kappas, training_config.x0s, R=None, use_tau=True)

    print("=== Pretraining ===")
    model = KdV_pinn(training_config).to(device)
    pretrain(model, sd.forward_fcn, training_config, device, num_epochs=training_config.num_pretrain_epochs)

    print("=== Training PINN ===")
    result = train_pinn(model, training_config, device, interactive=False, save_plot=f'{output_dir}/training_progress.png')
    print("Training complete!")

    # ---- Block 1: Analytic solution plots ----
    # Each block gets a fresh input_eval so the previous graph can be freed.
    print("=== Analytic Solution Plots ===")
    input_eval = _fresh_eval_grid(training_config, device)
    t = input_eval[:, 0:1]
    x = input_eval[:, 1:2]

    u_ana = sd.u(x, t)
    results_ana = kdv(u_ana, input_eval)
    plot_field_visualization(results_ana, training_config, 'deriv', f'{output_dir}/analytic_deriv.png', 'Analytic Solution')
    plot_field_visualization(results_ana, training_config, 'res', f'{output_dir}/analytic_res.png', 'Analytic Solution')

    # Keep detached copy for metrics; free the graph
    results_ana_detached = {k: v.detach().clone() for k, v in results_ana.items()}
    del results_ana, u_ana, input_eval, t, x
    _free_graph()

    # ---- Block 2: Pretrained model plots ----
    print("=== Pretrained Model Plots ===")
    input_eval = _fresh_eval_grid(training_config, device)

    u_0 = model(input_eval)
    results_0 = kdv(u_0, input_eval)
    plot_field_visualization(results_0, training_config, 'deriv', f'{output_dir}/pretrain_deriv.png', 'Pretrained Model')
    plot_field_visualization(results_0, training_config, 'res', f'{output_dir}/pretrain_res.png', 'Pretrained Model')

    del results_0, u_0, input_eval
    _free_graph()

    # ---- Block 3: 2-panel training summary ----
    print("=== Training Summary Plot ===")
    input_eval = _fresh_eval_grid(training_config, device)
    plot_results_2panel(result['model'], input_eval, training_config, result['metrics'],
                        filename=f'{output_dir}/training_summary_2panel.png')
    del input_eval
    _free_graph()

    # Save model
    model_path = f'{output_dir}/pinn_model.pt'
    torch.save({
        'model_state_dict': result['model'].state_dict(),
        'optimizer_state_dict': result['optimizer'].state_dict(),
        'config': config_to_dict(training_config),
        'metrics': result['metrics']
    }, model_path)
    print(f"Model saved to {model_path}")

    # ---- Block 4: Trained model full evaluation ----
    print("=== Trained Model Validation ===")
    input_eval = _fresh_eval_grid(training_config, device)

    u = model(input_eval)
    results = kdv(u, input_eval)

    plot_field_visualization(results, training_config, 'deriv', f'{output_dir}/trained_deriv.png', 'Trained Model')
    plot_field_visualization(results, training_config, 'res', f'{output_dir}/trained_res.png', 'Trained Model')
    plot_field_visualization(results, training_config, 'iom', f'{output_dir}/trained_iom.png', 'Trained Model')

    # Error plots (results vs analytic)
    delta = {k: (results[k].detach() - results_ana_detached[k]) for k in results}
    plot_field_visualization(delta, training_config, 'deriv', f'{output_dir}/error_deriv.png', 'Error (Trained - Analytic)')
    plot_field_visualization(delta, training_config, 'res', f'{output_dir}/error_res.png', 'Error (Trained - Analytic)')

    # Keep detached for metrics; free the big graph
    results_detached = {k: v.detach().clone() for k, v in results.items()}
    u_detached = results_detached['u']
    del results, u, delta, input_eval
    _free_graph()

    # ---- Block 5: Inverse scattering (no autograd needed) ----
    print("=== Inverse Scattering ===")
    # Use a fresh grid just to get the coordinate arrays (no grad needed)
    with torch.no_grad():
        coord_grid = sample_bulk(training_config, resample=False, num_samp=training_config.num_samp_eval).to(device)
    t_grid = coord_grid[:, 0:1]
    x_grid = coord_grid[:, 1:2]

    u_reshape = u_detached.reshape(training_config.num_samp_eval, training_config.num_samp_eval).cpu().numpy()
    tvals = t_grid.reshape(training_config.num_samp_eval, training_config.num_samp_eval).cpu().numpy()[:, 0]
    x_d = x_grid.reshape(training_config.num_samp_eval, training_config.num_samp_eval)[0, :].cpu().numpy()
    dx = x_d[1] - x_d[0]

    del coord_grid, t_grid, x_grid

    S = SchrodingerSolver()
    eigenvector_stack, eigenvalue_stack = S.solve(u_reshape, dx)
    kappa_rec = np.sqrt(-eigenvalue_stack[:, :sd.Ns] + 1e-8)

    plot_scattering_validation(sd, kappa_rec, eigenvector_stack, eigenvalue_stack, tvals, dx, output_dir, squared=True)
    plot_scattering_validation(sd, kappa_rec, eigenvector_stack, eigenvalue_stack, tvals, dx, output_dir, squared=False)

    # ---- Metrics ----
    metrics = compute_metrics(sd, kappa_rec, results_detached, results_ana_detached)
    print("\n=== Validation Metrics ===")
    for key, val in metrics.items():
        if isinstance(val, int):
            print(f"{key}: {val}")
        else:
            print(f"{key}: {val:.3e}")

    metrics_path = f'{output_dir}/metrics.txt'
    with open(metrics_path, 'w') as f:
        for key, val in metrics.items():
            if isinstance(val, int):
                f.write(f"{key}: {val}\n")
            else:
                f.write(f"{key}: {val:.6e}\n")
    print(f"\nMetrics saved to {metrics_path}")

    return metrics

if __name__ == "__main__":
    validate_run()
