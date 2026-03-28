"""Validation library for training and evaluating KdV PINN."""
import os
import gc
import torch
import numpy as np
from types import SimpleNamespace
from models import KdV_pinn
from scattering import ScatteringData, SchrodingerSolver
from train import train_pinn, pretrain
from sampling import sample_bulk
from configuration import kdv_config, config_to_dict
from plot_utils import (
    plot_field_visualization,
    plot_results_2panel,
    plot_scattering_validation,
    generate_animation
)
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


def compute_metrics(sd, kappa_rec, results, results_ana):
    """Compute validation metrics comparing PINN to analytic solution.

    Args:
        sd: ScatteringData object
        kappa_rec: Recovered kappa values (sorted largest to smallest)
        results: PINN field results (detached)
        results_ana: Analytic field results (detached)

    Returns:
        Dictionary of metrics

    Note:
        kappa_rec is sorted (largest first) to match eigenvalue solver output.
        We sort sd.kappas to match for metric computation.
    """
    # Sort kappas to match kappa_rec ordering
    kappas_sorted = sorted(sd.kappas, reverse=True)

    # Input parameters (store original unsorted values)
    input_params = {
        'n_solitons': sd.Ns,
    }
    for i in range(sd.Ns):
        input_params[f'kappa_{i}_input'] = sd.kappas[i]
        input_params[f'x0_{i}_input'] = sd.x0[i]
        input_params[f'c0_{i}_input'] = sd.c0[i]

    # Kappa recovery metrics (compare sorted values)
    kappa_metrics = {}
    for i in range(sd.Ns):
        kappa_mean = kappa_rec[:, i].mean()
        kappa_std = kappa_rec[:, i].std()
        kappa_true = kappas_sorted[i]  # Use sorted kappas
        kappa_metrics[f'kappa_{i}_mean'] = kappa_mean
        kappa_metrics[f'kappa_{i}_std'] = kappa_std
        kappa_metrics[f'kappa_{i}_true'] = kappa_true  # Store which true value this corresponds to
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


def validate_from_checkpoint(checkpoint_path, output_dir=None, generate_anim=False):
    """Load a saved model checkpoint and regenerate all validation outputs without re-training.

    Args:
        checkpoint_path: Path to saved model checkpoint (.pt file)
        output_dir: Directory for saving outputs. If None, uses same directory as checkpoint.
        generate_anim: If True, generate animations (default: False).
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config_dict = checkpoint['config']

    # Reconstruct config as SimpleNamespace
    training_config = SimpleNamespace(**config_dict)

    # Ensure animation params exist (for backward compatibility)
    if not hasattr(training_config, 'anim_fps'):
        training_config.anim_fps = 20
    if not hasattr(training_config, 'anim_max_frames'):
        training_config.anim_max_frames = 100

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device.")

    # Create model and load weights
    model = KdV_pinn(training_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded. Config: kappas={training_config.kappas}, x0s={training_config.x0s}")
    print(f"Output directory: {output_dir}")

    # Create scattering data
    sd = ScatteringData(training_config.kappas, training_config.x0s, R=None, use_tau=True)

    # ---- Block 1: Analytic solution plots ----
    print("=== Analytic Solution Plots ===")
    input_eval = _fresh_eval_grid(training_config, device)
    t = input_eval[:, 0:1]
    x = input_eval[:, 1:2]

    u_ana = sd.u(x, t)
    results_ana = kdv(u_ana, input_eval)
    plot_field_visualization(results_ana, training_config, 'deriv', f'{output_dir}/analytic_deriv.png', 'Analytic Solution')
    plot_field_visualization(results_ana, training_config, 'res', f'{output_dir}/analytic_res.png', 'Analytic Solution')

    # Keep detached copy for metrics
    results_ana_detached = {k: v.detach().clone() for k, v in results_ana.items()}
    del results_ana, u_ana, input_eval, t, x
    _free_graph()

    # ---- Block 2: Trained model evaluation ----
    print("=== Trained Model Validation ===")
    input_eval = _fresh_eval_grid(training_config, device)

    u = model(input_eval)
    results = kdv(u, input_eval)

    plot_field_visualization(results, training_config, 'deriv', f'{output_dir}/trained_deriv.png', 'Trained Model')
    plot_field_visualization(results, training_config, 'res', f'{output_dir}/trained_res.png', 'Trained Model')
    plot_field_visualization(results, training_config, 'iom', f'{output_dir}/trained_iom.png', 'Trained Model')

    # Error plots
    delta = {k: (results[k].detach() - results_ana_detached[k]) for k in results}
    plot_field_visualization(delta, training_config, 'deriv', f'{output_dir}/error_deriv.png', 'Error (Trained - Analytic)')
    plot_field_visualization(delta, training_config, 'res', f'{output_dir}/error_res.png', 'Error (Trained - Analytic)')

    # Keep detached for metrics
    results_detached = {k: v.detach().clone() for k, v in results.items()}
    u_detached = results_detached['u']
    del results, u, delta, input_eval
    _free_graph()

    # ---- Block 3: Inverse scattering ----
    print("=== Inverse Scattering ===")
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

    # ---- Animation (optional) ----
    if generate_anim:
        print("=== Generating Animations ===")
        generate_animation(u_reshape, eigenvector_stack, eigenvalue_stack, tvals, x_d, sd,
                          f'{output_dir}/animation.gif', squared=False,
                          fps=training_config.anim_fps, max_frames=training_config.anim_max_frames)
        generate_animation(u_reshape, eigenvector_stack, eigenvalue_stack, tvals, x_d, sd,
                          f'{output_dir}/animation_squared.gif', squared=True,
                          fps=training_config.anim_fps, max_frames=training_config.anim_max_frames)

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
    print(f"All validation outputs saved to {output_dir}")


def validate_run(config=None, output_dir=None, generate_anim=False):
    """Run complete validation: pretrain, train, and verify scattering data.

    Performs a full training run on a multi-soliton problem, then validates
    that the learned solution preserves scattering data (isospectrality).

    Memory management: each evaluation block (analytic, pretrained, trained)
    uses a fresh input tensor and explicitly frees the computational graph
    afterward, preventing memory accumulation from stacked autograd graphs.

    Args:
        config: Configuration object (SimpleNamespace). If None, uses kdv_config with defaults.
        output_dir: Directory for saving plots and results. If None, reads from config or uses default.
        generate_anim: If True, generate animation of potential and eigenfunctions (default: False).
                      Animation parameters (fps, max_frames) are read from config.

    Returns:
        Dictionary of validation metrics
    """
    # Use provided config or default
    if config is None:
        training_config = SimpleNamespace(**vars(kdv_config))
        # Set defaults if not already configured
        if not hasattr(training_config, '_configured'):
            training_config.num_epochs = 5000
            training_config.num_pretrain_epochs = 2500
            training_config.num_samp_bulk = 96
            training_config.num_samp_eval = 512
            training_config.plot_interval = 50
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
    else:
        training_config = config

    # Determine output directory
    if output_dir is None:
        if hasattr(training_config, 'output_dir'):
            output_dir = training_config.output_dir
        else:
            output_dir = 'validation_output'

    os.makedirs(output_dir, exist_ok=True)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device.")

    torch.manual_seed(training_config.seed)
    print(f"Using config: kappas={training_config.kappas}, x0s={training_config.x0s}")
    print(f"Lmax = {training_config.Lmax}")
    print(f"Output directory: {output_dir}")

    print("=== Analytic Solution ===")
    sd = ScatteringData(training_config.kappas, training_config.x0s, R=None, use_tau=True)

    # ---- Block 1: Analytic solution plots ----
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

    # ---- Block 2: Pretrain and evaluate ----
    print("=== Pretraining ===")
    model = KdV_pinn(training_config).to(device)
    pretrain(model, sd.forward_fcn, training_config, device, num_epochs=training_config.num_pretrain_epochs)

    print("=== Pretrained Model Plots ===")
    input_eval = _fresh_eval_grid(training_config, device)

    u_0 = model(input_eval)
    results_0 = kdv(u_0, input_eval)
    plot_field_visualization(results_0, training_config, 'deriv', f'{output_dir}/pretrain_deriv.png', 'Pretrained Model')
    plot_field_visualization(results_0, training_config, 'res', f'{output_dir}/pretrain_res.png', 'Pretrained Model')

    del results_0, u_0, input_eval
    _free_graph()

    # ---- Block 3: Train PINN ----
    print("=== Training PINN ===")
    result = train_pinn(model, training_config, device, interactive=False, save_plot=f'{output_dir}/training_progress.png')
    print("Training complete!")

    # ---- Block 4: 2-panel training summary ----
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

    # ---- Block 5: Trained model full evaluation ----
    print("=== Trained Model Validation ===")
    input_eval = _fresh_eval_grid(training_config, device)

    u = result['model'](input_eval)
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

    # ---- Block 6: Inverse scattering (no autograd needed) ----
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

    # ---- Animation (optional) ----
    if generate_anim:
        print("=== Generating Animation ===")
        # Generate both regular and squared versions
        generate_animation(u_reshape, eigenvector_stack, eigenvalue_stack, tvals, x_d, sd,
                          f'{output_dir}/animation.gif', squared=False,
                          fps=training_config.anim_fps, max_frames=training_config.anim_max_frames)
        generate_animation(u_reshape, eigenvector_stack, eigenvalue_stack, tvals, x_d, sd,
                          f'{output_dir}/animation_squared.gif', squared=True,
                          fps=training_config.anim_fps, max_frames=training_config.anim_max_frames)

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


def validate_analytical_only(config=None, output_dir=None, generate_anim=False):
    """Generate validation outputs using only the analytical solution (no training).

    This mode is useful for fast checks of analytical distributions and scattering
    data while reusing existing machinery. No model training or pretraining is performed.

    Args:
        config: Configuration object (SimpleNamespace). If None, uses kdv_config with defaults.
        output_dir: Directory for saving plots and results. If None, uses default.
        generate_anim: If True, generate animation of analytical solution (default: False).
                      Animation parameters (fps, max_frames) are read from config.

    Returns:
        Dictionary of analytical metrics
    """
    # Use provided config or default
    if config is None:
        training_config = SimpleNamespace(**vars(kdv_config))
        # Set defaults if not already configured
        if not hasattr(training_config, '_configured'):
            training_config.num_samp_eval = 512
            training_config.kappas = [1.8, 1.3]
            training_config.x0s = [8, 7]
            training_config.T = 1
            training_config.L = 10
            training_config.Lmax = 15
            training_config.vmin = 0
            training_config.vmax = 3
    else:
        training_config = config

    # Ensure animation params exist
    if not hasattr(training_config, 'anim_fps'):
        training_config.anim_fps = 20
    if not hasattr(training_config, 'anim_max_frames'):
        training_config.anim_max_frames = 100

    # Determine output directory
    if output_dir is None:
        if hasattr(training_config, 'output_dir'):
            output_dir = training_config.output_dir
        else:
            output_dir = 'analytical_output'

    os.makedirs(output_dir, exist_ok=True)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device.")

    torch.manual_seed(training_config.seed)
    print(f"Using config: kappas={training_config.kappas}, x0s={training_config.x0s}")
    print(f"Output directory: {output_dir}")

    print("=== Analytic Solution ===")
    sd = ScatteringData(training_config.kappas, training_config.x0s, R=None, use_tau=True)

    # Evaluate analytical solution on grid
    input_eval = _fresh_eval_grid(training_config, device)
    t = input_eval[:, 0:1]
    x = input_eval[:, 1:2]

    u_ana = sd.u(x, t)
    results_ana = kdv(u_ana, input_eval)

    # Generate plots
    plot_field_visualization(results_ana, training_config, 'deriv', f'{output_dir}/analytic_deriv.png', 'Analytic Solution')
    plot_field_visualization(results_ana, training_config, 'res', f'{output_dir}/analytic_res.png', 'Analytic Solution')
    plot_field_visualization(results_ana, training_config, 'iom', f'{output_dir}/analytic_iom.png', 'Analytic Solution')

    # Keep detached copy for inverse scattering
    u_detached = results_ana['u'].detach().clone()
    del results_ana, u_ana, input_eval
    _free_graph()

    # ---- Inverse scattering ----
    print("=== Inverse Scattering ===")
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

    # ---- Animation (optional) ----
    if generate_anim:
        print("=== Generating Animation (Analytical Solution) ===")
        # Generate both regular and squared versions
        generate_animation(u_reshape, eigenvector_stack, eigenvalue_stack, tvals, x_d, sd,
                          f'{output_dir}/analytical_animation.gif', squared=False,
                          fps=training_config.anim_fps, max_frames=training_config.anim_max_frames)
        generate_animation(u_reshape, eigenvector_stack, eigenvalue_stack, tvals, x_d, sd,
                          f'{output_dir}/analytical_animation_squared.gif', squared=True,
                          fps=training_config.anim_fps, max_frames=training_config.anim_max_frames)

    # ---- Metrics ----
    metrics = {
        'n_solitons': sd.Ns,
    }
    for i in range(sd.Ns):
        metrics[f'kappa_{i}_input'] = sd.kappas[i]
        metrics[f'x0_{i}_input'] = sd.x0[i]
        metrics[f'c0_{i}_input'] = sd.c0[i]

    # Kappa recovery metrics
    kappas_sorted = sorted(sd.kappas, reverse=True)
    for i in range(sd.Ns):
        kappa_mean = kappa_rec[:, i].mean()
        kappa_std = kappa_rec[:, i].std()
        kappa_true = kappas_sorted[i]
        metrics[f'kappa_{i}_mean'] = kappa_mean
        metrics[f'kappa_{i}_std'] = kappa_std
        metrics[f'kappa_{i}_true'] = kappa_true
        metrics[f'kappa_{i}_error'] = np.abs(kappa_mean - kappa_true)

    print("\n=== Analytical Validation Metrics ===")
    for key, val in metrics.items():
        if isinstance(val, int):
            print(f"{key}: {val}")
        else:
            print(f"{key}: {val:.3e}")

    metrics_path = f'{output_dir}/analytical_metrics.txt'
    with open(metrics_path, 'w') as f:
        for key, val in metrics.items():
            if isinstance(val, int):
                f.write(f"{key}: {val}\n")
            else:
                f.write(f"{key}: {val:.6e}\n")
    print(f"\nMetrics saved to {metrics_path}")

    return metrics
