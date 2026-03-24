import torch
from models import KdV_pinn
from scattering import ScatteringData, SchrodingerSolver
from train import train_pinn, pretrain
from sampling import sample_bulk
from configuration import kdv_config, config_to_dict, dict_to_config
from plot_utils import plot_field_visualization
from physics import kdv
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_scattering_validation(sd, kappa_rec, eigenvector_stack, tvals, output_dir, squared=True):
    vmin, vmax = (0, 0.1) if squared else (-0.3, 0.3)
    label_suffix = '^2' if squared else ''

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

    _, ax = plt.subplots(figsize=(8, 4))
    for idx in range(sd.Ns):
        line, = ax.plot(tvals[1:], kappa_rec[:, idx], label=f'$\\kappa_{{{idx}}}$ (recovered)')
        ax.axhline(sd.kappas[idx], color=line.get_color(), linestyle='--', alpha=0.7)
    ax.set_ylim(0, 2)
    ax.set_ylabel('$\\kappa$')
    ax.set_xlabel('$t$')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kappa_recovery.png', dpi=150, bbox_inches='tight')
    plt.close()

def compute_metrics(sd, kappa_rec, results, results_ana):
    kappa_metrics = {}
    for i in range(sd.Ns):
        kappa_mean = kappa_rec[:, i].mean()
        kappa_std = kappa_rec[:, i].std()
        kappa_true = sd.kappas[i]
        kappa_metrics[f'kappa_{i}_mean'] = kappa_mean
        kappa_metrics[f'kappa_{i}_std'] = kappa_std
        kappa_metrics[f'kappa_{i}_error'] = np.abs(kappa_mean - kappa_true)

    field_errors = {}
    for key in ['u', 'u_t', 'u_x']:
        if key in results and key in results_ana:
            diff = (results[key] - results_ana[key]).detach().cpu().numpy()
            field_errors[f'{key}_mae'] = np.abs(diff).mean()
            field_errors[f'{key}_rmse'] = np.sqrt((diff**2).mean())

    metrics = {
        **kappa_metrics,
        **field_errors
    }
    return metrics

def validate_run(output_dir='validation_output'):
    os.makedirs(output_dir, exist_ok=True)

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
    training_config.num_samp_eval = 128
    training_config.plot_interval = 50
    training_config.MLP = [2, 128, 128, 128, 1]
    training_config.kappas = [1.8, 1.3, 1,0.7]
    training_config.x0s = [8, 7, 5, 2]
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
    input_eval = sample_bulk(training_config, resample=False, num_samp=128).to(device)
    input_eval.requires_grad_(True)
    t = input_eval[:, 0:1]
    x = input_eval[:, 1:2]
    u_ana = sd.u(x, t)
    results_ana = kdv(u_ana, input_eval)
    plot_field_visualization(results_ana, training_config, 'deriv', f'{output_dir}/analytic_deriv.png', 'Analytic Solution')
    plot_field_visualization(results_ana, training_config, 'res', f'{output_dir}/analytic_res.png', 'Analytic Solution')

    print("=== Pretraining ===")
    model = KdV_pinn(training_config).to(device)
    pretrain(model, sd.forward_fcn, training_config, device, num_epochs=training_config.num_pretrain_epochs)
    u_0 = model(input_eval)
    results_0 = kdv(u_0, input_eval)
    plot_field_visualization(results_0, training_config, 'deriv', f'{output_dir}/pretrain_deriv.png', 'Pretrained Model')
    plot_field_visualization(results_0, training_config, 'res', f'{output_dir}/pretrain_res.png', 'Pretrained Model')

    print("=== Training PINN ===")
    result = train_pinn(model, training_config, device, interactive=False, save_plot=f'{output_dir}/training_progress.png')
    print("Training complete!")

    model_path = f'{output_dir}/pinn_model.pt'
    torch.save({
        'model_state_dict': result['model'].state_dict(),
        'optimizer_state_dict': result['optimizer'].state_dict(),
        'config': config_to_dict(training_config),
        'metrics': result['metrics']
    }, model_path)
    print(f"Model saved to {model_path}")

    print("=== Validation ===")
    input_eval = sample_bulk(training_config, resample=False, num_samp=128).to(device)
    input_eval.requires_grad_(True)
    t = input_eval[:, 0:1]
    x = input_eval[:, 1:2]
    u = model(input_eval)
    results = kdv(u, input_eval)

    plot_field_visualization(results, training_config, 'deriv', f'{output_dir}/trained_deriv.png', 'Trained Model')
    plot_field_visualization(results, training_config, 'res', f'{output_dir}/trained_res.png', 'Trained Model')
    plot_field_visualization(results, training_config, 'iom', f'{output_dir}/trained_iom.png', 'Trained Model')

    delta = {k: (results[k] - results_ana[k]) for k in results}
    plot_field_visualization(delta, training_config, 'deriv', f'{output_dir}/error_deriv.png', 'Error (Trained - Analytic)')
    plot_field_visualization(delta, training_config, 'res', f'{output_dir}/error_res.png', 'Error (Trained - Analytic)')

    print("=== Inverse Scattering ===")
    u_reshape = u.reshape(training_config.num_samp_eval, training_config.num_samp_eval).detach().cpu().numpy()
    tvals = t.reshape(training_config.num_samp_eval, training_config.num_samp_eval).detach().cpu().numpy()[:, 0]
    x_d = x.detach().reshape(training_config.num_samp_eval, training_config.num_samp_eval)[0, :].cpu().numpy()
    dx = x_d[1] - x_d[0]

    S = SchrodingerSolver()
    eigenvector_stack, eigenvalue_stack = S.solve(u_reshape, dx)
    kappa_rec = np.sqrt(-eigenvalue_stack[:, :sd.Ns] + 1e-8)

    plot_scattering_validation(sd, kappa_rec, eigenvector_stack, tvals, output_dir, squared=True)
    plot_scattering_validation(sd, kappa_rec, eigenvector_stack, tvals, output_dir, squared=False)

    metrics = compute_metrics(sd, kappa_rec, results, results_ana)
    print("\n=== Validation Metrics ===")
    for key, val in metrics.items():
        print(f"{key}: {val:.3}")

    metrics_path = f'{output_dir}/metrics.txt'
    with open(metrics_path, 'w') as f:
        for key, val in metrics.items():
            f.write(f"{key}: {val:.6e}\n")
    print(f"\nMetrics saved to {metrics_path}")

    return metrics

if __name__ == "__main__":
    validate_run()
    
