import torch
import numpy as np
import matplotlib.pyplot as plt

def _init_interactive_plot():
    try:
        from IPython.display import clear_output, display
        import matplotlib.pyplot as plt
        plt.ion()
        return clear_output, display
    except ImportError:
        return None, None

def _update_interactive_plot(model, input_eval, config, metrics):
    model.eval()
    with torch.no_grad():
        clear_output(wait=True)
        fig = plot_results(model, input_eval, config, metrics)
        display(fig)
        plt.close(fig)
    model.train()
    
def plot_results(model, input_eval, config, metrics, filename=None):
    num_points = int(np.sqrt(input_eval.shape[0]))
    fig = plt.figure(figsize=(12, 10))

    # Loss plot
    ax_loss = fig.add_subplot(321)
    ax_loss.semilogy(metrics['L_total'], label='$L_{total}$')
    ax_loss.semilogy(metrics['L_B'], label='$L_{B}$')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.set_title('Training Loss')
    ax_loss.grid(True)

    # Field statistics plot
    ax_stats = fig.add_subplot(322)
    epochs = np.arange(len(metrics['mean_u']))
    ax_stats.semilogy(epochs, metrics['mean_u'], label='$|u|_{mean}$')
    ax_stats.semilogy(epochs, metrics['mean_u_t'], label='$|u_{t}|_{mean}$')
    ax_stats.semilogy(epochs, metrics['mean_u_x'], label='$|u_{x}|_{mean}$')
    ax_stats.set_xlabel('Epoch')
    ax_stats.set_ylabel('Field Statistics')
    ax_stats.legend()
    ax_stats.set_title('Field Magnitude')
    ax_stats.grid(True)

    # Integrals of motion plot
    ax_iom = fig.add_subplot(323)
    ax_iom.plot(epochs, metrics['mean_rho_1'], label='Momentum')
    ax_iom.plot(epochs, metrics['mean_rho_2'], label='Energy')
    ax_iom.plot(epochs, metrics['mean_rho_3'], label='H_3/2 (I4)')
    ax_iom.set_xlabel('Epoch')
    ax_iom.set_ylabel('Integrals of Motion')
    ax_iom.legend()
    ax_iom.set_title('Conservation Laws')
    ax_iom.grid(True)

    # Field visualization (spacetime plot)
    u = model(input_eval).detach().cpu()
    u_reshaped = u.reshape(num_points, num_points).numpy()

    ax_field = fig.add_subplot(324)
    T = getattr(config, 'T', 1.0)
    L = config.L
    # extent: [left, right, bottom, top] = [x_min, x_max, t_min, t_max]
    im1 = ax_field.imshow(u_reshaped, extent=[-L, L, 0, T],
                          vmin=config.vmin, vmax=config.vmax,
                          origin='lower', cmap='coolwarm', aspect='auto')
    ax_field.set_title('u(t, x) - Spacetime')
    ax_field.set_xlabel('x')
    ax_field.set_ylabel('t')
    plt.colorbar(im1, ax=ax_field)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    return fig

def plot_2D_field(ax_field, u_reshaped, config, title='u(x,t)', show_colorbar=True):
    T = getattr(config, 'T', 1.0)
    L = config.L
    im1 = ax_field.imshow(u_reshaped, extent=[-L, L, 0, T],
                            vmin=config.vmin, vmax=config.vmax,
                            origin='lower', cmap='coolwarm', aspect='auto')
    ax_field.set_title(title)
    ax_field.set_xlabel('x')
    ax_field.set_ylabel('t')
    if show_colorbar:
        plt.colorbar(im1, ax=ax_field)
    return im1

def plot_field_visualization(model, input_eval, config, view='res', filename=None):
    results = kdv(model(input_eval), input_eval)
    field_quantities = []
    if view == 'res':
        field_quantities = ['res_kvd', 'res_H0', 'res_H1']
    elif view == 'deriv':
        field_quantities = ['u', 'u_t', 'u_x', 'u_xx', 'u_xxx']
    elif view == 'iom':
        field_quantities = ['u', 'rho_1', 'rho_2', 'rho_3']
    elif view == 'curr':
        field_quantities = ['u', 'J_0', 'rho_1', 'J_1']

    num_rows = 2
    num_cols = 2 if len(field_quantities) == 4 else 3

    num_points = int(np.sqrt(input_eval.shape[0]))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))
    axes = axes.flatten()
    for i, u_name in enumerate(field_quantities):
        u = results[u_name].field.detach().cpu()
        u_reshaped= u.reshape(num_points, num_points).numpy()
        plot_2D_field(axes[i], u_reshaped, config, title=u_name, show_colorbar=True)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    return fig