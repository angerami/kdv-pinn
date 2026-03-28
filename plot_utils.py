"""Plotting utilities for visualizing PINN training and results."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from physics import kdv


def _init_interactive_plot():
    """Initialize IPython interactive plotting (for Jupyter notebooks)."""
    try:
        from IPython.display import clear_output, display
        plt.ion()
        return clear_output, display
    except ImportError:
        return None, None


def _update_interactive_plot(model, input_eval, config, metrics):
    """Update interactive plot during training (for Jupyter notebooks)."""
    from IPython.display import clear_output, display
    model.eval()
    with torch.no_grad():
        clear_output(wait=True)
        fig = plot_results(model, input_eval, config, metrics)
        display(fig)
        plt.close(fig)
    model.train()


def plot_results(model, input_eval, config, metrics, filename=None):
    """Create comprehensive training summary plot.

    Shows loss curves, field statistics, conserved quantities, and spacetime visualization.

    Args:
        model: Trained neural network
        input_eval: Evaluation grid
        config: Configuration object
        metrics: Dictionary of training metrics
        filename: Optional path to save figure

    Returns:
        Matplotlib figure object
    """
    num_points = int(np.sqrt(input_eval.shape[0]))
    fig = plt.figure(figsize=(12, 10))

    # Loss plot
    ax_loss = fig.add_subplot(321)
    ax_loss.semilogy(metrics['L_total'], label='$L_{total}$')
    if 'L_KDV' in metrics and len(metrics['L_KDV']) > 0:
        ax_loss.semilogy(metrics['L_KDV'], label='$L_{KDV}$')
    if 'L_IC' in metrics and len(metrics['L_IC']) > 0:
        ax_loss.semilogy(metrics['L_IC'], label='$L_{IC}$')
    if 'L_BC' in metrics and len(metrics['L_BC']) > 0:
        ax_loss.semilogy(metrics['L_BC'], label='$L_{BC}$')
    if 'L_S' in metrics and len(metrics['L_S']) > 0:
        ax_loss.semilogy(metrics['L_S'], label='$L_{S}$', linestyle='--')
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

    # Spacetime visualization
    u = model(input_eval).detach().cpu()
    u_reshaped = u.reshape(num_points, num_points).numpy()

    ax_field = fig.add_subplot(324)
    T = getattr(config, 'T', 1.0)
    L = config.L
    im1 = ax_field.imshow(u_reshaped, extent=[-L, L, 0, T],
                          vmin=config.vmin, vmax=config.vmax,
                          origin='lower', cmap='coolwarm', aspect='auto')
    ax_field.set_title('u(t, x) - Spacetime')
    ax_field.set_xlabel('$x$')
    ax_field.set_ylabel('$t$')
    plt.colorbar(im1, ax=ax_field)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    return fig


def plot_results_2panel(model, input_eval, config, metrics, filename=None):
    """Create 2-panel summary plot for writeup.

    Shows training loss (left) and spacetime field u(x,t) (right).

    Args:
        model: Trained neural network
        input_eval: Evaluation grid
        config: Configuration object
        metrics: Dictionary of training metrics
        filename: Optional path to save figure

    Returns:
        Matplotlib figure object
    """
    num_points = int(np.sqrt(input_eval.shape[0]))
    fig, (ax_loss, ax_field) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot (left panel)
    ax_loss.semilogy(metrics['L_total'], label='$L_{total}$')
    if 'L_KDV' in metrics and len(metrics['L_KDV']) > 0:
        ax_loss.semilogy(metrics['L_KDV'], label='$L_{KDV}$')
    if 'L_IC' in metrics and len(metrics['L_IC']) > 0:
        ax_loss.semilogy(metrics['L_IC'], label='$L_{IC}$')
    if 'L_BC' in metrics and len(metrics['L_BC']) > 0:
        ax_loss.semilogy(metrics['L_BC'], label='$L_{BC}$')
    if 'L_S' in metrics and len(metrics['L_S']) > 0:
        ax_loss.semilogy(metrics['L_S'], label='$L_{S}$', linestyle='--')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.set_title('Training Loss')
    ax_loss.grid(True)

    # Spacetime visualization (right panel)
    u = model(input_eval).detach().cpu()
    u_reshaped = u.reshape(num_points, num_points).numpy()

    T = getattr(config, 'T', 1.0)
    L = config.L
    im = ax_field.imshow(u_reshaped, extent=[-L, L, 0, T],
                         vmin=config.vmin, vmax=config.vmax,
                         origin='lower', cmap='coolwarm', aspect='auto')
    ax_field.set_title('$u(x, t)$ - Spacetime')
    ax_field.set_xlabel('$x$')
    ax_field.set_ylabel('$t$')
    plt.colorbar(im, ax=ax_field)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved 2-panel plot to {filename}")
    return fig


def plot_2D_field(ax_field, u_reshaped, config, title='u(x,t)', show_colorbar=True):
    """Plot 2D spacetime field on given axes."""
    T = getattr(config, 'T', 1.0)
    L = config.L
    aspect_ratio = (2 * L) / T
    im1 = ax_field.imshow(u_reshaped, extent=[-L, L, 0, T],
                          origin='lower', cmap='coolwarm', aspect=aspect_ratio)
    ax_field.set_title(title)
    ax_field.set_xlabel('$x$')
    ax_field.set_ylabel('$t$')
    if show_colorbar:
        plt.colorbar(im1, ax=ax_field)
    return im1


def plot_field_visualization(results, config, view='res', filename=None, suptitle=None):
    """Create multi-panel visualization of field quantities.

    Args:
        results: Dictionary from kdv() function
        config: Configuration object
        view: One of 'res' (residuals), 'deriv' (derivatives),
              'iom' (integrals of motion), 'curr' (currents)
        filename: Optional path to save figure
        suptitle: Optional figure title

    Returns:
        None (displays or saves figure)
    """
    if view == 'res':
        field_quantities = ['res_KDV', 'res_H0', 'res_H1','res_H2']
    elif view == 'deriv':
        field_quantities = ['u', 'u_t', 'u_x', 'u_xx', 'u_xxx']
    elif view == 'iom':
        field_quantities = ['u', 'rho_1', 'rho_2', 'rho_3']
    elif view == 'curr':
        field_quantities = ['u', 'J_0', 'rho_1', 'J_1','J_2']
    else:
        raise ValueError(f"Unknown view type: {view}")

    # Determine layout based on number of plots
    n_plots = len(field_quantities)
    if n_plots == 3:
        num_rows, num_cols = 1, 3
        figsize = (15, 4)
    elif n_plots == 4:
        num_rows, num_cols = 2, 2
        figsize = (10, 8)
    elif n_plots in [5, 6]:
        num_rows, num_cols = 2, 3
        figsize = (15, 8)
    else:
        num_rows, num_cols = 2, 3
        figsize = (15, 8)

    # Get grid size from first field
    first_field = results[field_quantities[0]]
    num_points = int(np.sqrt(first_field.shape[0]))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    eqs, descs = load_equations(config.equation_file)
    for i, u_name in enumerate(field_quantities):
        u = results[u_name].detach().cpu()
        u_reshaped = u.reshape(num_points, num_points).numpy()
        plot_2D_field(axes[i], u_reshaped, config, title=eqs[u_name], show_colorbar=True)

    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.0)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filename}")

def load_equations(md_path):
    """Load equation labels from markdown file for plot titles.

    Parses markdown comments to extract LaTeX equation labels.

    Args:
        md_path: Path to markdown file with equation annotations.
                Can be absolute, or relative to either:
                - Current working directory
                - The kdv-pinn package directory (fallback)

    Returns:
        eqs: Dictionary mapping variable names to LaTeX labels
        descs: Dictionary mapping variable names to descriptions
    """
    import re
    import os

    # Try the path as-is first (absolute or relative to cwd)
    if os.path.exists(md_path):
        full_path = md_path
    else:
        # Fallback: look relative to this module's directory (kdv-pinn root)
        module_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(module_dir, md_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Cannot find equation file '{md_path}'. "
                f"Tried: current directory and {module_dir}"
            )

    with open(full_path) as f:
        text = f.read()

    eqs = {}
    for m in re.finditer(r'<!--\s*eq:(\w+)\s*-->.*?\n\$([^$]+)\$', text):
        eqs[m.group(1)] = f'${m.group(2)}$'

    descs = {}
    for m in re.finditer(r'<!--\s*desc:(\w+)\s*-->\s*(.+)', text):
        descs[m.group(1)] = m.group(2).strip()

    return eqs, descs


def plot_scattering_validation(sd, kappa_rec, eigenvector_stack, eigenvalue_stack, tvals, dx, output_dir, squared=True):
    """Plot eigenfunction evolution and recovered wave numbers.

    Args:
        sd: ScatteringData object
        kappa_rec: Recovered kappa values over time (sorted largest to smallest)
        eigenvector_stack: Time series of eigenfunctions
        eigenvalue_stack: Time series of eigenvalues
        tvals: Time values
        dx: Spatial grid spacing
        output_dir: Directory for saving plots
        squared: If True, plot |ψ|², else plot ψ

    Note:
        kappa_rec comes from eigenvalue solver which always returns eigenvalues sorted.
        Since λ = -κ², the most negative eigenvalue corresponds to the largest κ.
        We sort sd.kappas to match this ordering for comparison.
    """
    vmin, vmax = (0, 0.1) if squared else (-0.3, 0.3)
    label_suffix = '^2' if squared else ''

    # Sort kappas to match the eigenvalue solver ordering (largest first)
    kappas_sorted = sorted(sd.kappas, reverse=True)

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
    # Compare recovered kappas (sorted) to ground truth kappas (sorted)
    _, ax = plt.subplots(figsize=(8, 4))
    for idx in range(sd.Ns):
        kappa_true = kappas_sorted[idx]
        line, = ax.plot(tvals[1:], kappa_rec[:, idx], label=f'$\\kappa_{{{idx}}}$ (recovered, κ={kappa_true:.2f})')
        ax.axhline(kappa_true, color=line.get_color(), linestyle='--', alpha=0.7)
    ax.set_ylim(0, max(kappas_sorted) * 1.3)
    ax.set_ylabel('$\\kappa$')
    ax.set_xlabel('$t$')
    ax.legend()
    ax.set_title('Recovered Wave Numbers (sorted by magnitude)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kappa_recovery.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_animation(u, eigenvector_stack, eigenvalue_stack, tvals, x_d, sd, output_path,
                       squared=False, fps=20, max_frames=100):
    """Generate animation of potential and eigenfunctions over time.

    Args:
        u: Potential field u(x,t) as 2D array (time, space)
        eigenvector_stack: Time series of eigenfunctions
        eigenvalue_stack: Time series of eigenvalues
        tvals: Time values
        x_d: Spatial grid points
        sd: ScatteringData object
        output_path: Path to save animation (e.g., 'output/animation.gif')
        squared: If True, plot |ψ|², else plot ψ
        fps: Frames per second for the animation
        max_frames: Maximum number of frames to include (skips frames if needed)
    """
    print(f"Generating animation: {output_path}")

    n_evs = min(sd.Ns, 8, eigenvector_stack.shape[2])
    x_interior = x_d[1:-1]

    # Setup figure
    fig_anim, ax_anim = plt.subplots(figsize=(12, 6))
    fig_anim.tight_layout(pad=0.5)

    # Determine frames to use (skip frames if too many)
    max_frame = min(len(tvals), eigenvector_stack.shape[0])
    frame_skip = max(1, max_frame // max_frames)
    frames_to_use = list(range(0, max_frame, frame_skip))

    # Color scheme for eigenfunctions
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_evs))

    # Determine plot limits
    u_max = np.abs(u).max()
    u_min = np.min(u)
    ev_max = np.max([np.abs(eigenvector_stack[:, :, i]).max() for i in range(n_evs)])

    evs_t0 = eigenvalue_stack[0]

    # Initialize lines
    line_u, = ax_anim.plot([], [], 'k-', linewidth=2, label='$u(x,t)$')
    lines_evs = []
    for i in range(n_evs):
        if squared:
            label = f'$|\\psi_{{{i}}}|^2$ ($\\lambda$={evs_t0[i]:.2f})'
        else:
            label = f'$\\psi_{{{i}}}$ ($\\lambda$={evs_t0[i]:.2f})'
        line, = ax_anim.plot([], [], linewidth=1.5, alpha=0.7,
                            color=colors[i], label=label)
        lines_evs.append(line)

    # Set axis limits
    ax_anim.set_xlim(x_d.min(), x_d.max())
    global_max = max(u_min, ev_max * (u_max / ev_max) * 0.5)
    ax_anim.set_ylim(-1.2 * u_max, 1.2 * global_max)
                    #  global_max * 1.2, global_max * 1.2)

    # Labels and styling
    ax_anim.set_xlabel('$x$', fontsize=12)
    if squared:
        ax_anim.set_ylabel('$|\\psi|^2$', fontsize=12)
    else:
        ax_anim.set_ylabel('$\\psi$', fontsize=12)
    ax_anim.legend(loc='upper right', fontsize=9)
    ax_anim.grid(True, alpha=0.3)

    # Time text overlay
    time_text = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes,
                            fontsize=14, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def init():
        line_u.set_data([], [])
        for line in lines_evs:
            line.set_data([], [])
        time_text.set_text('')
        return [line_u] + lines_evs + [time_text]

    def animate(frame):
        u_slice = -u[frame, :] - u_min
        line_u.set_data(x_d, u_slice)

        for i in range(n_evs):
            psi = eigenvector_stack[frame, :, i]
            if squared:
                psi = psi**2
            psi_scaled = psi * (u_max / ev_max) * 0.5
            lines_evs[i].set_data(x_interior, psi_scaled)

        time_text.set_text(f't = {tvals[frame]:.3f}')

        return [line_u] + lines_evs + [time_text]

    # Create animation
    anim = FuncAnimation(fig_anim, animate, init_func=init,
                        frames=frames_to_use, interval=1000/fps, blit=True, repeat=True)

    # Save animation
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig_anim)

    print(f"Animation saved to {output_path} ({len(frames_to_use)} frames @ {fps} fps)")