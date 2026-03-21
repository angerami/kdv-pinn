import torch
import numpy as np
import matplotlib.pyplot as plt
from scattering import ScatteringData
from configuration import kdv_config
from sampling import sample_bulk

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using {device} device.")

psi_squared = False
config = kdv_config
config.L = 30
config.T = 3
config.num_samp_eval = 256
L = config.L
T = config.T
# config.kappas = [1.5, 0.97, 1.2, 1.3, 2]
# config.x0s = [-5, 8, 10, 12, -4]
config.kappas = [1.5 - 0.025 * i for i in range(8)]
config.x0s = [20 for _ in config.kappas]
# config.kappas = [1.6, 1.2, 1.1]
# config.x0s = [8, 4, 2]
#config.kappas = [1.2, 0.9]
#config.x0s = [8,2]
kappas_sorted = sorted(config.kappas)[::-1]
sd = ScatteringData(config.kappas, config.x0s, R=None, use_tau=True)


input_eval = sample_bulk(config, resample=False, num_samp=config.num_samp_eval)
input_eval.requires_grad_(True)
t = input_eval[:, 0:1]
x = input_eval[:, 1:2]
u_analytic = sd.u(x, t)
u = u_analytic.reshape(config.num_samp_eval,config.num_samp_eval).detach().cpu().numpy()

tvals =t.reshape(config.num_samp_eval,config.num_samp_eval).detach().cpu().numpy()[:,0]

def solve_schrodinger(u_vals, xv):
    dx = xv[1] - xv[0]
    u_int = u_vals[1:-1]  # interior points
    
    diag = 2/dx**2 - u_int # main diagonal of H
    off = -np.ones(len(u_int)-1) / dx**2  # sub/super diagonal
    
    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    return eigenvalues, eigenvectors

x_d = x.detach()
ev_vs_time = []
eigenvector_stack = []  # Stack eigenvectors across time
taxis = []
# for t_idx in range(0,config.num_samp_eval,config.num_samp_eval // 8):
for t_idx, tt in enumerate(tvals):
    evs, psis = solve_schrodinger(u[t_idx], x_d)
    bound = evs[evs < 0]
    if len(bound) != 0:
        taxis.append(tt)
        ev_vs_time.append(bound[0].item())

    # Store all eigenvectors for this time slice
    if psi_squared:
        psis = psis**2
    eigenvector_stack.append(psis)

# Convert to numpy array: shape (num_time_steps, num_spatial_points, num_eigenvectors)
eigenvector_stack = np.array(eigenvector_stack)

# Get eigenvalues at t=0 for comparison (they should be constant in time)
evs_t0, _ = solve_schrodinger(u[0], x_d)

# Create multipanel figure
n_kappas = len(config.kappas)
# Total panels needed (potential + eigenvectors)
n_panels = n_kappas + 1
# 4 columns per row if more than 3 kappas, otherwise fit to number of panels
ncolumns = 4 if n_kappas > 3 else n_panels
nrows = (n_panels + ncolumns - 1) // ncolumns  # Ceiling division
# Each panel should be square on the canvas
panel_size = 3  # inches per panel (square)
fig, axes = plt.subplots(nrows, ncolumns, figsize=(ncolumns * panel_size, nrows * panel_size))
axes = axes.flatten()
# First panel: potential u(x,t)
m1 = axes[0].imshow(-u, extent=[-L, L, 0, T],
                    vmin=-config.vmax, vmax=-config.vmin,
                    origin='lower', cmap='coolwarm', aspect='auto')
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$t$')
axes[0].set_title('Potential $u(x,t)$')
fig.colorbar(m1, ax=axes[0])
# Other panels: eigenvectors
for i in range(n_kappas):
    eigenvector_timeseries = eigenvector_stack[:, :, i]  # shape: (time, space)

    im = axes[i+1].imshow(eigenvector_timeseries, aspect='auto', origin='lower',
                          extent=[x_d.min().item(), x_d.max().item(), 0, T],
                        #   vmin=-0.3, vmax=0.3,
                          cmap='RdBu_r')
    axes[i+1].set_xlabel('x')
    axes[i+1].set_ylabel('t')

    # Get eigenvalue and compare to -kappa^2
    lambda_i = evs_t0[i]
    kappa_i = kappas_sorted[i]
    expected = -kappa_i**2

    axes[i+1].set_title(f'Eigenvector {i}\nλ = {lambda_i:.2f}\n-κ² = {expected:.2f} (κ={kappa_i:.2f})$')
    fig.colorbar(im, ax=axes[i+1])

    print(f"Eigenvector {i}: λ = {lambda_i:.4f}, -κ² = {expected:.2f} (κ={kappa_i:.3f}), diff = {lambda_i - expected:.4f}")

plt.tight_layout()
plt.show()

# ============================================================
# ANIMATION: u(x,t) and eigenvectors evolving together
# ============================================================
from matplotlib.animation import FuncAnimation

# Determine how many eigenvectors to show
n_evs = min(n_kappas, 5)  # Show up to 5 eigenvectors

# Create figure for animation
fig_anim, ax_anim = plt.subplots(figsize=(12, 6))

# Get x values for plotting
x_vals = x_d.reshape(config.num_samp_eval, config.num_samp_eval)[0, :].cpu().numpy()
x_interior = x_vals[1:-1]  # Interior points used in Schrodinger solver

# Initialize plot elements
line_u, = ax_anim.plot([], [], 'k-', linewidth=2, label='$u(x,t)$')
lines_evs = []
colors = plt.cm.viridis(np.linspace(0, 0.9, n_evs))

for i in range(n_evs):
    line, = ax_anim.plot([], [], linewidth=1.5, alpha=0.7,
                         color=colors[i], label=f'$\\psi_{i}$ (λ={evs_t0[i]:.2f})')
    lines_evs.append(line)

ax_anim.set_xlim(x_vals.min(), x_vals.max())
ax_anim.set_xlabel('x', fontsize=12)
if psi_squared:
    ax_anim.set_ylabel('$|\\psi|^2$', fontsize=12)
else:
    ax_anim.set_ylabel('$\\psi$', fontsize=12)
ax_anim.legend(loc='upper right', fontsize=10)
ax_anim.grid(True, alpha=0.3)

# Add a text element for time display
time_text = ax_anim.text(0.02, 0.98, '', transform=ax_anim.transAxes,
                         fontsize=14, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Determine y-limits by looking at all data
u_max = np.abs(u).max()
u_min = np.min(u)
ev_max = np.max([np.abs(eigenvector_stack[:, :, i]).max() for i in range(n_evs)])
ylim = max(u_max, ev_max) * 1.2

def init():
    """Initialize animation"""
    line_u.set_data([], [])
    for line in lines_evs:
        line.set_data([], [])
    time_text.set_text('')
    return [line_u] + lines_evs + [time_text]

def animate(frame):
    """Update animation at frame"""
    # Update u(x,t)
    u_slice = -u[frame, :] - u_min
    line_u.set_data(x_vals, u_slice)

    # Update eigenvectors
    for i in range(n_evs):
        psi = eigenvector_stack[frame, :, i]
        # Scale eigenvectors for better visibility alongside u
        # Normalize them to have similar scale as u
        psi_scaled = psi * (u_max / ev_max) * 0.5
        line.set_data(x_interior, psi_scaled)
        lines_evs[i].set_data(x_interior, psi_scaled)

    # Update time text
    time_text.set_text(f't = {tvals[frame]:.3f}')

    # Dynamically adjust y-limits based on current data
    current_max = max(np.abs(u_slice).max(),
                     max([np.abs(eigenvector_stack[frame, :, i]).max() * (u_max / ev_max) * 0.5
                          for i in range(n_evs)]))
    ax_anim.set_ylim(-current_max * 1.1, current_max * 1.1)

    return [line_u] + lines_evs + [time_text]

# Create animation
# Use fewer frames for faster animation, or all frames for smooth animation
frame_skip = max(1, len(tvals) // 100)  # Use ~100 frames max
frames_to_use = range(0, len(tvals), frame_skip)

anim = FuncAnimation(fig_anim, animate, init_func=init,
                    frames=frames_to_use, interval=50, blit=True, repeat=True)

plt.tight_layout()
print(f"\nCreating animation with {len(frames_to_use)} frames...")
print("Close the animation window to continue or save it.")
plt.show()

# Optionally save the animation (uncomment to save)
# print("Saving animation...")
# anim.save('kdv_eigenvectors_evolution.mp4', writer='ffmpeg', fps=20, dpi=150)
# print("Animation saved as 'kdv_eigenvectors_evolution.mp4'")
