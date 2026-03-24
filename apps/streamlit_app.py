# Add parent directory to path for imports
import sys
sys.path.insert(0, '..')

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scattering import ScatteringData, SchrodingerSolver
from configuration import kdv_config
from sampling import sample_bulk

st.set_page_config(page_title="KdV Soliton Inspector", layout="wide")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

PRESETS = {
    "2-Soliton": {
        "kappas": [1.2, 0.9],
        "x0s": [8, 2],
        "L": 30,
        "T": 3,
    },
    "3-Soliton": {
        "kappas": [1.6, 1.2, 1.1],
        "x0s": [8, 4, 2],
        "L": 30,
        "T": 3,
    },
    "5-Soliton": {
        "kappas": [1.5, 0.97, 1.2, 1.3, 2],
        "x0s": [-5, 8, 10, 12, -4],
        "L": 30,
        "T": 3,
    },
    "8-Soliton Cascade": {
        "kappas": [1.5 - 0.025 * i for i in range(8)],
        "x0s": [20 for _ in range(8)],
        "L": 30,
        "T": 3,
    },
    "10-Soliton Train": {
        "kappas": [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1],
        "x0s": [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0],
        "L": 40,
        "T": 5,
    },
}

def initialize_session_state():
    if 'kappas' not in st.session_state:
        st.session_state.kappas = [1.2, 0.9]
    if 'x0s' not in st.session_state:
        st.session_state.x0s = [8, 2]
    if 'L' not in st.session_state:
        st.session_state.L = 30
    if 'T' not in st.session_state:
        st.session_state.T = 3
    if 'num_samp_eval' not in st.session_state:
        st.session_state.num_samp_eval = 256
    if 'use_tau' not in st.session_state:
        st.session_state.use_tau = True
    if 'vmax' not in st.session_state:
        st.session_state.vmax = 5.0
    if 'config_changed' not in st.session_state:
        st.session_state.config_changed = True
    if 'cached_results' not in st.session_state:
        st.session_state.cached_results = None

initialize_session_state()

st.title("KdV Soliton Inspector")

with st.sidebar:
    st.header("Scattering Data Configuration")

    def load_preset():
        preset_name = st.session_state.preset_selector
        if preset_name != "Custom":
            preset_data = PRESETS[preset_name]
            st.session_state.kappas = preset_data["kappas"].copy()
            st.session_state.x0s = preset_data["x0s"].copy()
            st.session_state.L = preset_data["L"]
            st.session_state.T = preset_data["T"]
            st.session_state.config_changed = True

    preset = st.selectbox(
        "Load Preset",
        ["Custom"] + list(PRESETS.keys()),
        index=0,
        key="preset_selector",
        on_change=load_preset
    )

    st.subheader("Visualization Options")
    if 'psi_squared' not in st.session_state:
        st.session_state.psi_squared = False
    st.session_state.psi_squared = st.checkbox("Show $|\\psi|^2$", value=st.session_state.psi_squared)

    st.subheader("Solitons")

    if 'to_remove' not in st.session_state:
        st.session_state.to_remove = None

    if st.session_state.to_remove is not None:
        idx = st.session_state.to_remove
        st.session_state.kappas.pop(idx)
        st.session_state.x0s.pop(idx)
        st.session_state.to_remove = None
        st.rerun()

    new_kappas = []
    new_x0s = []
    for i in range(len(st.session_state.kappas)):
        col1, col2, col3 = st.columns([3, 3, 1])
        with col1:
            kappa_val = st.number_input(
                "$\\kappa$",
                value=float(st.session_state.kappas[i]),
                min_value=0.01,
                max_value=5.0,
                step=0.01,
                key=f"kappa_{i}",
                label_visibility="collapsed" if i > 0 else "visible"
            )
            new_kappas.append(kappa_val)
        with col2:
            x0_val = st.number_input(
                "$x_0$",
                value=float(st.session_state.x0s[i]),
                min_value=-50.0,
                max_value=50.0,
                step=0.1,
                key=f"x0_{i}",
                label_visibility="collapsed" if i > 0 else "visible"
            )
            new_x0s.append(x0_val)
        with col3:
            st.write("" if i > 0 else " ")
            if st.button("🗑️", key=f"remove_{i}"):
                st.session_state.to_remove = i
                st.rerun()

    if new_kappas != st.session_state.kappas or new_x0s != st.session_state.x0s:
        st.session_state.config_changed = True

    st.session_state.kappas = new_kappas
    st.session_state.x0s = new_x0s

    st.subheader("Add Soliton")
    col1, col2 = st.columns(2)
    with col1:
        new_kappa = st.number_input("New $\\kappa$", value=1.0, min_value=0.01, max_value=5.0, step=0.01, key="new_kappa")
    with col2:
        new_x0 = st.number_input("New $x_0$", value=0.0, min_value=-50.0, max_value=50.0, step=0.1, key="new_x0")

    if st.button("Add Soliton"):
        st.session_state.kappas.append(new_kappa)
        st.session_state.x0s.append(new_x0)
        st.rerun()

    st.subheader("Domain Parameters")

    new_L = st.number_input("$L$ (spatial domain)", value=float(st.session_state.L), min_value=1.0, max_value=100.0, key="L_input")
    new_T = st.number_input("$T$ (time domain)", value=float(st.session_state.T), min_value=0.1, max_value=10.0, key="T_input")
    new_num_samp = st.number_input("Grid points", value=int(st.session_state.num_samp_eval), min_value=64, max_value=512, step=64, key="samp_input")
    new_vmax = st.number_input("Color scale max", value=float(st.session_state.vmax), min_value=1.0, max_value=20.0, key="vmax_input")

    if new_L != st.session_state.L or new_T != st.session_state.T or new_num_samp != st.session_state.num_samp_eval:
        st.session_state.config_changed = True
    st.session_state.L = new_L
    st.session_state.T = new_T
    st.session_state.num_samp_eval = new_num_samp
    st.session_state.vmax = new_vmax

    st.subheader("Computation Options")
    new_use_tau = st.checkbox("Use tau method", value=st.session_state.use_tau, key="tau_checkbox")
    if new_use_tau != st.session_state.use_tau:
        st.session_state.config_changed = True
    st.session_state.use_tau = new_use_tau

if len(st.session_state.kappas) == 0:
    st.warning("Please add at least one soliton")
    st.stop()

try:
    sd = ScatteringData(
        st.session_state.kappas,
        st.session_state.x0s,
        R=None,
        use_tau=st.session_state.use_tau
    )
except AssertionError as e:
    st.error(f"Invalid scattering data: {e}")
    st.stop()

config = kdv_config
config.L = st.session_state.L
config.T = st.session_state.T
config.num_samp_eval = st.session_state.num_samp_eval
config.vmax = st.session_state.vmax

if st.session_state.config_changed or st.session_state.cached_results is None:
    with st.spinner("Computing solution..."):
        input_eval = sample_bulk(config, resample=False, num_samp=config.num_samp_eval)
        input_eval.requires_grad_(True)
        t = input_eval[:, 0:1]
        x = input_eval[:, 1:2]
        u_analytic = sd.u(x, t)
        u = u_analytic.reshape(config.num_samp_eval, config.num_samp_eval).detach().cpu().numpy()
        tvals = t.reshape(config.num_samp_eval, config.num_samp_eval).detach().cpu().numpy()[:, 0]
        x_d = x.detach().reshape(config.num_samp_eval, config.num_samp_eval)[0, :].cpu().numpy()
        dx = x_d[1] - x_d[0]

        S = SchrodingerSolver()
        eigenvector_stack, eigenvalue_stack = S.solve(u, dx)

        st.session_state.cached_results = {
            'u': u,
            'tvals': tvals,
            'x_d': x_d,
            'dx': dx,
            'input_eval': input_eval,
            'eigenvector_stack': eigenvector_stack,
            'eigenvalue_stack': eigenvalue_stack,
            'S': S
        }
        st.session_state.config_changed = False
else:
    u = st.session_state.cached_results['u']
    tvals = st.session_state.cached_results['tvals']
    x_d = st.session_state.cached_results['x_d']
    dx = st.session_state.cached_results['dx']
    input_eval = st.session_state.cached_results['input_eval']
    eigenvector_stack = st.session_state.cached_results['eigenvector_stack']
    eigenvalue_stack = st.session_state.cached_results['eigenvalue_stack']
    S = st.session_state.cached_results['S']

st.header("Potential $u(x,t)$")
fig1, ax1 = plt.subplots(figsize=(10, 6))
L = config.L
T = config.T
im1 = ax1.imshow(-u, extent=[-L, L, 0, T],
                 vmin=-config.vmax, vmax=0,
                 origin='lower', cmap='coolwarm', aspect='auto')
ax1.set_xlabel('$x$', fontsize=12)
ax1.set_ylabel('$t$', fontsize=12)
ax1.set_title('Potential $u(x,t)$', fontsize=14)
plt.colorbar(im1, ax=ax1)
st.pyplot(fig1)
plt.close()

st.header("Eigenvalue Analysis")

n_kappas = len(st.session_state.kappas)
kappas_sorted = sorted(st.session_state.kappas, reverse=True)

n_evs_to_show = min(n_kappas, eigenvector_stack.shape[2])

ncols = min(4, n_evs_to_show)
nrows = (n_evs_to_show + ncols - 1) // ncols

fig2, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
if nrows == 1 and ncols == 1:
    axes = np.array([axes])
axes = axes.flatten()

for i in range(n_evs_to_show):
    eigenvector_timeseries = eigenvector_stack[:, :, i]
    if st.session_state.psi_squared:
        eigenvector_timeseries = eigenvector_timeseries**2

    im = axes[i].imshow(eigenvector_timeseries, aspect='auto', origin='lower',
                        extent=[x_d.min(), x_d.max(), 0, T],
                        cmap='RdBu_r')
    axes[i].set_xlabel('$x$', fontsize=10)
    axes[i].set_ylabel('$t$', fontsize=10)

    evs_t0 = eigenvalue_stack[0]
    lambda_i = evs_t0[i]
    kappa_i = kappas_sorted[i]
    expected = -kappa_i**2

    if st.session_state.psi_squared:
        axes[i].set_title(f'$|\\psi_{{{i}}}|^2$\n$\\lambda = {lambda_i:.3f}$, $-\\kappa^2 = {expected:.3f}$', fontsize=11)
    else:
        axes[i].set_title(f'$\\psi_{{{i}}}$\n$\\lambda = {lambda_i:.3f}$, $-\\kappa^2 = {expected:.3f}$', fontsize=11)
    plt.colorbar(im, ax=axes[i])

for i in range(n_evs_to_show, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.header("Animation")

n_evs = min(n_kappas, 8)
x_interior = x_d[1:-1]

fig_anim, ax_anim = plt.subplots(figsize=(12, 6))

frame_skip = max(1, len(tvals) // 100)
frames_to_use = list(range(0, len(tvals), frame_skip))

colors = plt.cm.viridis(np.linspace(0, 0.9, n_evs))

u_max = np.abs(u).max()
u_min = np.min(u)
ev_max = np.max([np.abs(eigenvector_stack[:, :, i]).max() for i in range(n_evs)])

evs_t0 = eigenvalue_stack[0]

line_u, = ax_anim.plot([], [], 'k-', linewidth=2, label='$u(x,t)$')
lines_evs = []
for i in range(n_evs):
    if st.session_state.psi_squared:
        label = f'$|\\psi_{{{i}}}|^2$ ($\\lambda$={evs_t0[i]:.2f})'
    else:
        label = f'$\\psi_{{{i}}}$ ($\\lambda$={evs_t0[i]:.2f})'
    line, = ax_anim.plot([], [], linewidth=1.5, alpha=0.7,
                         color=colors[i], label=label)
    lines_evs.append(line)

ax_anim.set_xlim(x_d.min(), x_d.max())

global_max = max(u_max, ev_max * (u_max / ev_max) * 0.5)
ax_anim.set_ylim(-global_max * 1.2, global_max * 1.2)

ax_anim.set_xlabel('$x$', fontsize=12)
if st.session_state.psi_squared:
    ax_anim.set_ylabel('$|\\psi|^2$', fontsize=12)
else:
    ax_anim.set_ylabel('$\\psi$', fontsize=12)
ax_anim.legend(loc='upper right', fontsize=9)
ax_anim.grid(True, alpha=0.3)

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
        if st.session_state.psi_squared:
            psi = psi**2
        psi_scaled = psi * (u_max / ev_max) * 0.5
        lines_evs[i].set_data(x_interior, psi_scaled)

    time_text.set_text(f't = {tvals[frame]:.3f}')

    return [line_u] + lines_evs + [time_text]

anim = FuncAnimation(fig_anim, animate, init_func=init,
                    frames=frames_to_use, interval=50, blit=True, repeat=True)

st.components.v1.html(anim.to_jshtml(), height=700, scrolling=True)

st.header("Validation Results")

with st.spinner("Running validation checks..."):
    results = S.check_SD(sd, input_eval, verbose=False)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("MSE (Eigenvalues)", f"{results['mse_eigenvalues']:.2e}")
with col2:
    st.metric("Mean Time Variance", f"{results['mean_time_variance']:.2e}")
with col3:
    st.metric("MSE (Time Evolution)", f"{results['mse_psi_time']:.2e}")

eigenvalue_pass = results['mse_eigenvalues'] < 1e-3 and results['mean_time_variance'] < 1e-6
time_pass = results['mse_psi_time'] < 1e-2

evs_t0 = eigenvalue_stack[0]
max_ev_error = max([abs(evs_t0[i] - results['expected_eigenvalues'][i]) for i in range(n_kappas)])
max_time_var = max([np.var(eigenvalue_stack[:, i]) for i in range(n_kappas)])

if eigenvalue_pass:
    st.success(f"✓ Eigenvalues are isospectral (max error: {max_ev_error:.2e}, max time variance: {max_time_var:.2e})")
else:
    st.error(f"✗ Eigenvalues fail isospectrality check (max error: {max_ev_error:.2e}, max time variance: {max_time_var:.2e})")
    st.caption(f"Threshold: error < 1e-3 and time variance < 1e-6")

if time_pass:
    st.success(f"✓ Eigenvectors have correct time dependence (MSE: {results['mse_psi_time']:.2e})")
else:
    st.error(f"✗ Eigenvectors time evolution is incorrect (MSE: {results['mse_psi_time']:.2e})")
    st.caption(f"Threshold: MSE < 1e-2")

with st.expander("Expected vs Computed Eigenvalues"):
    st.markdown("**Eigenvalue Comparison Table**")
    ev_data = {
        "n": list(range(n_kappas)),
        "$\\kappa$": [f"{k:.3f}" for k in kappas_sorted],
        "Expected $\\lambda$": [f"{ev:.4f}" for ev in results['expected_eigenvalues'][:n_kappas]],
        "Computed $\\lambda$ (t=0)": [f"{evs_t0[n]:.4f}" for n in range(n_kappas)],
        "Error": [f"{abs(evs_t0[n] - results['expected_eigenvalues'][n]):.2e}" for n in range(n_kappas)]
    }
    st.table(ev_data)

# Kappa Recovery Plot
st.header("Recovered Wave Numbers ($\\kappa$)")
st.markdown("Verifying that wave numbers $\\kappa_n = \\sqrt{-\\lambda_n}$ remain constant over time (isospectrality).")

# Compute recovered kappa values: κ = sqrt(-λ)
kappa_rec = np.sqrt(-eigenvalue_stack[:, :n_kappas] + 1e-8)

fig_kappa, ax_kappa = plt.subplots(figsize=(10, 5))
for idx in range(n_kappas):
    line, = ax_kappa.plot(tvals[:-1], kappa_rec[:, idx],
                           label=f'$\\kappa_{{{idx}}}$ (recovered)',
                           linewidth=2, alpha=0.8)
    # Plot expected value as horizontal line
    ax_kappa.axhline(kappas_sorted[idx],
                     color=line.get_color(),
                     linestyle='--',
                     linewidth=1.5,
                     alpha=0.7,
                     label=f'$\\kappa_{{{idx}}}$ (expected: {kappas_sorted[idx]:.3f})')

ax_kappa.set_xlabel('Time $t$', fontsize=12)
ax_kappa.set_ylabel('Wave Number $\\kappa$', fontsize=12)
ax_kappa.set_title('Recovered Wave Numbers from Eigenvalues', fontsize=14)
ax_kappa.legend(loc='best', fontsize=10)
ax_kappa.grid(True, alpha=0.3)

# Set reasonable y-limits
kappa_min = min(kappas_sorted) * 0.8
kappa_max = max(kappas_sorted) * 1.2
ax_kappa.set_ylim(kappa_min, kappa_max)

plt.tight_layout()
st.pyplot(fig_kappa)
plt.close()

# Show statistics
st.subheader("Kappa Recovery Statistics")
cols = st.columns(n_kappas)
for idx in range(n_kappas):
    with cols[idx]:
        kappa_mean = kappa_rec[:, idx].mean()
        kappa_std = kappa_rec[:, idx].std()
        kappa_error = abs(kappa_mean - kappas_sorted[idx])

        st.metric(
            label=f"$\\kappa_{{{idx}}}$",
            value=f"{kappa_mean:.4f}",
            delta=f"±{kappa_std:.2e} std"
        )
        st.caption(f"Expected: {kappas_sorted[idx]:.4f}")
        st.caption(f"Error: {kappa_error:.2e}")

st.sidebar.markdown("---")
st.sidebar.info(f"Using device: {device}")
st.sidebar.info(f"Number of solitons: {len(st.session_state.kappas)}")
