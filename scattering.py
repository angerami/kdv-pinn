"""Inverse scattering transform for KdV N-soliton solutions.

This module implements the inverse scattering transform to construct
multi-soliton solutions from scattering data and solves the associated
Schrödinger eigenvalue problem to verify isospectrality.
"""
import torch
import numpy as np
from itertools import combinations
from physics import gradient


class ScatteringData:
    """Constructs N-soliton solutions from scattering data.

    Uses the inverse scattering transform to build reflectionless
    N-soliton solutions from discrete eigenvalues (kappas) and
    norming constants.

    Args:
        kappas: Wave numbers [κ₁, κ₂, ..., κ_N] (must be distinct and positive)
        x0s: Initial positions [x₀₁, x₀₂, ..., x₀_N]
        R: Reflection coefficient (None for reflectionless case)
        autograd: If True, use autograd for derivatives; else finite differences
        use_tau: If True, use tau-function formulation (more numerically stable)
    """
    def __init__(self, kappas, x0s, R=None, autograd=True, use_tau=False):
        assert len(set(kappas)) == len(kappas), "kappas must be distinct"
        assert all(k > 0 for k in kappas), "kappas must be positive"

        self.kappas = kappas
        self.x0 = x0s
        self.c0 = [np.sqrt(2*k) * np.exp(k*x0) for k, x0 in zip(kappas, x0s)]
        assert all(c > 0 for c in self.c0), "norming constants must be positive"

        self.Ns = len(kappas)
        self.R = R
        self.autograd = autograd
        self.use_tau = use_tau

        # Precompute coefficients for tau-function formulation
        if self.use_tau:
            self.alpha2 = {}
            for i in range(self.Ns):
                for j in range(i+1, self.Ns):
                    ratio = (kappas[i] - kappas[j]) / (kappas[i] + kappas[j])
                    self.alpha2[(i, j)] = ratio**2

            self.subsets = []
            for r in range(self.Ns + 1):
                for S in combinations(range(self.Ns), r):
                    coeff = 1.0
                    for i, j in combinations(S, 2):
                        coeff *= self.alpha2[(i, j)]
                    kappa_sum = sum(self.kappas[i] for i in S)
                    self.subsets.append((S, coeff, kappa_sum))

    def c(self, t):
        """Compute time-dependent norming constants c_n(t) = c_n(0) exp(-4κ_n³t)."""
        t_squeezed = t.squeeze()
        return [c0_n * torch.exp(-4 * k**3 * t_squeezed) for k, c0_n in zip(self.kappas, self.c0)]

    def A(self, x, t):
        """Compute the Gel'fand-Levitan-Marchenko matrix A(x,t).

        The potential is recovered via u = 2∂²_x log det A.

        Returns:
            Tensor of shape (Npts, Ns, Ns)
        """
        c_t = self.c(t)
        Npts = x.shape[0]
        x_squeezed = x.squeeze()

        # Initialize with identity (maintain gradient connection)
        A = torch.zeros(Npts, self.Ns, self.Ns, dtype=x.dtype, device=x.device)
        identity_scale = 1.0 + 0.0 * x.sum()  # Trick to maintain gradients
        for i in range(self.Ns):
            A[:, i, i] = identity_scale

        # Add off-diagonal terms
        for m in range(self.Ns):
            for n in range(self.Ns):
                exp_term = torch.exp((self.kappas[m] + self.kappas[n]) * x_squeezed)
                A[:, m, n] = A[:, m, n] + c_t[m] * c_t[n] * exp_term / (self.kappas[m] + self.kappas[n])

        return A

    def u(self, x, t):
        """Compute the potential u(x,t) = 2∂²_x log det A.

        Returns:
            Tensor of shape (Npts, 1)
        """
        if self.use_tau:
            return self._u_tau(x, t)

        if self.autograd:
            log_det = torch.logdet(self.A(x, t))
            log_det_x = gradient(log_det, x)
            log_det_xx = gradient(log_det_x, x)
            return 2 * log_det_xx
        else:
            # Finite difference approximation
            log_det = lambda xx: torch.logdet(self.A(xx, t))
            dx = 1e-3
            ld_p = log_det(x + dx)
            ld_0 = log_det(x)
            ld_m = log_det(x - dx)
            return 2 * (ld_p - 2*ld_0 + ld_m) / dx**2

    def psi(self, x, t, n, Ainv=None):
        """Compute the n-th bound state eigenfunction ψ_n(x,t).

        Args:
            x: Spatial coordinate
            t: Time coordinate
            n: Index of eigenfunction (0 to Ns-1)
            Ainv: Precomputed inverse of A matrix (optional)

        Returns:
            Tensor of shape (Npts, 1)
        """
        if Ainv is None:
            A = self.A(x, t)
            Ainv = torch.linalg.inv(A)

        c_t = self.c(t)
        x_squeezed = x.squeeze()

        psi_n = torch.zeros_like(x_squeezed)
        for m in range(self.Ns):
            exp_term = torch.exp(-self.kappas[m] * x_squeezed)
            psi_n = psi_n + Ainv[:, n, m] * exp_term

        psi_n = c_t[n] * psi_n
        return psi_n.unsqueeze(-1)

    def eigenvalues(self):
        """Return bound state eigenvalues λ_n = -κ_n²."""
        return [-k**2 for k in self.kappas]

    def forward_fcn(self, input):
        """Wrapper for use as a callable in training/pretraining."""
        input.requires_grad_(True)
        t = input[:, 0:1]
        x = input[:, 1:2]
        return self.u(x, t)

    def _u_tau(self, x, t):
        log_a = []
        for n in range(self.Ns):
            k = self.kappas[n]
            c0 = self.c0[n]
            log_a.append(np.log(c0**2 / (2*k)) + 2*k*x - 8*k**3*t)

        log_terms = []
        for S, coeff, kappa_sum in self.subsets:
            if coeff <= 0:
                continue
            if len(S) > 0:
                lt = np.log(coeff) + torch.zeros_like(x)
            else:
                lt = torch.zeros_like(x)
            for i in S:
                lt = lt + log_a[i]
            log_terms.append(lt)

        log_terms_stack = torch.stack(log_terms, dim=0)
        max_log = log_terms_stack.max(dim=0).values
        log_tau = max_log + torch.log(torch.sum(torch.exp(log_terms_stack - max_log), dim=0))

        log_tau_x = gradient(log_tau, x)
        log_tau_xx = gradient(log_tau_x, x)
        return 2 * log_tau_xx

    def _a(self, x, t):
        x_s = x.squeeze()
        t_s = t.squeeze()
        result = []
        for n in range(self.Ns):
            k = self.kappas[n]
            c0 = self.c0[n]
            cn_t = c0 * torch.exp(-4 * k**3 * t_s)
            result.append(cn_t**2 / (2 * k) * torch.exp(2 * k * x_s))
        return result
    
    def _tau_and_derivs(self, x, t):

        a = self._a(x, t)
        tau = torch.zeros_like(x.squeeze())
        tau_x = torch.zeros_like(tau)
        tau_xx = torch.zeros_like(tau)

        for S, coeff, kappa_sum in self.subsets:
            prod_a = coeff
            if len(S) == 0:
                term = torch.ones_like(tau)
            else:
                term = torch.ones_like(tau) * coeff
                for i in S:
                    term = term * a[i]
            
            tau = tau + term
            tau_x = tau_x + 2 * kappa_sum * term
            tau_xx = tau_xx + (2 * kappa_sum)**2 * term

        return tau, tau_x, tau_xx
    
class SchrodingerSolver:
    """Solves the Schrödinger eigenvalue problem for the KdV potential.

    Discretizes the operator L = -∂²_x + u and solves for eigenvalues
    and eigenfunctions to verify that the PINN or analytic solution
    preserves the isospectral properties.
    """
    def __init__(self):
        pass

    def solve_timeslice(self, u_vals, dx):
        """Solve eigenvalue problem for a single time slice.

        Constructs the discretized Hamiltonian H = -∂²_x + u and
        solves for eigenvalues and eigenvectors.

        Args:
            u_vals: Potential values on spatial grid
            dx: Grid spacing

        Returns:
            eigenvalues, eigenvectors (both numpy arrays)
        """
        u_int = u_vals[1:-1]
        diag = 2/dx**2 - u_int
        off = -np.ones(len(u_int)-1) / dx**2

        H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        return eigenvalues, eigenvectors

    def solve(self, u, dx):
        """Solve eigenvalue problem across all time slices.

        Args:
            u: Potential field, shape (num_time_steps, num_space_points)
            dx: Spatial grid spacing

        Returns:
            eigenvector_stack: shape (num_time_steps-1, num_interior_points, num_eigenvectors)
            eigenvalue_stack: shape (num_time_steps-1, num_eigenvalues)
        """
        eigenvector_stack = []
        eigenvalue_stack = []
        num_samp = u.shape[0]

        for t_idx in range(num_samp - 1):
            evs, psis = self.solve_timeslice(u[t_idx], dx)
            eigenvalue_stack.append(evs)
            eigenvector_stack.append(psis)

        eigenvector_stack = np.array(eigenvector_stack)
        eigenvalue_stack = np.array(eigenvalue_stack)
        return eigenvector_stack, eigenvalue_stack

    def check_SD(self, sd, input_eval, verbose=True):
        """Validate that scattering data is preserved (isospectrality).

        Checks that eigenvalues remain constant in time and that eigenfunctions
        have the correct time dependence exp(-4κ³t).

        Args:
            sd: ScatteringData object
            input_eval: Evaluation grid (t, x)
            verbose: If True, print detailed validation results

        Returns:
            Dictionary with validation metrics
        """
        t = input_eval[:, 0:1]
        x = input_eval[:, 1:2]

        num_samp = int(np.sqrt(input_eval.shape[0]))

        x_d = x.detach().reshape(num_samp, num_samp)[0, :].cpu().numpy()
        dx = x_d[1] - x_d[0]

        kappas = sd.kappas
        c0 = sd.c0
        Ns = sd.Ns

        u_vals = sd.u(x, t)
        u = u_vals.reshape(num_samp, num_samp).detach().cpu().numpy()

        eigenvector_stack, eigenvalue_stack = self.solve(u, dx)

        kappas_sorted = sorted(kappas, reverse=True)
        expected_eigenvalues_sorted = [-k**2 for k in kappas_sorted]

        num_time_steps = eigenvalue_stack.shape[0]
        eigenvalue_residuals = []
        eigenvalue_time_variance = []

        for n in range(Ns):
            lam_n = eigenvalue_stack[:, n]
            expected_lam = expected_eigenvalues_sorted[n]

            residual_n = np.abs(lam_n - expected_lam)
            eigenvalue_residuals.append(residual_n)

            time_variance = np.var(lam_n)
            eigenvalue_time_variance.append(time_variance)

        eigenvalue_residuals = np.array(eigenvalue_residuals)
        mse_eigenvalues = np.mean(eigenvalue_residuals**2)
        mean_time_variance = np.mean(eigenvalue_time_variance)

        t_vals = t.detach().reshape(num_samp, num_samp)[:, 0].cpu().numpy()
        t_vals = t_vals[:-1]

        psi_time_residuals = []
        for n in range(Ns):
            kappa_n = kappas_sorted[n]
            expected_decay = np.exp(-4 * kappa_n**3 * t_vals)

            psi_n_norms = []
            for t_idx in range(num_time_steps):
                psi_n_t = eigenvector_stack[t_idx, :, n]
                norm = np.sqrt(np.sum(psi_n_t**2) * dx)
                psi_n_norms.append(norm)
            psi_n_norms = np.array(psi_n_norms)

            if psi_n_norms[0] > 0:
                normalized_norms = psi_n_norms / psi_n_norms[0]
                residual = np.abs(normalized_norms - expected_decay)
                psi_time_residuals.append(residual)

        psi_time_residuals = np.array(psi_time_residuals)
        mse_psi_time = np.mean(psi_time_residuals**2)

        if verbose:
            print("=" * 60)
            print("Scattering Data Validation")
            print("=" * 60)
            print(f"\nNumber of solitons: {Ns}")
            print(f"Kappas (sorted): {[f'{k:.3f}' for k in kappas_sorted]}")
            print("\n--- Isospectrality Check ---")
            print(f"Expected eigenvalues: {[f'{ev:.4f}' for ev in expected_eigenvalues_sorted[:Ns]]}")
            print(f"Computed eigenvalues (t=0): {[f'{eigenvalue_stack[0, n]:.4f}' for n in range(Ns)]}")
            print(f"MSE (eigenvalues): {mse_eigenvalues:.6e}")
            print(f"Mean time variance: {mean_time_variance:.6e}")

            print("\n--- Time Dependence Check ---")
            print(f"MSE (psi time evolution): {mse_psi_time:.6e}")

            print("\n--- Summary ---")
            total_mse = mse_eigenvalues + mse_psi_time
            print(f"Total MSE: {total_mse:.6e}")

            if mse_eigenvalues < 1e-3 and mean_time_variance < 1e-6:
                print("PASS: Eigenvalues are isospectral")
            else:
                print("FAIL: Eigenvalues vary too much")

            if mse_psi_time < 1e-2:
                print("PASS: Eigenvectors have correct time dependence")
            else:
                print("FAIL: Eigenvectors time evolution is incorrect")
            print("=" * 60)

        results = {
            'eigenvalue_stack': eigenvalue_stack,
            'eigenvector_stack': eigenvector_stack,
            'expected_eigenvalues': expected_eigenvalues_sorted,
            'mse_eigenvalues': mse_eigenvalues,
            'mean_time_variance': mean_time_variance,
            'mse_psi_time': mse_psi_time,
            'total_mse': mse_eigenvalues + mse_psi_time,
            'eigenvalue_residuals': eigenvalue_residuals,
            'psi_time_residuals': psi_time_residuals
        }

        return results

