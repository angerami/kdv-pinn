import torch
from itertools import combinations
from physics import gradient
import numpy as np

class ScatteringData:
    def __init__(self, kappas, x0s, R=None, autograd=True, use_tau=False):
        assert len(set(kappas)) == len(kappas), "kappas must be distinct"
        assert all(k > 0 for k in kappas), "kappas must be positive"

        self.kappas = kappas          # [κ₁, κ₂, ..., κ_N]
        self.x0 = x0s   # [c₁(0), c₂(0), ..., c_N(0)]
        self.c0 = [np.sqrt(2*k) * np.exp(k*x0) for k, x0 in zip(kappas, x0s)]
        assert all(c > 0 for c in self.c0), "norming constants must be positive"

        self.Ns= len(kappas)
        self.R = R                     # R(k), None for reflectionless
        self.autograd = autograd
        self.use_tau = use_tau
        if self.use_tau:
            self.alpha2 = {}
            for i in range(self.Ns):
                for j in range(i+1, self.Ns):
                    self.alpha2[(i,j)] = ((kappas[i] - kappas[j]) / (kappas[i] + kappas[j]))**2
            self.subsets = []
            for r in range(self.Ns + 1):
                for S in combinations(range(self.Ns), r):
                    coeff = 1.0
                    for i, j in combinations(S, 2):
                        coeff *= self.alpha2[(i,j)]
                    kappa_sum = sum(self.kappas[i] for i in S)
                    self.subsets.append((S, coeff, kappa_sum))

    def c(self, t):
        # t has shape (Npts, 1), squeeze to (Npts,) for proper broadcasting
        t_squeezed = t.squeeze()
        return [c0_n * torch.exp(-4 * k**3 * t_squeezed) for k, c0_n in zip(self.kappas, self.c0)]

    def A(self, x, t):
        c_t = self.c(t) #list (Ns) of tensors (Npts,)
        Npts = x.shape[0]
        x_squeezed = x.squeeze()  # (Npts,)

        # Initialize A as zeros that depend on x to maintain gradient connection
        A = torch.zeros(Npts, self.Ns, self.Ns, dtype=x.dtype, device=x.device)

        # Add identity matrix by multiplying with x-dependent term
        # Use a term like (1 + 0*x.sum()) to keep gradient connection
        identity_scale = 1.0 + 0.0 * x.sum()
        for i in range(self.Ns):
            A[:, i, i] = identity_scale

        for m in range(self.Ns):
            for n in range(self.Ns):
                # All terms are now 1D (Npts,) for proper broadcasting
                exp_term = torch.exp((self.kappas[m] + self.kappas[n]) * x_squeezed)
                A[:,m, n] = A[:,m, n] + c_t[m] * c_t[n] * exp_term / (self.kappas[m] + self.kappas[n])
        return A #(Npts, Ns, Ns)

    def u(self, x, t):
        if self.use_tau:
            # tau, tau_x, tau_xx = self._tau_and_derivs(x, t)
            # return (2 * (tau_xx * tau - tau_x**2) / tau**2).unsqueeze(-1)
            return self._u_tau(x,t)
        if self.autograd:
            # 2 ∂²_x log det A, via finite difference or autograd
            log_det = torch.logdet(self.A(x, t))
            log_det_x = gradient(log_det, x)
            log_det_xx = gradient(log_det_x, x)
            return 2 * log_det_xx
        
        else: #finite difference
            log_det = lambda xx: torch.logdet(self.A(xx, t))
            dx = 1e-3 #not 2L/N
            ld_p = log_det(x + dx)
            ld_0 = log_det(x)
            ld_m = log_det(x - dx)
            return 2 * (ld_p - 2*ld_0 + ld_m) / dx**2

    def psi(self, x, t, n, Ainv=None):
        # ψ_n = c_n(t) × Σ_m (A⁻¹)_nm × exp(-κ_m x)
        if Ainv is None:
            A = self.A(x, t)
            Ainv = torch.linalg.inv(A)  # (Npts, Ns, Ns)

        c_t = self.c(t)  # list of Ns tensors, each shape (Npts,)
        x_squeezed = x.squeeze()  # (Npts,)

        psi_n = torch.zeros_like(x_squeezed)  # (Npts,)
        for m in range(self.Ns):
            exp_term = torch.exp(-self.kappas[m] * x_squeezed)  # (Npts,)
            psi_n = psi_n + Ainv[:, n, m] * exp_term  # (Npts,)

        psi_n = c_t[n] * psi_n  # (Npts,)
        return psi_n.unsqueeze(-1)  # (Npts, 1)
                           

    def eigenvalues(self):
        return [-k**2 for k in self.kappas]
    
    def forward_fcn(self, input):
        input.requires_grad_(True)
        t = input[:, 0:1]
        x = input[:, 1:2]
        return self.u(x,t)

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