# KdV Equation: Definitions and Conserved Quantities

## PDE

<!-- eq:kdv -->
$u_t + 6uu_x + u_{xxx} = 0$

Soliton solution: $u = 2\kappa^2 \operatorname{sech}^2[\kappa(x - 4\kappa^2 t - x_0)]$

---

## Fields and Derivatives

<!-- desc:u --> Solution field
<!-- eq:u -->
$u$

&nbsp;

<!-- desc:u_t --> Time derivative
<!-- eq:u_t -->
$u_t$

&nbsp;

<!-- desc:u_x --> Spatial derivative
<!-- eq:u_x -->
$u_x$

&nbsp;

<!-- desc:u_xx --> Second spatial derivative
<!-- eq:u_xx -->
$u_{xx}$

&nbsp;

<!-- desc:u_xxx --> Third spatial derivative
<!-- eq:u_xxx -->
$u_{xxx}$

---

## Conserved Densities

Satisfy $\partial_t \rho_n + \partial_x J_n = 0$.

&nbsp;

<!-- desc:rho_1 --> **Momentum density (mass)**
<!-- eq:rho_1 --> <!-- def:rho_1 -->
$\rho_1 = \frac{1}{2}u^2$

&nbsp;

<!-- desc:rho_2 --> **Energy density (Hamiltonian)**
<!-- eq:rho_2 --> <!-- def:rho_2 -->
$\rho_2 = -u^3 + \frac{1}{2}u_x^2$

&nbsp;

<!-- desc:rho_3 --> **Third conserved density** — *signs to be numerically verified*
<!-- eq:rho_3 --> <!-- def:rho_3 -->
$\rho_3 = -\frac{5}{2}u^4 + 5uu_x^2 - \frac{1}{2}u_{xx}^2$

---

## Fluxes

<!-- desc:J_0 --> **Flux for** $\rho_1$
<!-- eq:J_0 --> <!-- def:J_0 -->
$J_0 = 3u^2 + u_{xx}$

&nbsp;

<!-- desc:J_1 --> **Flux for** $\rho_2$
<!-- eq:J_1 --> <!-- def:J_1 -->
$J_1 = -\frac{9}{2}u^4 - 3u^2 u_{xx} + 6uu_x^2 + u_x u_{xxx} - \frac{1}{2}u_{xx}^2$

---

## Residuals

<!-- desc:res_kdv --> **KdV PDE residual**
<!-- eq:res_kdv --> <!-- def:res_kdv -->
$\mathcal{R}_{\mathrm{KdV}} = u_t + 6uu_x + u_{xxx}$

&nbsp;

<!-- desc:res_H0 --> **Conservation residual for momentum**
<!-- eq:res_H0 --> <!-- def:res_H0 -->
$\mathcal{R}_{H_0} = \partial_t \rho_1 + \partial_x J_0$

&nbsp;

<!-- desc:res_H1 --> **Conservation residual for energy**
<!-- eq:res_H1 --> <!-- def:res_H1 -->
$\mathcal{R}_{H_1} = \partial_t \rho_2 + \partial_x J_1$