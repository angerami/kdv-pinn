# KdV Project

## Analytical Tools
Visualize soliton behavior for $N$ soliton solutions as specified by scattering data.
Use this to test later machinery.
Solutions to show:
- Single soliton
- 2 and 3 solitons
- Cascade
- Rolling train

Have yet to explore
- Non-zero reflection coefficients
- Alternative boundary conditions (cnoidal waves)

### Lax problem/scattering data/ eigenvalue problem
Given scattering data solve Schrodinger EV problem
Demonstrate solutions are Isospectral, Compute Lax operators and validate time expectation value, show time dependence of states matches what you get by applying evolution operator
Streamlit demo

### Deliverables
- Functinality mostly in `scattering.py`
- Streamlit front end `streamlit_app.py`

TODO: drill down on eigenvalue checker warnings

## KdV Hierarchy Using Autograd
Function `physics.kdv` computes:
- Field derivatives: $u_t, u_{x}, u_{xx}, u_{xxx}$
- The full KdV residual
- $(\rho_i, J_i)$ as fields and conservation residuals $= \rho_t - J_x$, with $i = 0, 1, 2$ corresponding to the field (mass), momentum and energy densities, respectively. 

### Deliverables
Notebook evaluating the `kdv` function on analytical soliton results should show:
- zero residuals
- fields consistent with those expected from previous study (compare to streamlit app plots)
This is more or less `analytic_solutions.ipynb` 

TODO: Is this fully up to date with latest developments in `physics.py` and `scattering.py`.

## Training

### Approach
Now `kdv` should be fully validated and thus we can use the residuals to train the PINN. 

Model implemented in `models.py` is simple MLP, with `tanh` activation units explicitly chosen (verus SIREN) to match the soliton shapes better: well separated solitons (sech solution) effectively are derivatives of tanh. 

For now, the boundary and initial conditions are enforced by evaluating the learned function and a boundary function on the boundary surfaces. Thus the loss is a simple MSE residual and the details are encoded in the boundary function. For the test cases we can simply use the analytic solutions themselves, turning it into a direct supervised learning problem on the boundary. 

TODO: does this last point make sense?

TODO: add a supervised loss for tracking only that does this evaluation on the interior and add to visualization

STATUS: training convergence brittle not scalable, needs deeper dive. only tried single soliton with limited success.

### Deliverables

Code to do this is in `training.py`. It can be driven either by `run_training.py` from the command line or using the notebook `kdv_pinn.ipynb`.

TODO: interactive mode works well but cmd line should generate a plot that gets updated or series of plots or something. ideally i could run it from the command line and open the PNG/PDF file in VS Code that would update (using the same frequency as from the interactive mode in the notebook)

STATUS: Probably more needs to be fleshed out as training becomes more stable


## Results
Can we show that a PINN trained to learn the KdV equation carries its spectral properties? 

Train PINN $P$ for KdV operator $K[u]$ and appropriate boundary conditions
$$
P(x,t) \rightarrow u(x,t)
$$ 

The operators $L$ and $A$ satisfying $L_t = [A, L]$, the Lax Pair are:
<!-- $$u_t + 6uu_x + u_{xxx} = 0$$ -->

$$L = -\partial_x^2 + u$$

$$A = -4\partial_x^3 + 6u\,\partial_x + 3u_x$$

Solve the eigenvalue problem:
$$
L\psi(x,t) = \lambda \psi(x,t)
$$

We should see that the solutions have the same properties as the analytic ones. 

TODO: set things up so we can just point the solver and validation code to the output of the PINN; that exact code has been validated on the analytic solutions already.

TODO: we have no direct checks on $A$ right now, even in the analytic, e.g. we do not monitor $A \psi(x,t) = \psi_t(x,t)$. Right now we validate the time dependence by comparing the wavefunctions to the time dependence extracted from the scattering data, but this isn't quite the same thing. However $A$ seems a touch messy as an operator as we have to compute a third order derivative for $\psi$ now as well. 

**Key Plot** - show that when applying all of this one can extract the scattering data consistent with the boundary conditions used to define the PINN.

**Key Plot** - Show first three residuals as functions of x and t are zero ("local"), and the space-integrated time-slices of the various densities should be constant ("global"). Have infrastructure to make these. 



