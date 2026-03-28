# KdV PINN

[![Post](https://img.shields.io/badge/📝-Read_the_Post-blue)](https://angerami.github.io/posts/2026/kdv-pinn/)
[![Demo](https://img.shields.io/badge/📊-Interactive_Demo-orange)](https://huggingface.co/spaces/angerami/kdv-pinn)

Physics-informed neural network for the Korteweg–de Vries equation, with boundary conditions specified through the inverse scattering transform rather than conventional Dirichlet/Neumann conditions. The KdV equation's integrability provides a rich validation framework: an infinite hierarchy of conservation laws serves as unsupervised diagnostics independent of the training loss, and the spectral structure of the associated Lax pair (isospectrality, eigenfunction dynamics) can be extracted and verified directly from the learned solution.

## Overview

- **`scattering.py`** — Inverse scattering transform: constructs exact N-soliton solutions from scattering data via the Marchenko matrix and τ-function formulations. Also solves the Schrödinger eigenvalue problem to extract spectral data from arbitrary potentials.
- **`physics.py`** — KdV operator and conservation hierarchy. Computes the PDE residual and conserved densities/fluxes (mass, momentum, energy, and higher) via autograd.
- **`train.py`** — Two-phase training: supervised pretraining against the analytic solution to select the correct topological sector, followed by physics-informed fine-tuning using the KdV residual.
- **`validation.py`** — End-to-end validation pipeline: trains, evaluates conservation law residuals, and runs inverse scattering to verify isospectra   lity.
- **`parameter_sweep.py`** — Sweeps over soliton number, κ values, and initial positions.

The [interactive demo](https://huggingface.co/spaces/angerami/kdv-pinn) computes analytic N-soliton solutions on the fly and displays the potential, eigenfunctions, and recovered eigenvalues. See the [writeup](https://angerami.github.io/posts/2026/kdv-pinn/) for background and results.

## Project Structure
```
kdv-pinn/
├── .github
│   └── workflows
│       └── sync_to_hf.yml
├── .gitignore
├── apps
│   └── streamlit_app.py
├── configuration.py
├── Dockerfile
├── equations.md
├── HF_README.md
├── models.py
├── notebooks
│   ├── analytic_solutions.ipynb
│   ├── example_kdv_slices.ipynb
│   └── kdv_pinn.ipynb
├── parameter_sweep.py
├── physics.py
├── plot_utils.py
├── README.md
├── requirements.txt
├── run_training.py
├── sampling.py
├── scattering.py
├── train.py
└── validation.py
```

## Quick Start

### Training
```bash
# Train with default configuration
python train.py

# Train with custom epochs
python train.py --epochs 5000

# Run full validation
python validation.py
```

### Interactive Application
```bash
# Launch Streamlit app for exploring N-soliton solutions
streamlit run apps/streamlit_app.py
```

### Notebooks
```bash
jupyter notebook notebooks/kdv_pinn.ipynb
```