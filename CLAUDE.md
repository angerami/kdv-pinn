# kdv-pinn — project instructions

PINN for the KdV equation. Trains a network against the analytic IST (inverse
scattering transform) solution and validates by recovering the discrete spectrum.
Sibling to `hydro/` and `ym-pinn/` — `mlflow_utils.py` and `device.py` are ports;
keep them aligned by hand, not by import.

## Layout

- `src/kdv_pinn/` — the importable package (library code + `equations.md` data).
- `scripts/` — thin entry points (`run_training.py`, `parameter_sweep.py`) that
  import `kdv_pinn` and call into it.
- `configs/` — JSON run configs fed via `--config` (reproduce the blog runs).
- `apps/streamlit_app.py` — HF Space UI; bootstraps `src/` onto `sys.path`.
- Dev install: `pip install -e .[train,dev]` (or `-r requirements-dev.txt`).

## Experiment tracking — MLflow

MLflow is the system of record for every training run. The tracking store is a
local sqlite `mlflow.db` at the repo root (gitignored); worktrees log to the
main repo's db automatically. Browse with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

`validate_run` (in `src/kdv_pinn/validation.py`) owns the run boundary — it logs:
- all config fields as params, plus `git.commit` / `git.branch` / `git.dirty`;
- per-epoch training curves (replayed from the returned history);
- final validation metrics (κ recovery, field MAE/RMSE);
- artifacts: every plot, `metrics.txt`, and the trained model (`mlflow.pytorch`);
- physics tags (`physics.n_solitons`, `physics.kappas`, …) + a run note.

New training entry points should open a run via `mlflow_utils.start_run(cfg)` and
log the same way. Models live in the MLflow artifact store — never commit `.pt`.

## Device

`device.get_device()` is the single source of truth. `KDV_DEVICE=cpu|cuda|mps|auto`
(default `auto`: cuda → mps → cpu).

## Repo hygiene

- `scratch/`, `dev/`, `notes/`, `archive/`, `mlruns/`, `mlflow.db` are gitignored.
- Output dirs (`output*/`, `results/`, `sweep_results/`, `figures/`,
  `validation_output*/`) are gitignored — they're regenerable / live in MLflow.
- Never commit `.pt` / `.ckpt`.
- MLflow is an *extra* (`pip install -e .[train]`), not a core dependency. The
  HuggingFace/Streamlit deploy uses the lean `requirements.txt`, copies `src/`,
  and imports `kdv_pinn` via the app's `sys.path` bootstrap — don't pull mlflow
  into that path (the app only imports `scattering`/`configuration`/`sampling`).
