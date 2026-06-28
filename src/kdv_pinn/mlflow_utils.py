"""MLflow plumbing for kdv-pinn — git-SHA tagging, run context, figure logging.

Ported from hydro/bjorken_pinn and ym-pinn. Tracking store is a local sqlite
mlflow.db at the repo root; worktrees log to the main repo's db so runs from a
branch land in the same store.
"""
from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import mlflow


def _resolve_tracking_uri() -> str:
    """Respect MLFLOW_TRACKING_URI; otherwise use the main repo's mlflow.db.

    Worktrees share the main repo's store: the main root is the parent of the
    common git dir.
    """
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        main_root = Path(common).resolve().parent
        return f"sqlite:///{main_root}/mlflow.db"
    except Exception:
        return f"sqlite:///{Path(__file__).parent}/mlflow.db"


mlflow.set_tracking_uri(_resolve_tracking_uri())


def get_git_info(repo_dir: str | Path | None = None) -> dict[str, str]:
    # Resolve git relative to the package source by default, not the process cwd,
    # so provenance is correct regardless of where the script is launched from.
    cwd = str(repo_dir) if repo_dir else str(Path(__file__).parent)
    def run(cmd: list[str]) -> str:
        return subprocess.check_output(
            cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    commit = run(["git", "rev-parse", "HEAD"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=cwd, text=True,
        stderr=subprocess.DEVNULL).strip())
    return {"git.commit": commit, "git.branch": branch, "git.dirty": str(dirty)}


def _config_params(cfg) -> dict:
    """Config fields as MLflow params. Accepts dataclass, SimpleNamespace, or dict.

    Scalars pass through; lists/tuples (kappas, x0s, MLP_dims) are stringified so
    they survive as params rather than being dropped.
    """
    if is_dataclass(cfg):
        raw = asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        raw = vars(cfg)
    else:
        raw = dict(cfg)
    params = {}
    for k, v in raw.items():
        if v is None:
            continue
        params[k] = v if isinstance(v, (int, float, bool, str)) else str(v)
    return params


@contextmanager
def start_run(cfg, experiment: str = "kdv_pinn", repo_dir: str | Path | None = None):
    """Start an MLflow run; log all config fields as params + git info as tags."""
    mlflow.set_experiment(experiment)
    base = getattr(cfg, "run_name", None) or getattr(cfg, "name", None)
    # Timestamp the run name so each execution is a distinct, persistent record
    # (don't reuse one name and clobber history). git.commit tag still identifies code.
    run_name = f"{base}_{datetime.now():%Y%m%d-%H%M%S}" if base else None
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_config_params(cfg))
        try:
            git = get_git_info(repo_dir)
            mlflow.set_tags(git)
            mlflow.log_params(git)
        except subprocess.CalledProcessError:
            pass
        mlflow.set_tag("mlflow.source.name", sys.argv[0] if sys.argv else "interactive")
        yield run


def log_pytorch_model(model, name: str = "model") -> None:
    mlflow.pytorch.log_model(model, name)


def register_pytorch_model(model, registered_name: str, description: str | None = None,
                           name: str = "model"):
    """Log the model to the active run and register a new version in the Model Registry.

    Sets the registered-model and version descriptions so the registry is
    self-documenting. Returns the ModelInfo.
    """
    info = mlflow.pytorch.log_model(model, name=name, registered_model_name=registered_name)
    if description:
        client = mlflow.MlflowClient()
        try:
            client.update_registered_model(registered_name, description=description)
            client.update_model_version(registered_name, info.registered_model_version,
                                        description=description)
        except Exception:
            pass
    return info


def set_run_description(text: str, tags: dict | None = None) -> None:
    """Set the active run's UI description (mlflow.note.content) plus optional tags."""
    mlflow.set_tag("mlflow.note.content", text)
    if tags:
        mlflow.set_tags(tags)


def set_experiment_description(experiment: str, text: str) -> None:
    """Set an experiment's UI description (rendered from its mlflow.note.content tag)."""
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is not None:
        client.set_experiment_tag(exp.experiment_id, "mlflow.note.content", text)


def set_run_metadata(cfg, description: str | None = None) -> None:
    """Attach KdV physics tags (soliton count, spectral params) + a note to the run."""
    kappas = getattr(cfg, "kappas", []) or []
    tags = {
        "physics.equation": "KdV",
        "physics.n_solitons": str(len(kappas)),
        "physics.kappas": str(kappas),
        "physics.x0s": str(getattr(cfg, "x0s", [])),
    }
    if description:
        tags["mlflow.note.content"] = description
    mlflow.set_tags(tags)


def log_figure(fig, name: str) -> None:
    """Save fig to a tempfile then log as artifact — sidesteps mlflow.log_figure
    re-rendering that fails on log-axis edge cases."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, name)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        mlflow.log_artifact(path)
