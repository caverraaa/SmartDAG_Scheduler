"""Typed config for the M5 evaluation grid, loaded from config.yaml `eval:` block."""

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class EvalConfig:
    noise_std: list[float]
    beta: list[float]
    failure_rate: float
    failures: list[bool]
    n_dags: int
    dag_sizes: list[int]
    n_nodes: int
    noise_seeds: list[int]
    dag_seed_base: int
    benchmark_dir: str
    checkpoint_glob: str
    results_dir: str


def load_eval_config(path: str = "config.yaml") -> EvalConfig:
    """Parse the ``eval:`` block of config.yaml into a typed, frozen EvalConfig."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    e = raw["eval"]
    return EvalConfig(
        noise_std=[float(x) for x in e["noise_std"]],
        beta=[float(x) for x in e["beta"]],
        failure_rate=float(e["failure_rate"]),
        failures=[bool(x) for x in e["failures"]],
        n_dags=int(e["n_dags"]),
        dag_sizes=[int(x) for x in e["dag_sizes"]],
        n_nodes=int(e["n_nodes"]),
        noise_seeds=[int(x) for x in e["noise_seeds"]],
        dag_seed_base=int(e["dag_seed_base"]),
        benchmark_dir=str(e["benchmark_dir"]),
        checkpoint_glob=str(e["checkpoint_glob"]),
        results_dir=str(e["results_dir"]),
    )
