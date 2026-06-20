"""Typed config loaded from config.yaml."""

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class Config:
    w1: float
    w2: float
    w3: float
    seed: int
    n_tasks: int
    n_nodes: int
    n_layers: int
    beta: float
    ccr: float
    edge_prob: float
    noise_std: float
    failure_rate: float


def load_config(path: str = "config.yaml") -> Config:
    """Parse the YAML config into a typed, frozen Config."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(
        w1=float(raw["w1"]),
        w2=float(raw["w2"]),
        w3=float(raw["w3"]),
        seed=int(raw["seed"]),
        n_tasks=int(raw["n_tasks"]),
        n_nodes=int(raw["n_nodes"]),
        n_layers=int(raw["n_layers"]),
        beta=float(raw["beta"]),
        ccr=float(raw["ccr"]),
        edge_prob=float(raw["edge_prob"]),
        noise_std=float(raw["noise_std"]),
        failure_rate=float(raw["failure_rate"]),
    )
