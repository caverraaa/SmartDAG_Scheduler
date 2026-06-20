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
    lr: float
    clip_eps: float
    gae_lambda: float
    ppo_epochs: int
    minibatch_size: int
    entropy_coef: float
    value_coef: float
    rollout_episodes: int
    total_updates: int
    max_grad_norm: float
    gnn_hidden: int
    gnn_layers: int


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
        lr=float(raw["lr"]),
        clip_eps=float(raw["clip_eps"]),
        gae_lambda=float(raw["gae_lambda"]),
        ppo_epochs=int(raw["ppo_epochs"]),
        minibatch_size=int(raw["minibatch_size"]),
        entropy_coef=float(raw["entropy_coef"]),
        value_coef=float(raw["value_coef"]),
        rollout_episodes=int(raw["rollout_episodes"]),
        total_updates=int(raw["total_updates"]),
        max_grad_norm=float(raw["max_grad_norm"]),
        gnn_hidden=int(raw["gnn_hidden"]),
        gnn_layers=int(raw["gnn_layers"]),
    )
