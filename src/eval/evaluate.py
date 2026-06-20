# src/eval/evaluate.py
"""Regime-grid evaluation runner (TZ §7, §10). Fairness is structural via run_episode."""

import dataclasses
import glob
import os

import pandas as pd
import torch

from src.dag_factory.factory import DAGFactory
from src.env.cluster_env import ClusterEnv
from src.env.cluster_factory import make_cluster
from src.eval.eval_config import EvalConfig
from src.eval.metrics import TimingStrategy, compute_run_metrics
from src.rl.gnn_encoder import GNNEncoder
from src.rl.policy import TwoHeadPolicy
from src.rl.rl_strategy import RLStrategy
from src.scheduler.task_scheduler import run_episode
from src.strategies.base import BaseSchedulingStrategy
from src.strategies.cpop import CPOPStrategy
from src.strategies.heft import HEFTStrategy
from src.strategies.min_min import MinMinStrategy
from src.strategies.random_strategy import RandomStrategy
from src.strategies.weighted_sum_greedy import WeightedSumGreedyStrategy
from src.utils.config import Config
from src.utils.seeding import derive_rng, make_rng


def build_dags(eval_cfg: EvalConfig, base_cfg: Config) -> list[tuple[str, object, int]]:
    """Held-out synthetic DAGs (pinned seeds) + every committed benchmark JSON."""
    dags: list[tuple[str, object, int]] = []
    for k in range(eval_cfg.n_dags):
        seed = eval_cfg.dag_seed_base + k
        n_tasks = eval_cfg.dag_sizes[k % len(eval_cfg.dag_sizes)]
        dag = DAGFactory.create(
            "synthetic",
            make_rng(seed),
            n_tasks=n_tasks,
            n_layers=base_cfg.n_layers,
            edge_prob=base_cfg.edge_prob,
            ccr=base_cfg.ccr,
        )
        dags.append((f"synthetic:{n_tasks}:{seed}", dag, seed))
    for path in sorted(glob.glob(os.path.join(eval_cfg.benchmark_dir, "*.json"))):
        name = os.path.basename(path).replace(".json", "")
        recipe = name.split("_")[0]
        # rng is unused by the deterministic parser; seed it from dag_seed_base for clarity.
        dag = DAGFactory.load_from_wfcommons(path, make_rng(eval_cfg.dag_seed_base), recipe=recipe)
        dags.append((f"bench:{name}", dag, eval_cfg.dag_seed_base))
    return dags


def load_checkpoints(eval_cfg: EvalConfig, base_cfg: Config) -> list[tuple[str, TwoHeadPolicy]]:
    """Load every checkpoint matching the glob into an eval-mode policy."""
    out: list[tuple[str, TwoHeadPolicy]] = []
    for path in sorted(glob.glob(eval_cfg.checkpoint_glob)):
        policy = TwoHeadPolicy(
            GNNEncoder(hidden=base_cfg.gnn_hidden, layers=base_cfg.gnn_layers),
            hidden=base_cfg.gnn_hidden,
        )
        policy.load_state_dict(torch.load(path, weights_only=True))
        policy.eval()
        out.append((os.path.basename(path).replace(".pth", ""), policy))
    return out


def build_strategies(
    base_cfg: Config, checkpoints: list[tuple[str, TwoHeadPolicy]]
) -> list[tuple[str, BaseSchedulingStrategy]]:
    """The fixed baseline set + one RLStrategy per loaded checkpoint."""
    strategies: list[tuple[str, BaseSchedulingStrategy]] = [
        ("heft", HEFTStrategy()),
        ("cpop", CPOPStrategy()),
        ("min_min", MinMinStrategy()),
        ("wsg", WeightedSumGreedyStrategy(base_cfg.w1, base_cfg.w2)),
        ("random", RandomStrategy(make_rng(base_cfg.seed))),
    ]
    for label, policy in checkpoints:
        strategies.append((f"rl@{label}", RLStrategy(policy)))
    return strategies


def run_grid(
    eval_cfg: EvalConfig,
    base_cfg: Config,
    checkpoints: list[tuple[str, TwoHeadPolicy]],
) -> pd.DataFrame:
    """Run every strategy on every (regime, instance, noise_seed); return raw rows."""
    dags = build_dags(eval_cfg, base_cfg)
    strategies = build_strategies(base_cfg, checkpoints)
    rows: list[dict] = []
    for noise_std in eval_cfg.noise_std:
        for beta in eval_cfg.beta:
            for failures in eval_cfg.failures:
                failure_rate = eval_cfg.failure_rate if failures else 0.0
                for dag_label, dag, dag_seed in dags:
                    # One cluster per (instance, beta); reused across noise seeds.
                    nodes = make_cluster(
                        derive_rng(dag_seed, f"cluster-beta{beta}"), eval_cfg.n_nodes, beta
                    )
                    for noise_seed in eval_cfg.noise_seeds:
                        cfg = dataclasses.replace(
                            base_cfg,
                            seed=noise_seed,
                            noise_std=noise_std,
                            failure_rate=failure_rate,
                        )
                        env = ClusterEnv(cfg)
                        for name, strategy in strategies:
                            timed = TimingStrategy(strategy)
                            schedule, info = run_episode(env, timed, dag=dag, nodes=nodes)
                            alive_ids = [n.node_id for n in env.state.nodes if n.alive]
                            metrics = compute_run_metrics(
                                schedule, info, dag, nodes, alive_ids, timed.predict_seconds
                            )
                            rows.append(
                                {
                                    **metrics,
                                    "noise_std": noise_std,
                                    "beta": beta,
                                    "failures": failures,
                                    "dag_label": dag_label,
                                    "noise_seed": noise_seed,
                                    "strategy": name,
                                }
                            )
    return pd.DataFrame(rows)
