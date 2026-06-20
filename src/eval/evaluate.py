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


# ---------------------------------------------------------------------------
# Aggregation and output (Task 5)
# ---------------------------------------------------------------------------

_METRIC_COLS = [
    "makespan",
    "energy",
    "utilisation",
    "load_balance",
    "slr",
    "speedup",
    "overhead_ms",
]
_REGIME_COLS = ["noise_std", "beta", "failures"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean+std per (regime, strategy) for each metric, plus a robustness column."""
    grouped = df.groupby([*_REGIME_COLS, "strategy"])
    summary = grouped[_METRIC_COLS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    # robustness = mean over instances of (makespan std across noise seeds).
    per_instance = (
        df.groupby([*_REGIME_COLS, "strategy", "dag_label"])["makespan"].std().reset_index()
    )
    robustness = (
        per_instance.groupby([*_REGIME_COLS, "strategy"])["makespan"]
        .mean()
        .reset_index()
        .rename(columns={"makespan": "robustness"})
    )
    return summary.merge(robustness, on=[*_REGIME_COLS, "strategy"], how="left")


def compare_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Per regime, each rl@* vs each non-RL baseline, paired on (dag_label, noise_seed)."""
    from src.eval.significance import paired_wilcoxon

    rl_names = sorted({s for s in df["strategy"].unique() if s.startswith("rl@")})
    baseline_names = sorted({s for s in df["strategy"].unique() if not s.startswith("rl@")})
    rows: list[dict] = []
    for regime, sub in df.groupby(_REGIME_COLS):
        for rl in rl_names:
            rl_df = sub[sub["strategy"] == rl].set_index(["dag_label", "noise_seed"])
            for base_name in baseline_names:
                b_df = sub[sub["strategy"] == base_name].set_index(["dag_label", "noise_seed"])
                common = rl_df.index.intersection(b_df.index)
                a = [float(rl_df.loc[k, "makespan"]) for k in common]
                b = [float(b_df.loc[k, "makespan"]) for k in common]
                stat, p = paired_wilcoxon(a, b)
                rows.append(
                    {
                        "noise_std": regime[0],
                        "beta": regime[1],
                        "failures": regime[2],
                        "rl_strategy": rl,
                        "baseline": base_name,
                        "n_pairs": len(common),
                        "wilcoxon_stat": stat,
                        "p_value": p,
                    }
                )
    return pd.DataFrame(rows)


def write_results(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    results_dir: str,
) -> None:
    """Write raw rows, summary, and significance CSVs to results_dir."""
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, "eval_runs.csv"), index=False)
    summary.to_csv(os.path.join(results_dir, "eval_summary.csv"), index=False)
    significance.to_csv(os.path.join(results_dir, "eval_significance.csv"), index=False)


def print_tables(summary: pd.DataFrame, significance: pd.DataFrame) -> None:
    """Print one comparison table per regime followed by RL-vs-baseline p-values."""
    for regime, sub in summary.groupby(_REGIME_COLS):
        print(f"\n=== regime noise_std={regime[0]} beta={regime[1]} failures={regime[2]} ===")
        cols = [
            "strategy",
            "makespan_mean",
            "makespan_std",
            "energy_mean",
            "load_balance_mean",
            "slr_mean",
            "speedup_mean",
            "overhead_ms_mean",
            "robustness",
        ]
        print(sub[cols].to_string(index=False))
    if not significance.empty:
        print("\n=== RL vs baselines (Wilcoxon paired on makespan) ===")
        print(significance.to_string(index=False))
