"""SmartDAG Scheduler CLI: train one RL model or run the evaluation grid (TZ §7, §12)."""

import argparse
import dataclasses
import os
import random
import statistics

import numpy as np
import pandas as pd
import torch

from src.core.compute_node import ComputeNode
from src.core.dag import TaskDAG
from src.dag_factory.factory import DAGFactory
from src.env.cluster_env import ClusterEnv
from src.env.cluster_factory import make_cluster
from src.eval.eval_config import load_eval_config
from src.eval.evaluate import (
    compare_significance,
    load_checkpoints,
    print_tables,
    run_grid,
    summarize,
    write_results,
)
from src.rl.gnn_encoder import GNNEncoder
from src.rl.policy import TwoHeadPolicy
from src.rl.ppo_trainer import PPOTrainer
from src.rl.rl_strategy import RLStrategy
from src.scheduler.task_scheduler import run_episode
from src.strategies.heft import HEFTStrategy
from src.strategies.random_strategy import RandomStrategy
from src.utils.config import Config, load_config
from src.utils.seeding import derive_rng, make_rng

Instance = tuple[TaskDAG, list[ComputeNode]]


def _build_val_instances(cfg: Config) -> list[Instance]:
    """Fixed validation set from a dedicated seed stream (disjoint from train + eval grid)."""
    insts: list[Instance] = []
    for k in range(cfg.n_val_dags):
        s = cfg.val_seed_base + k
        dag = DAGFactory.create(
            "synthetic",
            make_rng(s),
            n_tasks=cfg.n_tasks,
            n_layers=cfg.n_layers,
            edge_prob=cfg.edge_prob,
            ccr=cfg.ccr,
        )
        nodes = make_cluster(derive_rng(s, "val-cluster"), cfg.n_nodes, cfg.beta)
        insts.append((dag, nodes))
    return insts


def _baselines(val: list[Instance], cfg: Config) -> list[tuple[float, float, float, float]]:
    """Per-instance (HEFT makespan, HEFT energy, HEFT balance, Random balance), clean env, once."""
    clean = dataclasses.replace(cfg, noise_std=0.0, failure_rate=0.0)
    env = ClusterEnv(clean)
    heft, rnd = HEFTStrategy(), RandomStrategy(make_rng(0))
    out: list[tuple[float, float, float, float]] = []
    for dag, nodes in val:
        ids = [n.node_id for n in nodes]
        hs, _ = run_episode(env, heft, dag=dag, nodes=nodes)
        ns, _ = run_episode(env, rnd, dag=dag, nodes=nodes)
        out.append(
            (hs.makespan(), hs.total_energy, hs.load_balance_index(ids), ns.load_balance_index(ids))
        )
    return out


def _validate_rl(
    policy: TwoHeadPolicy,
    val: list[Instance],
    baselines: list[tuple[float, float, float, float]],
    cfg: Config,
) -> dict[str, float]:
    """Frozen-policy eval-vs-HEFT (TZ §9): mean makespan AND energy ratio + balance.

    Energy is half the structural claim, so it is tracked live alongside makespan/balance
    to catch a balance-via-high-wattage regression during training, not at final eval.
    """
    clean = dataclasses.replace(cfg, noise_std=0.0, failure_rate=0.0)
    env = ClusterEnv(clean)
    rl = RLStrategy(policy)
    mk_ratios, en_ratios, rl_bal, objective = [], [], [], []
    for (dag, nodes), (h_mk, h_en, _, _) in zip(val, baselines, strict=True):
        ids = [n.node_id for n in nodes]
        rs, info = run_episode(env, rl, dag=dag, nodes=nodes)
        mk_ratios.append(rs.makespan() / h_mk if h_mk > 0 else float("inf"))
        en_ratios.append(rs.total_energy / h_en if h_en > 0 else float("inf"))
        bal = rs.load_balance_index(ids)
        rl_bal.append(bal)
        # The agent's own multi-criteria objective (higher = better) = the training reward
        # shape: -(w1*mk/Mref + w2*en/Eref) + w3*balance. Used to pick the BEST checkpoint
        # so we keep the Pareto-best policy, not the makespan-only-best one.
        objective.append(
            -(cfg.w1 * rs.makespan() / info["m_ref"] + cfg.w2 * rs.total_energy / info["e_ref"])
            + cfg.w3 * bal
        )
    return {
        "mk_ratio_vs_heft": statistics.mean(mk_ratios),
        "energy_ratio_vs_heft": statistics.mean(en_ratios),
        "rl_balance": statistics.mean(rl_bal),
        "heft_balance": statistics.mean([b[2] for b in baselines]),
        "random_balance": statistics.mean([b[3] for b in baselines]),
        "val_objective": statistics.mean(objective),
    }


def cmd_train(seed: int, config_path: str = "config.yaml") -> str:
    """Train one model on sampled instances; save the BEST checkpoint + a curve CSV.

    With eval_interval > 0, every eval_interval updates the frozen policy is evaluated
    vs HEFT on a fixed validation set (makespan ratio + balance), logged to the history
    CSV, and the checkpoint with the best makespan ratio is kept (TZ §9). Entropy is
    annealed entropy_coef -> entropy_coef_final across the run by the trainer.
    """
    base = load_config(config_path)
    cfg = dataclasses.replace(base, seed=seed)
    # Reproducibility (invariant #6): --seed must drive torch (policy init + action
    # sampling) and the global numpy/random streams, not just the env's isolated RNGs.
    # (PYTHONHASHSEED must be exported before launch; it cannot be set from here.)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    policy = TwoHeadPolicy(
        GNNEncoder(hidden=cfg.gnn_hidden, layers=cfg.gnn_layers), hidden=cfg.gnn_hidden
    )
    trainer = PPOTrainer(policy, cfg)
    env = ClusterEnv(cfg)
    ckpt = os.path.join("models", f"rl_seed{seed}.pth")
    os.makedirs("models", exist_ok=True)
    results_dir = load_eval_config(config_path).results_dir
    os.makedirs(results_dir, exist_ok=True)
    hist_path = os.path.join(results_dir, f"train_history_seed{seed}.csv")
    # Start clean: never leave a stale prior-run curve in place while this run warms up.
    if os.path.exists(hist_path):
        os.remove(hist_path)
    history: list[dict] = []

    if cfg.eval_interval and cfg.eval_interval > 0:
        val = _build_val_instances(cfg)
        baselines = _baselines(val, cfg)
        best_objective = float("-inf")
        done = 0
        while done < cfg.total_updates:
            k = min(cfg.eval_interval, cfg.total_updates - done)
            chunk = trainer.train(env, n_updates=k)  # dag=None => sampled instances
            done += k
            v = _validate_rl(policy, val, baselines, cfg)
            history.append({"update": done, **chunk[-1], **v})
            # Write the Pareto curve INCREMENTALLY each interval: live half-budget reads,
            # crash-resilient, and overwrites any stale file at the first eval point.
            pd.DataFrame(history).to_csv(hist_path, index=False)
            if v["val_objective"] > best_objective:
                best_objective = v["val_objective"]
                trainer.save_checkpoint(ckpt)  # keep BEST by multi-criteria objective, not last
            print(
                f"[seed{seed}] upd {done}/{cfg.total_updates} obj={v['val_objective']:.3f} "
                f"mk/HEFT={v['mk_ratio_vs_heft']:.2f} en/HEFT={v['energy_ratio_vs_heft']:.2f} "
                f"rl_bal={v['rl_balance']:.3f} (rand {v['random_balance']:.3f}, "
                f"heft {v['heft_balance']:.3f}) ent={chunk[-1]['entropy']:.2f} "
                f"ent_coef={chunk[-1]['entropy_coef']:.3f} reward={chunk[-1]['mean_reward']:.2f}",
                flush=True,
            )
        if best_objective == float("-inf"):
            trainer.save_checkpoint(ckpt)
    else:
        history = trainer.train(env, n_updates=cfg.total_updates)
        trainer.save_checkpoint(ckpt)
        pd.DataFrame(history).to_csv(hist_path, index=False)
    return ckpt


def cmd_eval(config_path: str = "config.yaml") -> None:
    """Run the regime grid over loaded checkpoints; write CSVs + print tables."""
    base = load_config(config_path)
    eval_cfg = load_eval_config(config_path)
    checkpoints = load_checkpoints(eval_cfg, base)
    df = run_grid(eval_cfg, base, checkpoints)
    summary = summarize(df)
    significance = compare_significance(df)
    write_results(df, summary, significance, eval_cfg.results_dir)
    print_tables(summary, significance)


def main(argv: list[str] | None = None) -> None:
    # Cap torch's thread pools so parallel per-seed processes don't oversubscribe cores
    # (the simulator is GIL-bound; parallelism comes from processes, not torch threads).
    _threads = os.environ.get("TORCH_NUM_THREADS")
    if _threads:
        torch.set_num_threads(int(_threads))
    parser = argparse.ArgumentParser(description="SmartDAG Scheduler train/eval CLI.")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train", help="train one RL model")
    p_train.add_argument("--seed", type=int, required=True)
    sub.add_parser("eval", help="run the evaluation grid")
    args = parser.parse_args(argv)
    if args.command == "train":
        path = cmd_train(args.seed, args.config)
        print(f"saved {path}")
    elif args.command == "eval":
        cmd_eval(args.config)


if __name__ == "__main__":
    main()
