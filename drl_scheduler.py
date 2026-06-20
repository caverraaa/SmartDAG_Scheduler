"""SmartDAG Scheduler CLI: train one RL model or run the evaluation grid (TZ §7, §12)."""

import argparse
import dataclasses
import os

import pandas as pd

from src.env.cluster_env import ClusterEnv
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
from src.utils.config import load_config


def cmd_train(seed: int, config_path: str = "config.yaml") -> str:
    """Train one model on sampled instances; save checkpoint + history CSV."""
    base = load_config(config_path)
    cfg = dataclasses.replace(base, seed=seed)
    policy = TwoHeadPolicy(
        GNNEncoder(hidden=cfg.gnn_hidden, layers=cfg.gnn_layers), hidden=cfg.gnn_hidden
    )
    trainer = PPOTrainer(policy, cfg)
    env = ClusterEnv(cfg)
    history = trainer.train(env, n_updates=cfg.total_updates)  # dag=None => sampled instances
    ckpt = os.path.join("models", f"rl_seed{seed}.pth")
    trainer.save_checkpoint(ckpt)
    results_dir = load_eval_config(config_path).results_dir
    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(history).to_csv(
        os.path.join(results_dir, f"train_history_seed{seed}.csv"), index=False
    )
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
