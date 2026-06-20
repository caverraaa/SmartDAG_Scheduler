# tests/test_cli.py
import os

import pandas as pd


def _fast_config(tmp_path) -> str:
    """Write a tiny config.yaml copy with a 2-update training budget for speed."""
    import yaml

    with open("config.yaml", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["total_updates"] = 2
    raw["rollout_episodes"] = 1
    raw["gnn_hidden"] = 16
    raw["eval"] = {
        "noise_std": [0.0],
        "beta": [2.0],
        "failure_rate": 0.0,
        "failures": [False],
        "n_dags": 1,
        "dag_sizes": [20],
        "n_nodes": 4,
        "noise_seeds": [0],
        "dag_seed_base": 100000,
        "benchmark_dir": "dag_benchmarks",
        "checkpoint_glob": os.path.join(str(tmp_path), "models", "rl_seed*.pth"),
        "results_dir": os.path.join(str(tmp_path), "results"),
    }
    path = os.path.join(str(tmp_path), "config.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh)
    return path


def test_cmd_train_writes_checkpoint_and_history(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg_path = _fast_config(tmp_path)
    from drl_scheduler import cmd_train

    ckpt = cmd_train(seed=0, config_path=cfg_path)
    assert os.path.exists(ckpt)
    assert os.path.exists(os.path.join(str(tmp_path), "results", "train_history_seed0.csv"))


def test_cmd_eval_writes_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # benchmark dir must exist for build_dags glob (empty is fine here).
    os.makedirs(os.path.join(str(tmp_path), "dag_benchmarks"), exist_ok=True)
    cfg_path = _fast_config(tmp_path)
    from drl_scheduler import cmd_eval, cmd_train

    cmd_train(seed=0, config_path=cfg_path)  # produce one checkpoint for the glob
    cmd_eval(config_path=cfg_path)
    summary = os.path.join(str(tmp_path), "results", "eval_summary.csv")
    assert os.path.exists(summary)
    assert len(pd.read_csv(summary)) >= 1
