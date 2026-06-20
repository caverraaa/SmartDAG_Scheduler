# tests/test_evaluate.py
from src.eval.eval_config import EvalConfig
from src.eval.evaluate import build_dags, build_strategies, run_grid
from src.utils.config import load_config


def _eval_cfg() -> EvalConfig:
    return EvalConfig(
        noise_std=[0.0],
        beta=[2.0],
        failure_rate=0.0,
        failures=[False],
        n_dags=2,
        dag_sizes=[20],
        n_nodes=4,
        noise_seeds=[0, 1],
        dag_seed_base=100000,
        benchmark_dir="dag_benchmarks",
        checkpoint_glob="models/__none__*.pth",  # no checkpoints in this smoke test
        results_dir="results",
    )


def test_build_dags_includes_synthetic_and_benchmarks() -> None:
    e = _eval_cfg()
    dags = build_dags(e, load_config("config.yaml"))
    labels = [lbl for lbl, _dag, _seed in dags]
    assert sum(1 for lbl in labels if lbl.startswith("synthetic")) == 2
    assert any(lbl.startswith("bench:") for lbl in labels)  # committed dag_benchmarks/*.json


def test_run_grid_fairness_identical_instance_sets() -> None:
    e = _eval_cfg()
    base = load_config("config.yaml")
    strategies = build_strategies(base, checkpoints=[])
    df = run_grid(e, base, checkpoints=[])
    # Every strategy must have been run on the identical (dag_label, noise_seed) set.
    keysets = {
        name: set(map(tuple, df[df["strategy"] == name][["dag_label", "noise_seed"]].values))
        for name, _ in strategies
    }
    reference = next(iter(keysets.values()))
    assert all(ks == reference for ks in keysets.values())
    assert len(reference) >= 1


def test_run_grid_has_metric_columns() -> None:
    e = _eval_cfg()
    df = run_grid(e, load_config("config.yaml"), checkpoints=[])
    for col in [
        "makespan",
        "energy",
        "utilisation",
        "load_balance",
        "slr",
        "speedup",
        "overhead_ms",
        "noise_std",
        "beta",
        "failures",
        "dag_label",
        "noise_seed",
        "strategy",
    ]:
        assert col in df.columns


def test_summarize_and_significance_and_write(tmp_path) -> None:
    from src.eval.evaluate import compare_significance, run_grid, summarize, write_results

    base = load_config("config.yaml")
    # Build a tiny grid with ONE fake checkpoint so the RL-vs-baseline path runs.
    from src.rl.gnn_encoder import GNNEncoder
    from src.rl.policy import TwoHeadPolicy

    policy = TwoHeadPolicy(
        GNNEncoder(hidden=base.gnn_hidden, layers=base.gnn_layers), hidden=base.gnn_hidden
    )
    df = run_grid(_eval_cfg(), base, checkpoints=[("rl_seed0", policy)])

    summary = summarize(df)
    assert {"makespan_mean", "makespan_std", "robustness"}.issubset(summary.columns)

    sig = compare_significance(df)
    assert {"rl_strategy", "baseline", "p_value", "n_pairs"}.issubset(sig.columns)
    assert len(sig) >= 1  # at least one rl-vs-baseline comparison

    write_results(df, summary, sig, str(tmp_path))
    import os

    for fname in ["eval_runs.csv", "eval_summary.csv", "eval_significance.csv"]:
        assert os.path.exists(os.path.join(str(tmp_path), fname))
