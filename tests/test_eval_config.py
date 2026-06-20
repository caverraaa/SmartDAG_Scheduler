from src.eval.eval_config import EvalConfig, load_eval_config


def test_load_eval_config_from_config() -> None:
    e = load_eval_config("config.yaml")
    assert isinstance(e, EvalConfig)
    assert len(e.noise_std) >= 1 and len(e.beta) >= 1
    assert e.failures == [False, True]
    assert e.n_dags >= 1 and e.n_nodes >= 1
    assert len(e.noise_seeds) >= 1
    assert e.dag_seed_base >= 1
    assert e.benchmark_dir and e.checkpoint_glob and e.results_dir


def test_eval_config_is_frozen() -> None:
    import dataclasses

    import pytest

    e = load_eval_config("config.yaml")
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.n_dags = 99  # type: ignore[misc]
