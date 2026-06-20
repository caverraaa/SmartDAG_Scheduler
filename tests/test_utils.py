import numpy as np

from src.utils.config import Config, load_config
from src.utils.normalization import safe_div
from src.utils.seeding import make_rng


def test_make_rng_is_deterministic_and_isolated() -> None:
    a = make_rng(42).random(5)
    b = make_rng(42).random(5)
    c = make_rng(7).random(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert isinstance(make_rng(0), np.random.Generator)


def test_safe_div_handles_zero_ref() -> None:
    assert abs(safe_div(10.0, 2.0) - 5.0) < 1e-6
    assert safe_div(1.0, 0.0) < 1e9  # eps prevents div-by-zero blow-up


def test_load_config_reads_defaults() -> None:
    cfg = load_config("config.yaml")
    assert isinstance(cfg, Config)
    assert cfg.w1 == 1.0 and cfg.w2 == 0.3 and cfg.w3 == 0.2
    assert cfg.noise_std == 0.0 and cfg.failure_rate == 0.0
    assert cfg.n_tasks == 30 and cfg.n_nodes == 8
