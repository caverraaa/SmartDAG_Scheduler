import numpy as np

from src.utils.seeding import derive_rng, make_rng


def test_derive_rng_is_reproducible() -> None:
    a = derive_rng(0, "noise").random(5)
    b = derive_rng(0, "noise").random(5)
    assert np.array_equal(a, b)


def test_derive_rng_streams_are_independent_per_salt() -> None:
    noise = derive_rng(0, "noise").random(5)
    failure = derive_rng(0, "failure").random(5)
    assert not np.array_equal(noise, failure)


def test_derive_rng_does_not_consume_base_stream() -> None:
    # Drawing from a derived stream must not perturb make_rng(seed)'s sequence.
    base_first = make_rng(0).random(3)
    _ = derive_rng(0, "noise").random(3)
    base_again = make_rng(0).random(3)
    assert np.array_equal(base_first, base_again)


def test_derive_rng_returns_generator() -> None:
    assert isinstance(derive_rng(1, "x"), np.random.Generator)
