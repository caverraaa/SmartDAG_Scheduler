"""Isolated RNG helpers (never touch global numpy state)."""

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return an independent generator seeded deterministically."""
    return np.random.default_rng(seed)
