"""Isolated RNG helpers (never touch global numpy state)."""

import hashlib

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return an independent generator seeded deterministically."""
    return np.random.default_rng(seed)


def derive_rng(seed: int, salt: str) -> np.random.Generator:
    """Independent sub-stream keyed by (seed, salt) for per-concern RNG isolation.

    Drawing from a derived stream never consumes from make_rng(seed) or any other
    salt's stream, so toggling one concern (e.g. noise) cannot perturb another's
    draws or the base DAG/cluster-generation stream.
    """
    sub = int.from_bytes(hashlib.sha256(salt.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, sub]))
