"""Small normalization helpers for feature/reward scaling."""


def safe_div(value: float, ref: float, eps: float = 1e-8) -> float:
    """Divide by a reference, guarding against zero references."""
    return value / (ref + eps)
