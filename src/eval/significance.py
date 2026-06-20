"""Paired significance test for RL-vs-baseline comparisons (TZ §10)."""

from scipy.stats import wilcoxon


def paired_wilcoxon(a: list[float], b: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank over paired samples; (statistic, p_value).

    Returns (0.0, 1.0) when every paired difference is zero (scipy raises in
    that degenerate case), so callers never need to special-case identical runs.
    """
    if len(a) != len(b):
        raise ValueError("paired_wilcoxon requires equal-length samples.")
    if all(x == y for x, y in zip(a, b, strict=False)):
        return (0.0, 1.0)
    stat, p = wilcoxon(a, b)
    return (float(stat), float(p))
