import pytest

from src.eval.significance import paired_wilcoxon


def test_all_equal_returns_p_one() -> None:
    stat, p = paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert stat == 0.0
    assert p == 1.0


def test_clear_difference_is_significant() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    b = [x + 5.0 for x in a]  # b strictly larger everywhere
    stat, p = paired_wilcoxon(a, b)
    assert p < 0.05


def test_matches_scipy_reference() -> None:
    from scipy.stats import wilcoxon

    a = [5.0, 3.0, 8.0, 2.0, 7.0, 6.0, 9.0]
    b = [4.0, 4.0, 6.0, 3.0, 5.0, 7.0, 6.0]
    stat, p = paired_wilcoxon(a, b)
    ref_stat, ref_p = wilcoxon(a, b)
    assert stat == pytest.approx(float(ref_stat))
    assert p == pytest.approx(float(ref_p))
