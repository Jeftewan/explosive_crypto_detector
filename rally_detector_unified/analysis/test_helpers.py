"""
Tests for analysis._helpers — verifies safety against duplicate DatetimeIndex
and cross-symbol boundary contamination.
"""
import numpy as np
import pandas as pd

from ._helpers import forward_returns_by_symbol, positional_select


def test_forward_returns_no_cross_symbol_boundary():
    idx = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.concat([
        pd.DataFrame({"symbol": "A", "close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx),
        pd.DataFrame({"symbol": "B", "close": [10.0, 1.0, 10.0, 1.0, 10.0]}, index=idx),
    ]).sort_index()

    r = forward_returns_by_symbol(df, horizon=1)

    a = r[df["symbol"] == "A"].values
    b = r[df["symbol"] == "B"].values

    assert np.allclose(a[:-1], [1.0, 0.5, 1.0 / 3.0, 0.25])
    assert np.allclose(b[:-1], [-0.9, 9.0, -0.9, 9.0])
    assert np.isnan(a[-1]) and np.isnan(b[-1])


def test_positional_select_with_duplicate_index():
    s = pd.Series([1, 2, 3, 4], index=[10, 10, 20, 20])
    mask = pd.Series([True, False, True, False], index=[10, 10, 20, 20])
    out = positional_select(s, mask)
    assert list(out) == [1, 3]


def test_positional_select_length_mismatch_raises():
    try:
        positional_select(np.arange(5), np.array([True, False]))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for length mismatch")


if __name__ == "__main__":
    test_forward_returns_no_cross_symbol_boundary()
    test_positional_select_with_duplicate_index()
    test_positional_select_length_mismatch_raises()
    print("ok")
