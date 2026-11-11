import numpy as np
import pandas as pd
import pytest

from fpa import pvm


def _check_sum(b):
    assert b.volume + b.mix + b.price == pytest.approx(b.actual - b.base, abs=1e-9)


def test_effects_sum_exactly_to_variance():
    q0, p0 = np.array([10, 20, 5.0]), np.array([2.0, 3.0, 10.0])
    q1, p1 = np.array([12, 15, 9.0]), np.array([2.2, 2.9, 9.0])
    b = pvm.bridge(q0, q0 * p0, q1, q1 * p1)
    _check_sum(b)
    assert b.base == pytest.approx(130)
    assert b.actual == pytest.approx(12 * 2.2 + 15 * 2.9 + 81)


def test_zero_variance_gives_zero_effects():
    q, r = np.array([3.0, 4.0]), np.array([6.0, 20.0])
    b = pvm.bridge(q, r, q, r)
    assert (b.volume, b.mix, b.price) == pytest.approx((0, 0, 0))


def test_pure_price_change_only_price_effect():
    q = np.array([3.0, 4.0])
    b = pvm.bridge(q, q * np.array([2.0, 5.0]), q, q * np.array([2.5, 5.5]))
    assert b.volume == pytest.approx(0)
    assert b.mix == pytest.approx(0)
    assert b.price == pytest.approx(3 * 0.5 + 4 * 0.5)


def test_pure_mix_shift_only_mix_effect():
    p = np.array([2.0, 10.0])
    q0, q1 = np.array([5.0, 5.0]), np.array([2.0, 8.0])  # same total units, richer mix
    b = pvm.bridge(q0, q0 * p, q1, q1 * p)
    assert b.volume == pytest.approx(0)
    assert b.price == pytest.approx(0)
    assert b.mix == pytest.approx((2 * 2 + 8 * 10) - (5 * 2 + 5 * 10))


def test_pure_volume_change_only_volume_effect():
    p = np.array([2.0, 10.0])
    q0 = np.array([5.0, 5.0])
    b = pvm.bridge(q0, q0 * p, 2 * q0, 2 * q0 * p)
    assert b.volume == pytest.approx(60)
    assert b.mix == pytest.approx(0) and b.price == pytest.approx(0)


def test_new_and_discontinued_items():
    q0, r0 = np.array([5.0, 0.0, 3.0]), np.array([10.0, 0.0, 9.0])
    q1, r1 = np.array([4.0, 6.0, 0.0]), np.array([10.0, 30.0, 0.0])
    b = pvm.bridge(q0, r0, q1, r1)
    _check_sum(b)


def test_empty_base_group():
    b = pvm.bridge(np.zeros(2), np.zeros(2), np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    _check_sum(b)
    assert b.volume == 0


def test_bridge_by_group_and_roll_up():
    df = pd.DataFrame(
        {
            "store_id": ["A", "A", "B", "B"],
            "dept_id": ["X", "X", "X", "X"],
            "q0": [1.0, 2.0, 3.0, 4.0],
            "r0": [2.0, 6.0, 3.0, 8.0],
            "q1": [2.0, 2.0, 3.0, 5.0],
            "r1": [5.0, 6.0, 3.3, 10.0],
        }
    )
    br = pvm.bridge_by_group(df, ["store_id", "dept_id"])
    assert len(br) == 2
    tot = pvm.roll_up(br, [])
    t = tot.iloc[0]
    assert t["volume"] + t["mix"] + t["price"] == pytest.approx(t["variance"])
    by = pvm.roll_up(br, ["dept_id"])
    assert by["variance"].iloc[0] == pytest.approx(df["r1"].sum() - df["r0"].sum())
