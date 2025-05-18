import os
import csv
import pytest
import tempfile
from ..utils import (
    generate_random_zero_pairs,
    generate_real_positions,
    round_to_nearest_hundred,
    apply_trades,
    write_to_csv,
)

def test_generate_random_zero_pairs_bounds():
    # lower=0, upper=0 should produce exactly zero pairs list of length 0
    zeros = generate_random_zero_pairs(0, 0)
    assert isinstance(zeros, list)
    assert all(pair == (0,0) for pair in zeros)
    # bounds check
    for _ in range(10):
        zs = generate_random_zero_pairs(1, 3)
        assert 1 <= len(zs) <= 3


def test_generate_real_positions_exact():
    # exact count
    positions = generate_real_positions(5)
    assert len(positions) == 5
    for start, end in positions:
        assert start == end
        assert start % 10 == 0


def test_round_to_nearest_hundred():
    assert round_to_nearest_hundred(150) == 200
    assert round_to_nearest_hundred(149) == 100
    assert round_to_nearest_hundred(-150) == -200


def test_apply_trades_positive_and_negative():
    positions = [(100,100), (-50,-50), (30,30)]
    trades = {0: 20, 1: 10}
    updated = apply_trades(trades, positions)
    # for positive start=100, end=100-20=80
    assert updated[0] == (100, 80)
    # for negative start=-50, end=-50+10=-40
    assert updated[1] == (-50, -40)
    # unchanged index 2
    assert updated[2] == (30,30)


def test_write_to_csv_and_io_error(tmp_path, monkeypatch):
    data = [(1,2),(3,4)]
    file = tmp_path / "out.csv"
    write_to_csv(data, str(file))
    # ensure file written with header and rows
    with open(file, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert rows[0] == ["Start","End"]
    assert rows[1] == ['1', '2']
    assert rows[2] == ['3', '4']
    # simulate permission error
    def fake_open(*args, **kwargs):
        raise OSError("perm")
    monkeypatch.setattr("builtins.open", fake_open)
    with pytest.raises(OSError):
        write_to_csv(data, "irrelevant.csv")

