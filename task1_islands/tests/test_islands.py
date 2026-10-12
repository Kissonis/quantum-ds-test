"""Tests for the island counter."""

import io
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from islands import count_islands, parse_input  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "islands.py")


# --- reference cases from the task statement -------------------------------

def test_case_1():
    grid = [[0, 1, 0],
            [0, 0, 0],
            [0, 1, 1]]
    assert count_islands(grid) == 2


def test_case_2_diagonals_are_not_connected():
    grid = [[0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]]
    assert count_islands(grid) == 3


def test_case_3():
    grid = [[0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1]]
    assert count_islands(grid) == 2


# --- edge cases -------------------------------------------------------------

def test_empty_matrix():
    assert count_islands([]) == 0
    assert count_islands([[]]) == 0


def test_all_ocean():
    assert count_islands([[0] * 5 for _ in range(4)]) == 0


def test_all_land_is_one_island():
    assert count_islands([[1] * 5 for _ in range(4)]) == 1


def test_single_cell():
    assert count_islands([[1]]) == 1
    assert count_islands([[0]]) == 0


def test_single_row_and_column():
    assert count_islands([[1, 0, 1, 1, 0, 1]]) == 3
    assert count_islands([[1], [0], [1], [1]]) == 2


def test_checkerboard_every_cell_is_separate():
    grid = [[(r + c) % 2 for c in range(6)] for r in range(6)]
    assert count_islands(grid) == 18


def test_ring_with_a_hole():
    grid = [[1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]]
    assert count_islands(grid) == 1


def test_input_is_not_modified():
    grid = [[1, 1], [0, 1]]
    original = [row[:] for row in grid]
    count_islands(grid)
    assert grid == original


def test_large_map_does_not_hit_recursion_limit():
    # A 1000x1000 map of solid land: a recursive DFS would blow the stack.
    n = 1000
    grid = [[1] * n for _ in range(n)]
    assert count_islands(grid) == 1


# --- parsing ---------------------------------------------------------------

def test_parse_input():
    text = "3 3\n0 1 0\n0 0 0\n0 1 1\n"
    assert parse_input(io.StringIO(text)) == [[0, 1, 0], [0, 0, 0], [0, 1, 1]]


def test_parse_zero_dimensions():
    assert parse_input(io.StringIO("0 0\n")) == []


@pytest.mark.parametrize("text", [
    "",                       # no header
    "2 2\n1 0\n",             # too few values
    "2 2\n1 0 0 1 1\n",       # too many values
    "1 2\n1 2\n",             # value outside {0, 1}
])
def test_parse_errors(text):
    with pytest.raises(ValueError):
        parse_input(io.StringIO(text))


# --- CLI --------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("3 3\n0 1 0\n0 0 0\n0 1 1\n", "2"),
    ("3 4\n0 0 0 1\n0 0 1 0\n0 1 0 0\n", "3"),
    ("3 4\n0 0 0 1\n0 0 1 1\n0 1 0 1\n", "2"),
])
def test_cli(text, expected):
    result = subprocess.run(
        [sys.executable, SCRIPT], input=text, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == expected


# --- robustness of the public API -------------------------------------------

def test_ragged_grid_is_rejected_with_clear_error():
    with pytest.raises(ValueError, match="rectangular"):
        count_islands([[1, 1], [1]])


def test_non_integer_header_is_rejected_with_clear_error():
    with pytest.raises(ValueError, match="two integers"):
        parse_input(["a b", "0 1"])
