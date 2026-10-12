"""Count islands on a binary map.

The map is an M x N matrix where ``1`` is land and ``0`` is ocean.
Two land cells belong to the same island when they share an edge
(4-connectivity: up / down / left / right).  Diagonal neighbours are
**not** connected -- this is what the reference test cases require.

Algorithm
---------
Iterative flood fill (depth-first search with an explicit stack).
Every cell is visited at most twice (once by the outer scan, once by a
flood fill), so the running time is O(M * N), which is optimal: any
correct algorithm has to read every cell at least once.  Extra memory is
O(M * N) in the worst case for the stack (a map that is all land), and
no recursion is used, so Python's recursion limit is never an issue.

Usage
-----
    python islands.py < input.txt

Input format::

    M N
    row_1
    ...
    row_M

Each row contains N integers (0 or 1) separated by whitespace.
"""

from __future__ import annotations

import sys
from typing import Iterable, List, Sequence

Matrix = List[List[int]]

# Offsets of the four edge-sharing neighbours.
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def count_islands(grid: Sequence[Sequence[int]]) -> int:
    """Return the number of 4-connected islands in ``grid``.

    Parameters
    ----------
    grid:
        Rectangular matrix of 0/1 values. It is **not** modified.

    Returns
    -------
    int
        Number of islands. An empty matrix contains zero islands.

    Raises
    ------
    ValueError
        If the rows of ``grid`` have different lengths.
    """
    rows = len(grid)
    if rows == 0:
        return 0
    cols = len(grid[0])
    for i, row in enumerate(grid):
        if len(row) != cols:
            raise ValueError(
                f"grid must be rectangular: row 0 has {cols} values, row {i} has {len(row)}"
            )
    if cols == 0:
        return 0

    # Work on a private copy so the caller's data stays intact.  Visited
    # land cells are flipped to 0, which doubles as the "visited" marker
    # and avoids a separate visited structure.
    land = [list(row) for row in grid]

    islands = 0
    for r in range(rows):
        for c in range(cols):
            if land[r][c] != 1:
                continue
            # Found an unvisited land cell -> new island; sink it entirely.
            islands += 1
            _flood_fill(land, r, c)
    return islands


def _flood_fill(land: Matrix, start_r: int, start_c: int) -> None:
    """Set every land cell reachable from ``(start_r, start_c)`` to 0."""
    rows, cols = len(land), len(land[0])
    stack = [(start_r, start_c)]
    land[start_r][start_c] = 0
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and land[nr][nc] == 1:
                land[nr][nc] = 0
                stack.append((nr, nc))


def parse_input(lines: Iterable[str]) -> Matrix:
    """Parse the ``M N`` header followed by M rows of N values.

    Raises
    ------
    ValueError
        If the header is missing, dimensions are negative, the number of
        rows/columns does not match the header, or a value is not 0/1.
    """
    tokens = " ".join(lines).split()
    if len(tokens) < 2:
        raise ValueError("expected header 'M N'")
    try:
        m, n = int(tokens[0]), int(tokens[1])
    except ValueError:
        raise ValueError(
            f"header must be two integers 'M N', got {tokens[0]!r} {tokens[1]!r}"
        ) from None
    if m < 0 or n < 0:
        raise ValueError("dimensions must be non-negative")
    values = tokens[2:]
    if len(values) != m * n:
        raise ValueError(f"expected {m * n} matrix values, got {len(values)}")
    grid: Matrix = []
    for i in range(m):
        row = []
        for v in values[i * n:(i + 1) * n]:
            if v not in ("0", "1"):
                raise ValueError(f"matrix values must be 0 or 1, got {v!r}")
            row.append(int(v))
        grid.append(row)
    return grid


def main() -> None:
    grid = parse_input(sys.stdin)
    print(count_islands(grid))


if __name__ == "__main__":
    main()
