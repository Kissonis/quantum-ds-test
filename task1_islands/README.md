# Task 1 — Counting islands

Count the number of islands on an `M x N` map where `1` is land and `0` is ocean.

## Connectivity

Two land cells belong to the same island only if they share an **edge**
(4-connectivity). Diagonal neighbours are *not* connected. This follows directly
from the reference test cases: in the second case

```
0 0 0 1
0 0 1 0
0 1 0 0
```

the three ones touch only diagonally and the expected answer is `3`, not `1`.

## Algorithm

Iterative flood fill (DFS with an explicit stack):

1. Scan the matrix row by row.
2. Each time an unvisited land cell is found, increment the counter and
   "sink" the whole island it belongs to by marking every reachable land cell
   as visited.

* **Time:** `O(M * N)` — every cell is touched a constant number of times.
  This is optimal, since any correct algorithm has to read every cell at least once.
* **Memory:** `O(M * N)` worst case for the stack (a map that is all land).
  No recursion is used, so very large maps do not hit Python's recursion limit.
* The input matrix is not modified (the function works on a private copy).

## Usage

No third-party dependencies, Python 3.8+.

```bash
python islands.py < input.txt
```

Input format:

```
M N
row_1
...
row_M
```

Example:

```bash
printf '3 4\n0 0 0 1\n0 0 1 1\n0 1 0 1\n' | python islands.py
# 2
```

The function can also be imported:

```python
from islands import count_islands
count_islands([[0, 1, 0], [0, 0, 0], [0, 1, 1]])  # -> 2
```

## Tests

```bash
pip install pytest
pytest task1_islands/tests -q
```

Covers the three reference cases, edge cases (empty map, all ocean, all land,
single row/column, checkerboard, ring with a hole, 1000x1000 solid map),
input validation (ragged matrix, malformed header, wrong value count, values
other than 0/1) and the CLI.
