from typing import List, Tuple, Callable
from grid import Grid
import heapq
import bisect
import random


def sign(x: float) -> int:
    """
    Computes the sign of a float.

    Parameter:
    -----------
    x: float
        The float.

    Output:
    -------
    sign: int
        The sign of the float.
    """
    if x > 0:
        return 1
    return -1


def make_swap(grid: List[int], swap: Tuple[int, int]) -> None:
    """
    Makes a swap on the tuple representation of a grid

    Parameters:
    -----------

    grid: List[int]
        The tuple representation of a grid.
    swap: Tuple[int, int]
        The swap a the (i, j) format.
    """
    i, j = swap[0][0] * grid[1] + swap[0][1] + 2, swap[1][0] * grid[1] + swap[1][1] + 2
    # The "+ 2" accounts for the first two elements being the numbers of lines and columns.
    grid[i], grid[j] = grid[j], grid[i]


def reconstruct_path(
    path: List[Tuple[int, ...]]
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Finds the sequence of swaps needed to go from each grid to the next one.

    Parameter:
    ----------
    path: List[Tuple[int, ...]]
        A list of grids as tuple representations. The function assumes that for each
        grid of the list can be obtained by making an allowed swap on the previous one.

    Output:
    -------
    swaps: List[Tuple[Tuple[int, int], Tuple[int, int]]]
        The sequence of swaps at the format [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
    """
    if not path:
        return []
    m, n = path[0][0], path[0][1]
    swaps = []
    for i in range(len(path) - 1):
        j1 = 0
        # Looks for the first element where the two grids differ.
        while path[i][j1 + 2] == path[i + 1][j1 + 2]:
            j1 += 1
        # Looks for the second element involved in the swap.
        if j1 > 0 and path[i][j1 + 1] != path[i + 1][j1 + 1]:
            j2 = j1 - 1
        elif j1 < n * m - 1 and path[i][j1 + 3] != path[i + 1][j1 + 3]:
            j2 = j1 + 1
        elif j1 < n * m - n and path[i][j1 + 2 + n] != path[i + 1][j1 + 2 + n]:
            j2 = j1 + n
        else:
            j2 = j1 - n
        # Conversion to the standard format (i, j).
        swaps.append(((j1 // n, j1 % n), (j2 // n, j2 % n)))
    return swaps


def halved_manhattan_distance(grid: Tuple[int, ...]) -> float:
    """
    Compute the half sum of the Manhattan distances (also known as L1 distance on R^2)
    between each the cell occupied by each number in a given grid and the cell it
    should occupy in the sorted grid. It can be shown that this function is a
    monotone (and thus admissible) heuristic.

    Parameter:
    ----------
    grid: Tuple[int, ...]
        The tuple representation of a grid.

    Output:
    -------
    sum_manhattan: float
        The half sum of the manhattan distances.
    """
    sum_manhattan = 0
    m, n = grid[0], grid[1]
    for i in range(m):
        for j in range(n):
            # grid[i * n + j + 2] is at
            # ((grid[i * n + j + 2] - 1) // n,
            #  (grid[i * n + j + 2] - 1) % n) in the sorted grid.
            sum_manhattan += abs((grid[i * n + j + 2] - 1) // n - i) + abs(
                (grid[i * n + j + 2] - 1) % n - j
            )
    return sum_manhattan / 2


class SortedList:

    def __init__(self, max_length, elements):
        self.max_length = max_length
        self.elements = [*sorted(elements)][:max_length]

    def remove(self, node):
        for i, element in enumerate(self.elements):
            if element[-1] == node:
                break
        if i < len(self.elements):
            self.elements.pop(i)

    def min(self):
        return self.elements[0][0]

    def add(self, node, value):
        if value > self.elements[-1][0]:
            return
        if len(self.elements) == self.max_length:
            self.elements.pop()
        bisect.insort(self.elements, (value, node))


def remove_from_heapq(heap, val):
    for i, el in enumerate(heap):
        if heap[-1] == val:
            break
    if i < len(heap):
        heap.pop(i)
        heapq.heapify(heap)
