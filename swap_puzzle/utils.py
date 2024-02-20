from typing import List, Tuple, Callable
from grid import Grid
import heapq
import bisect
import random
import math


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
    if x == 0:
        return 0
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
        A list of grids as tuple representations. The function assumes that each
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


def general_halved_manhattan_distance(target: Tuple[int, ...]):
    """
    Returns a function which computes the halved Manhattan distance
    from a target grid.
    """
    m, n = target[0], target[1]

    def aux(grid):
        sum_manhattan = 0
        for i in range(m):
            for j in range(n):
                sum_manhattan += abs(
                    (grid[i * n + j + 2] - target[i * n + j + 2]) // n
                ) + abs((grid[i * n + j + 2] - target[i * n + j + 2]) % n)
        return sum_manhattan / 2

    return aux


def halved_manhattan_distance(grid: Tuple[int, ...]) -> float:
    """
    Compute the half sum of the Manhattan distances (also known as L1 distance on R^2)
    between the cell occupied by each number in a given grid and the cell it
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


def generate_grid(m: int, n: int, a: float, b: float) -> "Grid":
    """
    Generate randomly a grid whose difficulty is contained in a given interval.

    Parameters:
    -----------
    m, n: int
        Number of lines and columns of the grid wanted.
    a, b: float
        Minmal and maximal percentage of difficulty wanted.

    Output:
    -------
    grid: Grid
        A grid whose difficulty is contained in a given interval.
    """
    reversed_grid = [m, n] + [*range(m * n, 0, -1)]
    max_distance = halved_manhattan_distance(reversed_grid)
    # The reversed grid is the furthest grid from the sorted grid.

    grid = Grid(m, n)
    mini, maxi = math.ceil(a * max_distance), math.floor(
        b * max_distance
    )  # interval of distance
    k = random.randint(mini, maxi)  # level of difficulty within this interval

    swaps = Grid.all_swaps(m, n)
    while halved_manhattan_distance(grid.to_tuple()) != k:
        grid.swap(*random.choice(swaps))

    return grid


class SortedList:
    """
    A class which represents a bounded list which remains sorted after addition and deletion.
    Its elements are (value, node) where value is a comparable item (in our case a float or integer)
    and node an item (in our case a tuple).
    """

    def __init__(self, max_length, elements):
        self.max_length = max_length
        self.elements = [*sorted(elements)][:max_length]

    def remove(self, node):
        """
        Removes the first element of the form (x, node) from the list.
        """
        i = 0
        for element in self.elements:
            if element[-1] == node:
                break
            i += 1
        if i < len(self.elements):
            self.elements.pop(i)

    def min(self):
        """
        Returns the minimum of the list.
        """
        return self.elements[0][0]

    def add(self, node, value):
        """
        Adds (value, node) to the list and keeps it sorted.
        """
        if self.elements and (value > self.elements[-1][0]):
            return
        if len(self.elements) == self.max_length:
            self.elements.pop()
        bisect.insort(self.elements, (value, node))


def remove_from_heapq(heap, val):
    """
    Remove an element from an heapq.
    """
    for i, el in enumerate(heap):
        if heap[-1] == val:
            break
    if i < len(heap):
        heap.pop(i)
        heapq.heapify(heap)

class BubbleSortSolver(NaiveSolver):
   def solve(self, grid: Grid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    "Implementing a solver using the bubble sort, derived from"
    "https://en.wikipedia.org/wiki/Bubble_sort"

    m, n = grid.m, grid.n
    l = m*n
    swapped = False
    for i in range(l):
        if grid[i+1] > grid(i+2):
            Grid.swap(grid[i+1], grid[i+2])
    


def full_linear_conflict(grid: "Grid") -> float:
    """
    An attempt to mimic the Linear Conflict heuristic used in the
    (n^2-1)-puzzle. See https://mice.cs.columbia.edu/getTechreport.php?techreportID=1026&format=pdf&
    page 15 for the pseudo-code for the original implementation.
    The idea of this function comes from the fact that the manhattan distance isn't an admissible heuristic,
    we have to halve it because somes moves make two different tiles closer.
    It seems to not be admissible.
    """
    m, n = grid[0], grid[1]
    lc = 0
    # Horizontally
    for i in range(m):
        C = [0] * n
        for j in range(n):
            line_dest_j, col_dest_j = (grid[n * i + j + 2] - 1) % n, (
                grid[n * i + j + 2] - 1
            ) // n
            for k in range(j):
                line_dest_k, col_dest_k = (grid[n * i + k + 2] - 1) % n, (
                    grid[n * i + k + 2] - 1
                ) // n
                if line_dest_j == line_dest_k and col_dest_j > col_dest_k:
                    # In the original implementation the condition was col_dest_k > col_dest_j
                    # In other words, it computes the number of inversion
                    # In the (n^2-1) puzzle, an inversion cost more moves but in the swap puzzle
                    # it's the opposite. The non-inversion are actually less preferable in this game.
                    C[j] += 1
        while max(C) > 0:
            C[C.index(max(C))] = 0
            lc += 1
            for k in range(n):
                if C[k]:
                    C[k] -= 1

    for j in range(n):
        C = [0] * m
        for i in range(m):
            line_dest_i, col_dest_i = (grid[n * i + j + 2] - 1) % n, (
                grid[n * i + j + 2] - 1
            ) // n
            for k in range(i):
                line_dest_k, col_dest_k = (grid[n * k + j + 2] - 1) % n, (
                    grid[n * k + j + 2] - 1
                ) // n
                if col_dest_i == col_dest_k and line_dest_i > line_dest_k:
                    C[i] += 1

        while max(C) > 0:
            C[C.index(max(C))] = 0
            lc += 1
            for k in range(m):
                if C[k]:
                    C[k] -= 1
        return lc + halved_manhattan_distance(grid)


def compose(grid1: Tuple[int, ...], grid2: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Computes the composition of two grids by considering them as permutations.
    """
    grid3 = [0] * len(grid1)
    grid3[0], grid3[1] = grid1[0], grid1[1]
    for i in range(grid3[0] * grid3[1]):
        grid3[i + 2] = grid1[grid2[i + 2] + 1]
    return tuple(grid3)


def inverse(grid: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Compute the inverse of a grid by considering it as a permutation.
    """
    inv_grid = [0] * len(grid)
    inv_grid[0], inv_grid[1] = grid[0], grid[1]
    for i in range(grid[0] * grid[1]):
        inv_grid[grid[i + 2] + 1] = i
    return tuple(inv_grid)
