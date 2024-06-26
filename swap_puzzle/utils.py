from typing import List, Tuple, Callable
from grid import Grid
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


def make_swap(grid: List[int], swap: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
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


def alt_make_swap(grid: List[int], swap: int) -> None:
    """
    Makes a swap on the tuple representation of a grid

    Parameters:
    -----------

    grid: List[int]
        The tuple representation of a grid.
    swap: int
        The swap a the n*i + j format.
    """
    grid[swap[0]], grid[swap[1]] = grid[swap[1]], grid[swap[0]]


def reconstruct_path(
    path: List[Tuple[int, ...]]
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Finds the sequence of swaps needed to go from each grid to the next one.

    Parameters:
    -----------

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


def half_manhattan_distance(grid: Tuple[int, ...]) -> float:
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


def integer_half_manhattan_distance(grid: Tuple[int, ...]) -> int:
    return math.ceil(half_manhattan_distance(grid))


def general_half_manhattan_distance(
    grid2: Tuple[int, ...]
) -> Callable[[Tuple[int, ...]], float]:
    """
    Returns a function which computes the halved Manhattan distance
    from a reference grid.
    """
    m, n = grid2[0], grid2[1]
    inv = inverse(grid2)

    def aux(grid1: Tuple[int, ...]) -> float:
        sum_manhattan = 0
        for i in range(m):
            for j in range(n):
                sum_manhattan += abs(
                    (inv[grid1[i * n + j + 2] + 1] - 1) // n - i
                ) + abs((inv[grid1[i * n + j + 2] + 1] - 1) % n - j)
        return sum_manhattan / 2

    return aux


def alt_general_half_manhattan_distance(
    grid1: Tuple[int, ...], grid2: Tuple[int, ...]
) -> float:
    """
    Computes directly the Manhattan distance between two grids.
    """
    m, n = grid2[0], grid2[1]
    inv = inverse(grid2)
    sum_manhattan = 0
    for i in range(m):
        for j in range(n):
            sum_manhattan += abs((inv[grid1[i * n + j + 2] + 1] - 1) // n - i) + abs(
                (inv[grid1[i * n + j + 2] + 1] - 1) % n - j
            )
    return sum_manhattan / 2


def update(
    grid: Tuple[int, ...], swap: Tuple[Tuple[int, int], Tuple[int, int]]
) -> float:
    """
    Computes the change of the Manhattan distance made by the swap.

    Parameters:
    ----------
        grid: Tuple[int, ...]
        The grid before the swap.

        swap: Tuple[Tuple[int, int], Tuple[int, int]
        The swap to make.

    Output:
    -------
    The variation of the Manhattan distance caused by the swap.
    """
    m, n = grid[0], grid[1]
    if swap[0][0] == swap[1][0]:
        # Déplacement horizontal
        dx_1 = sign((grid[swap[0][0] * n + swap[0][1] + 2] - 1) % n - swap[0][1])
        if dx_1 == 0:
            sign_1 = 1
        else:
            sign_1 = -sign(swap[1][1] - swap[0][1]) * dx_1
        dx_2 = sign((grid[swap[1][0] * n + swap[1][1] + 2] - 1) % n - swap[1][1])
        if dx_2 == 0:
            sign_2 = 1
        else:
            sign_2 = -sign(swap[0][1] - swap[1][1]) * dx_2
    if swap[0][1] == swap[1][1]:
        # Déplacement vertical
        dy_1 = sign((grid[swap[0][0] * n + swap[0][1] + 2] - 1) // n - swap[0][0])
        if dy_1 == 0:
            sign_1 = 1
        else:
            sign_1 = -sign(swap[1][0] - swap[0][0]) * dy_1
        dy_2 = sign((grid[swap[1][0] * n + swap[1][1] + 2] - 1) // n - swap[1][0])
        if dy_2 == 0:
            sign_2 = 1
        else:
            sign_2 = -sign(swap[0][0] - swap[1][0]) * dy_2
    return (sign_1 + sign_2) / 2


def difficulty_bounded_grid(m: int, n: int, a: float, b: float) -> "Grid":
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
    max_distance = half_manhattan_distance(reversed_grid)
    # The reversed grid is the furthest grid from the sorted grid.

    grid = Grid(m, n)
    mini, maxi = math.ceil(a * max_distance), math.floor(
        b * max_distance
    )  # interval of difficuly
    k = random.randint(mini, maxi)  # level of difficulty chosen in the interval

    swaps = Grid.all_swaps(m, n)
    while half_manhattan_distance(grid.to_tuple()) != k:
        grid.swap(*random.choice(swaps))

    return grid


def compose(grid1: Tuple[int, ...], grid2: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Computes the composition of two grids considered as permutations.
    """
    grid3 = [0] * len(grid1)
    grid3[0], grid3[1] = grid1[0], grid1[1]
    for i in range(grid3[0] * grid3[1]):
        grid3[i + 2] = grid1[grid2[i + 2] + 1]
    return tuple(grid3)


def inverse(grid: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Compute the inverse of a grid considered as a permutation.
    """
    inv_grid = [0] * len(grid)
    inv_grid[0], inv_grid[1] = grid[0], grid[1]
    for i in range(grid[0] * grid[1]):
        inv_grid[grid[i + 2] + 1] = i + 1
    return tuple(inv_grid)


def inversion_count(grid: Tuple[int, ...]) -> int:
    """
    Computes the number of inversions of a grid by considering it as a permutation.
    """
    nb_inv = 0
    for i in range(2, len(grid)):
        for j in range(i + 1, len(grid)):
            if grid[i] > grid[j]:
                nb_inv += 1
    return nb_inv


def reflection(grid: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Relfects the positions and the values of the grid along the main diagonal.
    """
    m, n = grid[0], grid[1]
    grid_reflexion = [n, m] + [0] * n * m
    for i in range(n):
        for j in range(m):
            k = n * j + i + 2
            x, y = (grid[k] - 1) % n, (grid[k] - 1) // n
            grid_reflexion[m * i + j + 2] = m * x + y + 1
    return grid_reflexion


def inversion_distance(grid: Tuple[int, ...]) -> int:
    m, n = grid[0], grid[1]
    inv_h = inversion_count(grid)
    inv_v = inversion_count(reflection(grid))
    return math.ceil((inv_h + inv_v) / (max(n, m) * 2))


def half_sum(x):
    return math.ceil(sum(x) / 2)
