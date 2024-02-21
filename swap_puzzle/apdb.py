from grid import Grid
import copy
import itertools
import pickle
import os
import utils
from typing import List, Dict, Tuple, Callable
import collections


def to_group_representation(group: List[int], grid: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Computes the group representation of a grid.

    Parameters:
    -----------
    group: List[int]
        The numbers defining the pattern group.
    grid: Tuple[int, ...]
        The grid.

    Output:
    -------
    tuple: Tuple[int, ...]
        The group representation of the grid. It contains
        the indices where the numbers of the pattern
        groups are.

    """
    return tuple(grid.index(k, 2) - 2 for k in group)


def compute_apdb(m: int, n: int, group: List[int]) -> Dict[Tuple[int, ...], int]:
    """
    Computes additive pattern database (APDB) for the swap puzzle as
    described in https://arxiv.org/pdf/1107.0050.pdf.

    Paramters:
    ---------
    m, n: int
        Number of lines and columns of the grids considered.
    group: List[Tuple[int, int]]
        The numbers defining the pattern group. Only these numbers are considered,
        the other ones are replaced by -1.

    Output:
    -------
    distances: Dict[Tuple[int, ...], int]
        A dictionnary which associates to each pattern group representation
        grid in the pattern group the minimal number of moves to solve it.
    """
    distances = {}
    src = list(Grid(m, n).to_tuple())
    for k in range(2, m * n + 2):
        if src[k] not in group:
            src[k] = -1
    src = tuple(src)
    all_swaps = Grid.all_swaps(m, n)
    distances = {src: 0}
    queue = collections.deque([src])
    while queue:
        node = queue.pop()
        d = distances[node]
        L = list(node)
        for swap in all_swaps:
            i, j = swap[0][0] * n + swap[0][1] + 2, swap[1][0] * n + swap[1][1] + 2
            if L[i] == L[j] == -1:
                continue
            # We don't consider a swap between two -1 tiles
            utils.make_swap(L, swap)
            cpy = tuple(L)
            if cpy not in distances:
                queue.appendleft(cpy)
                distances[cpy] = d + 1
            utils.make_swap(L, swap)
    # The group representation is far more compact
    return {to_group_representation(group, grid): d for grid, d in distances.items()}


def get_filename(m: int, n: int, group: List[int]) -> str:
    """
    Gives a generic filename for storing APDBs.
    """
    return f"pdb\\{m} x {n}, " + " ".join([str(k) for k in group])


def save_apdb(m: int, n: int, group: List[int], filename: str = "") -> None:
    """
    Generates and saves an APDB.
    """
    filename = filename or get_filename(m, n, group)
    pdb = compute_apdb(m, n, group)
    print(len(pdb))
    with open(filename, "wb") as f:
        pickle.dump(pdb, f)


def loads_apdb(filename: str) -> Dict[Tuple[int, ...], int]:
    """
    Loads and returns an APDB.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def apdb_heuristic(
    m: int, n: int, groups: List[List[int]], filenames=[]
) -> Callable[[Tuple[int, ...]], int]:
    """
    Returns a function to compute an heuristic using APDB.
    """
    N = len(groups)
    filenames = filenames or [get_filename(m, n, group) for group in groups]
    apdbs = [loads_apdb(filename) for filename in filenames]

    def heuristic(grid: Tuple[int, ...]) -> int:
        return sum(apdbs[i][to_group_representation(groups[i], grid)] for i in range(N))

    return heuristic


if __name__ == "__main__":
    m, n = 4, 4
    groups = [[1, 2, 3, 5, 6], [9, 10, 13, 14, 15, 16], [4, 7, 8, 11, 12]]
