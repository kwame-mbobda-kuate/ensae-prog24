from grid import Grid
import copy
import itertools
import utils
from typing import List, Dict, Tuple
import collections


def compute_all_distances(m, n, indices) -> Dict[Tuple[int, ...], int]:
    """
    An attempt to compute pattern database for the swap puzzle as described in https://arxiv.org/pdf/1107.0050.pdf.

    Paramters:
    ---------
    m, n: int
        Number of lines and columns of the grids considered.
    indices: List[Tuple[int, int]]
        The indices defining the pattern group. Only the numbers in the grids corresponding
        to indices in this list are considered, the other ones are replaced by -1.

    Output:
    -------
    distances: Dict[Tuple[int, ...], int]
        A dictionnary which associates to each grid in the pattern group the
        minimal number of moves to solve it.
    """
    distances = {}
    for perm in itertools.permutations(range(1, m * n + 1), len(indices)):
        tup_perm = tuple(perm)
        src_state = [[-1] * n for _ in range(m)]
        for i, k in enumerate(perm):
            src_state[indices[i][0]][indices[i][1]] = k
        src = Grid(m, n, src_state).to_tuple()
        dst_state = [list(range(i * n + 1, (i + 1) * n + 1)) for i in range(m)]
        for i in range(m):
            for j in range(m):
                if i * n + j + 1 not in perm:
                    dst_state[i][j] = -1
        dst = Grid(m, n, dst_state).to_tuple()
        all_swaps = Grid.all_swaps(m, n)
        d = 0
        seen = set()
        queue = collections.deque([src])
        node = None
        while queue and (node != dst):
            node = queue.pop()
            d += 1
            L = list(node)
            for swap in all_swaps:
                i, j = swap[0][0] * n + swap[0][1] + 2, swap[1][0] * n + swap[1][1] + 2
                if L[i] == L[j] == -1:
                    continue
                utils.make_swap(L, swap)
                cpy = tuple(L)
                if cpy not in seen:
                    queue.appendleft(cpy)
                    seen.add(cpy)
                utils.make_swap(L, swap)
        distances[tup_perm] = d
    return distances
