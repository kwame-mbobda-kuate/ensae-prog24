from grid import Grid
import random
from graph import Graph
import itertools
import copy
from typing import List, Tuple
import collections


def sign(x: float) -> int:
    if x > 0:
        return 1
    return -1


class Solver:
    # La structure de cette classe sera probablement modifiée.
    """
    A solver class, to be implemented.
    """

    def get_solution(self, grid: Grid) -> None:
        """
        Solves the grid and returns the sequence of swaps at the format
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """

        raise NotImplementedError

    def random_solve(self, grid: Grid) -> List[Tuple[int, int], Tuple[int, int]]:
        """
        Swaps randomly cells in the grid until it is solved.
        It then returns the sequence of swaps at the format
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """
        swaps = []
        while not grid.is_sorted():
            i1, j1 = random.randrange(grid.n), random.randrange(grid.m)
            i2, j2 = random.choice(grid.allowed_swaps((i1, j1)))
            swaps.append(((i1, j1), (i2, j2)))
        return swaps

    def naive_solve(self, grid: Grid) -> List[Tuple[int, int], Tuple[int, int]]:
        """
        Swaps cells in order to put, in increasing order, each number at its place.
        It then returns the sequence of swaps at the format
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """
        swaps = []
        k = 1  # The number about to be placed in its right place.
        for i in range(grid.m):
            for j in range(grid.n):
                # Searches the grid for k
                for i1 in range(grid.n):
                    for j1 in range(grid.m):
                        if grid.state[i1][j1] == k:
                            break
                    if grid.state[i1][j1] == k:
                        break
                # Swaps are done horizontally then vertically.
                while j1 != j:
                    swaps.apend((i1, j1), (i1, j1 + sign(j - j1)))
                    j1 += sign(j - j1)
                while i1 != i:
                    swaps.apend((i1, j1), (i1 + sign(i - i1), j1))
                    i1 += sign(i - i1)
        return swaps

    def naive_bfs_solve(self, grid: Grid) -> List[Tuple[int, int], Tuple[int, int]]:
        """
        Constructs the full graph of possible grids and performs a breadth-first search
        to compute an optimal solution. It then returns the sequence of swaps at the format
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """
        src = grid
        m, n = grid.m, grid.n
        nodes = []
        all_swaps = grid.all_swaps()
        for perm in itertools.permutations(range(1, 1 + m * n)):
            nodes.append([m, n] + [*perm])
        graph = Graph(nodes)
        for node in nodes:
            for swap in all_swaps:
                cpy = list(node)
                swap_arr(
                    cpy,
                    swap[0][0] * n + swap[0][1] + 2,
                    swap[1][0] * n + swap[1][1] + 2,
                )
                # The "+2" accounts for the first two elements being the numbers of lines and columns.
                graph.add_edge(node, tuple(cpy))
        path = graph.bfs(src.to_tuple(), tuple([m, n] + [x + 1 for x in range(n * m)]))
        swaps = []
        for i in range(len(path) - 1):
            j1 = 0
            # Looks for the first element where the two tuples differ.
            while path[i][j1 + 2] == path[i + 1][j1 + 2]:
                j1 += 1
            # Looks for the second element involved in the swap.
            if j1 > 0 and path[i][j1 + 1] != path[i + 1][j1 + 1]:
                j2 = j1 - 1
            elif j1 < n * m - 1 and path[i][j1 + 3] != path[i + 1][j1 + 3]:
                j2 = j1 + 1
            elif (
                j1 < n * m - grid.n
                and path[i][j1 + 2 + grid.n] != path[i + 1][j1 + 2 + grid.n]
            ):
                j2 = j1 + grid.n
            else:
                j2 = j1 - grid.n
            # Conversion to the standard format (i, j)
            swaps.append(((j1 // grid.n, j1 % grid.n), (j2 // grid.n, j2 % grid.n)))
        return swaps

    def bfs_solve(self, grid: Grid) -> List[Tuple[int, int], Tuple[int, int]]:
        """
        Performs a breadth-first search on a graph which is expanded gradually
        to compute an optimal solution. It then returns the sequence of swaps at the format
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """
        all_swaps = grid.all_swaps()
        m, n = grid.m, grid.n
        src = grid.to_tuple()
        dst = Grid(m, n).to_tuple()
        seen = set([src])
        queue = collections.deque([(src, [])])
        node = None
        while queue and (node is None or node[0] != dst):
            node = queue.pop()
            L = list(node[0])
            for swap in all_swaps:
                swap_arr(
                    L, swap[0][0] * n + swap[0][1] + 2, swap[1][0] * n + swap[1][1] + 2
                )
                cpy = tuple(L)
                if cpy not in seen:
                    path = node[1] + [swap]
                    queue.appendleft((cpy, path))
                    seen.add(cpy)
                swap_arr(
                    L, swap[0][0] * n + swap[0][1] + 2, swap[1][0] * n + swap[1][1] + 2
                )
        return node[1]


def swap_arr(arr: List, i: int, j: int) -> None:
    """
    Swaps two elements of an array.

    Parameters:
    -----------

    arr: List
        The array.
    i, j: int
        The indices involved in the swap.
    """
    arr[i], arr[j] = arr[j], arr[i]
