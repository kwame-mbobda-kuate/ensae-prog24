from grid import Grid
import random
from graph import Graph
import itertools
import copy
from typing import List, Tuple, Callable
import collections
import heapq
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
    return -1


def naive_solver(grid: Grid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Swaps cells in order to put, in increasing order, each number at its place.
    It then returns the sequence of swaps at the format
    [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
    """
    cpy = copy.deepcopy(grid)
    swaps = []
    k = 1  # The number about to be placed in the right place.
    for i in range(grid.m):
        for j in range(grid.n):
            # Searches the grid for k
            for i2 in range(grid.m):
                for j2 in range(grid.n):
                    if cpy.state[i2][j2] == k:
                        break
                if cpy.state[i2][j2] == k:
                    break
            # Swaps are done horizontally then vertically.
            while j2 != j:
                swaps.append(((i2, j2), (i2, j2 + sign(j - j2))))
                cpy.swap(*swaps[-1])
                j2 += sign(j - j2)
            while i2 != i:
                swaps.append(((i2, j2), (i2 + sign(i - i2), j2)))
                cpy.swap(*swaps[-1])
                i2 += sign(i - i2)
            k += 1

    return swaps


def make_swap(grid: Tuple[int, ...], swap: Tuple[int, int]) -> None:
    """
    Makes a swap on the tuple representation of a grid

    Parameters:
    -----------

    grid: Tuple[int, ...]
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


def naive_bfs_solver(grid: Grid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Constructs the full graph of possible grids and performs a breadth-first search
    to compute an optimal solution. It then returns the sequence of swaps at the format
    [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
    """
    src = grid.to_tuple()
    m, n = grid.m, grid.n
    all_swaps = grid.all_swaps()
    nodes = [
        tuple([m, n] + [*perm]) for perm in itertools.permutations(range(1, 1 + m * n))
    ]
    graph = Graph(nodes)
    for node in nodes:
        list_node = list(node)
        for swap in all_swaps:
            make_swap(list_node, swap)
            graph.add_edge(node, tuple(list_node))
            make_swap(list_node, swap)  # Redoing the swap to undo it
    path = graph.bfs(src, tuple([m, n] + [x + 1 for x in range(n * m)]))
    return reconstruct_path(path)


def bfs_solver(grid: Grid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
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
    # The element of the queue are of the form (Tuple[int, ...], List[Tuple[Tuple[int, int], Tuple[int, int]]])
    # where the first element is tuple representation of a grid and the second the swaps needed to obtain it
    # starting from src.
    node = None
    while queue and (node is None or node[0] != dst):
        node = queue.pop()
        L = list(node[0])
        for swap in all_swaps:
            make_swap(L, swap)
            cpy = tuple(L)
            if cpy not in seen:
                path = node[1] + [swap]
                queue.appendleft((cpy, path))
                seen.add(cpy)
            make_swap(L, swap)
    return node[1]


class ComparableTuple(tuple):
    """
    Tuple comparaisons are, by default, done by a lexicographic comparison which is not suited here.
    The comparisons are here done using a custom function.
    """

    def __new__(cls, cmp, *args):
        return super(ComparableTuple, cls).__new__(cls, *args)

    def __init__(self, cmp, *args):
        self.cmp = cmp

    def __lt__(self, other):
        return self.cmp(self, other)


def a_star_solver(
    grid: Grid, heuristic: Callable[[Tuple[int, ...]], float]
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Compute an optimal solution by using the A* algorithm.
    Implementation derived from https://en.wikipedia.org/wiki/A*_search_algorithm.

    Parameters:
    -----------
    grid: Grid
        The grid to be solved.
    heuristic: Callable[[Tuple[int, ...]], float]
        An heuristic function, which takes a tuple representation of a grid
        and retuns a float.

    Ouput:
    ------
    swaps: List[Tuple[Tuple[int, int], Tuple[int, int]]]
        The sequence of swaps at the format [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
    """
    counter = itertools.count()
    open_set, closed_set = dict(), dict()
    src = grid.to_tuple()
    dst = Grid(grid.m, grid.n).to_tuple()
    all_swaps = grid.all_swaps()
    queue = [(0, -next(counter), src, 0, None)]
    heapq.heapify(queue)
    while queue:
        _, _, node, node_g, parent = heapq.heappop(queue)
        if node == dst:
            path = [node]
            while parent:
                path.append(parent)
                parent = closed_set[parent]
            path.reverse()
            return reconstruct_path(path)
        if node in closed_set:
            continue
        closed_set[node] = parent
        tentative_g = node_g + 1
        list_ = list(node)
        for swap in all_swaps:
            make_swap(list_, swap)
            neighbor = tuple(list_)
            make_swap(list_, swap)
            if neighbor in closed_set:
                continue
            if neighbor in open_set:
                move_g, move_h = open_set[neighbor]
                if move_g <= tentative_g:
                    continue
            else:
                move_h = heuristic(neighbor)
            open_set[neighbor] = tentative_g, move_h
            heapq.heappush(
                queue,
                (move_h + tentative_g, -next(counter), neighbor, tentative_g, node),
            )


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


def manhattan_a_star_solver(grid: Grid, weight: float = 1):
    """
    Compute an optimal solution by using the A* algorithm and
    the weighted Manhattan distance as heuristic.

    Parameters:
    -----------
    grid: Grid
        The grid to solve.
    weight: float
        A real number larger than 1.
        The algorithm is optimal for a weight of 1 but increasing
        it speeds up the algorithm at the cost of the quality of
        the solution.
    """
    return a_star_solver(grid, lambda g: weight * halved_manhattan_distance(g))
