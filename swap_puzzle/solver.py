from grid import Grid
from graph import Graph
import utils
import itertools
import copy
from typing import List, Tuple, Callable
import collections
import heapq
import math


FWD = 0
BWD = 1
MAX_LENGTH = 50


class NaiveSolver:
    """
    A class to represent solvers which don't use an heuristic.
    """

    def __init__(self, name="") -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def solve(self, grid):
        """
        Solves the grid and returns the sequence of swaps at the format
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """
        raise NotImplementedError


class HeuristicSolver:
    """
    A class to represent solvers which use an heuristic.
    """

    def __init__(self, heuristic: Callable[[Tuple[int, ...]], float], name=""):
        self.heuristic = heuristic
        self.name = name

    def __repr__(self):
        return self.name

    def solve(self, grid: "Grid") -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Solves the grid and returns the sequence of swaps at the format
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """
        raise NotImplementedError


class GreedySolver(NaiveSolver):

    def solve(self, grid: Grid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
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
                    swaps.append(((i2, j2), (i2, j2 + utils.sign(j - j2))))
                    cpy.swap(*swaps[-1])
                    j2 += utils.sign(j - j2)
                while i2 != i:
                    swaps.append(((i2, j2), (i2 + utils.sign(i - i2), j2)))
                    cpy.swap(*swaps[-1])
                    i2 += utils.sign(i - i2)
                k += 1

        return swaps


class NaiveBFSSolver(NaiveSolver):

    def solve(self, grid: Grid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Constructs the full graph of possible grids and performs a breadth-first search
        to compute an optimal solution. It then returns the sequence of swaps at the format
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """
        src = grid.to_tuple()
        m, n = grid.m, grid.n
        all_swaps = Grid.all_swaps(m, n)
        nodes = [
            tuple([m, n] + [*perm])
            for perm in itertools.permutations(range(1, 1 + m * n))
        ]
        graph = Graph(nodes)
        for node in nodes:
            list_node = list(node)
            for swap in all_swaps:
                utils.make_swap(list_node, swap)
                graph.add_edge(node, tuple(list_node))
                utils.make_swap(list_node, swap)  # Redoing the swap to undo it
        path = graph.bfs(src, tuple([m, n] + [x + 1 for x in range(n * m)]))
        return utils.reconstruct_path(path)


class BFSSolver(NaiveSolver):
    def solve(
        self, grid: Grid, debug: bool = False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Performs a breadth-first search on a graph which is expanded gradually
        to compute an optimal solution. It then returns the sequence of swaps at the format
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """
        m, n = grid.m, grid.n
        nb_nodes = 0
        all_swaps = Grid.all_swaps(m, n)
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
                utils.make_swap(L, swap)
                cpy = tuple(L)
                if cpy not in seen:
                    nb_nodes += 1
                    path = node[1] + [swap]
                    queue.appendleft((cpy, path))
                    seen.add(cpy)
                utils.make_swap(L, swap)
        if debug:
            print(f"{self.name}: {nb_nodes} nodes expanded")
        return path


class OptimizedBFSSolver(NaiveSolver):
    """
    Performs a breadth-first search on a graph starting from the sorted grid and expanding it gradually
    to compute an optimal solution.  When a grid is seen, it checks if its symetric has been seen. If so,
    it can deduce an optimal path. It expands less nodes than the classic version but takes more time.
    It then returns the sequence of swaps at the format [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
    """

    def solve(
        self, grid: Grid, debug: bool = False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        m, n = grid.m, grid.n
        nb_nodes = 1
        all_swaps = Grid.all_swaps(m, n)
        src = grid.to_tuple()
        dst = Grid(m, n).to_tuple()
        seen = dict({dst: []})
        queue = collections.deque([dst])
        # The search starts from the destination
        inv = utils.inverse(src)
        while queue:
            node = queue.pop()
            if node == src:
                return seen[node][::-1]
            L = list(node)
            for swap in all_swaps:
                utils.make_swap(L, swap)
                cpy = tuple(L)
                path = seen[node] + [swap]
                if cpy == src:
                    if debug:
                        print(f"{self.name}: {nb_nodes} nodes expanded")
                    return path[::-1]
                if cpy not in seen:
                    nb_nodes += 1
                    queue.appendleft(cpy)
                    seen[cpy] = path
                sym = utils.compose(inv, cpy)
                # We know a path from dst to cpy
                # Finding a path from cpy to src
                # amounts to finding a path from
                # src^(-1) o cpy to src o src^(-1)
                # i.e. from src^(-1) o cpy to dst
                if sym in seen:
                    if debug:
                        print(f"{self.name}: {nb_nodes} nodes expanded")
                    return (path + seen[sym])[::-1]
                utils.make_swap(L, swap)


class BubbleSortSolver(NaiveSolver):
    """Solves a grid of size 1 x n by performing a bubble sort."""

    def solve(self, grid: Grid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        cpy = copy.deepcopy(grid)
        n = grid.n
        swaps = []
        for i in range(n - 1, 0, -1):
            for j in range(i):
                if cpy.state[0][j + 1] < cpy.state[0][j]:
                    swaps.append(((0, j), (0, j + 1)))
                    cpy.swap((0, j), (0, j + 1))
        return swaps


class AStarSolver(HeuristicSolver):

    def solve(self, grid: Grid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Compute an optimal solution by using the A* algorithm.
        """
        counter = itertools.count()
        open_set, closed_set = dict(), dict()
        # closed_set contains the parent from which we discovered the node
        # open_set contains (heuristic + distance, distance known)
        #                    (f, g)
        src = grid.to_tuple()
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.all_swaps(grid.m, grid.n)
        queue = [(0, -next(counter), src, 0, None)]
        # [(heuristic + distance, index, node, distance, father)]
        # index allows A* to work in a LIFO way across equal cost
        # path
        heapq.heapify(queue)
        while queue:
            _, _, node, node_g, parent = heapq.heappop(queue)
            if node == dst:
                path = [node]
                while parent:
                    path.append(parent)
                    parent = closed_set[parent]
                path.reverse()
                return utils.reconstruct_path(path)
            if node in closed_set:
                continue
            closed_set[node] = parent
            tentative_g = node_g + 1
            list_ = list(node)
            for swap in all_swaps:
                utils.make_swap(list_, swap)
                neighbor = tuple(list_)
                utils.make_swap(list_, swap)
                if neighbor in closed_set:
                    continue
                if neighbor in open_set:
                    move_g, move_h = open_set[neighbor]
                    if move_g <= tentative_g:
                        continue
                else:
                    move_h = self.heuristic(neighbor)
                open_set[neighbor] = tentative_g, move_h
                heapq.heappush(
                    queue,
                    (move_h + tentative_g, -next(counter), neighbor, tentative_g, node),
                )


class MMUCe:
    """
    An  implementation of Meet in the Middle for Unit Costs,
    a bidirectional search algorithm described in
    https://ojs.aaai.org/index.php/SOCS/article/download/18514/18305/22030.
    """

    def __init__(
        self,
        name: str,
        heuristic_f: Callable[[Tuple[int, ...]], float],
        heuristic_b: Callable[[Tuple[int, ...]], float],
        eps: float,
    ) -> None:
        self.name = name
        self.heuristic_f = heuristic_f
        self.heuristic_b = heuristic_b
        self.eps = eps

    def __repr__(self):
        return self.name

    def choose_direction(
        self,
        open_sets: List["set"],
        pr_queues: List[List],
        U: float,
        last_direction: int,
        changed: bool,
    ) -> int:

        pr_min = [pr_queues[FWD][0][0], pr_queues[BWD][0][0]]

        if pr_min[FWD] < pr_min[BWD]:
            return FWD
        if pr_min[FWD] > pr_min[BWD]:
            return BWD
        f_min = [math.inf] * 2
        if U == math.inf:
            for dir in (BWD, FWD):
                for element in sorted(pr_queues[dir]):
                    if element[0] == pr_min[dir]:
                        f_min[dir] = min(element[2], f_min[dir])
                    else:
                        break
            if f_min[FWD] <= f_min[BWD]:
                return FWD
            return BWD
        if changed:
            if len(open_sets[FWD]) <= len(open_sets[BWD]):
                return BWD
            return FWD
        return last_direction

    def solve(self, grid: "Grid") -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        src = grid.to_tuple()
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.all_swaps(grid.m, grid.n)
        counter = itertools.count()
        nodes = [src, dst]
        heuristics = [self.heuristic_f, self.heuristic_b]
        came_froms = [dict(), dict()]
        open_sets, closed_sets, f_queues, g_queues, pr_queues, g_scores = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for dir in (FWD, BWD):
            g_scores.append({nodes[dir]: 0})
            open_sets.append(set([nodes[dir]]))
            closed_sets.append(set())
            pr_queue = [
                (
                    max(self.eps, heuristics[dir](nodes[dir])),
                    -next(counter),
                    heuristics[dir](nodes[dir]),
                    nodes[dir],
                )
            ]
            # (priority, index, f_score, node)
            g_queue = utils.SortedList(MAX_LENGTH, [(0, nodes[dir])])
            f_queue = utils.SortedList(
                MAX_LENGTH, [(heuristics[dir](nodes[dir]), nodes[dir])]
            )
            f_queues.append(f_queue)
            g_queues.append(g_queue)
            pr_queues.append(pr_queue)

        U = math.inf
        last_good_node = None
        changed = False
        pr_min_f, pr_min_b = -1, -1

        while open_sets[FWD] and open_sets[BWD]:
            if pr_min_f != pr_queues[FWD][0][0] or pr_min_b != pr_queues[BWD][0][0]:
                changed = True
            pr_min_f = pr_queues[FWD][0][0]
            pr_min_b = pr_queues[BWD][0][0]
            g_min_f, g_min_b = g_queues[FWD].min(), g_queues[BWD].min()
            f_min_f, f_min_b = f_queues[FWD].min(), f_queues[BWD].min()
            C = min(pr_min_f, pr_min_b)
            if U <= max(C, f_min_f, f_min_b, g_min_f + g_min_b + self.eps):
                path = []
                node = last_good_node
                while node:
                    path.append(node)
                    node = came_froms[FWD].get(node, None)
                path.reverse()
                node = came_froms[BWD].get(last_good_node, None)
                while node:
                    path.append(node)
                    node = came_froms[BWD].get(node, None)
                return utils.reconstruct_path(path)

            dir = self.choose_direction(open_sets, pr_queues, U, dir, changed)
            changed = False
            _, _, _, node = heapq.heappop(pr_queues[dir])
            open_sets[dir].remove(node)
            g_queues[dir].remove(node)
            f_queues[dir].remove(node)
            closed_sets[dir].add(node)
            list_ = list(node)
            for swap in all_swaps:
                utils.make_swap(list_, swap)
                neighbor = tuple(list_)
                utils.make_swap(list_, swap)
                if (
                    neighbor in open_sets[dir] or neighbor in closed_sets[dir]
                ) and g_scores[dir][neighbor] <= g_scores[dir][node] + 1:
                    continue
                if neighbor in closed_sets[dir]:
                    closed_sets[dir].remove(neighbor)
                if neighbor in open_sets[dir]:
                    open_sets[dir].remove(neighbor)
                    g_queues[dir].remove(neighbor)
                    f_queues[dir].remove(neighbor)
                    utils.remove_from_heapq(pr_queues[dir], neighbor)

                g_scores[dir][neighbor] = g_scores[dir][node] + 1
                f_score = g_scores[dir][neighbor] + heuristics[dir](neighbor)

                open_sets[dir].add(neighbor)
                heapq.heappush(
                    pr_queues[dir],
                    (
                        max(2 * g_scores[dir][neighbor] + self.eps, f_score),
                        -next(counter),
                        f_score,
                        neighbor,
                    ),
                )
                f_queues[dir].add(neighbor, f_score)
                g_queues[dir].add(neighbor, g_scores[dir][neighbor])
                came_froms[dir][neighbor] = node

                if neighbor in open_sets[1 - dir]:
                    if U > g_scores[dir][neighbor] + g_scores[1 - dir][neighbor]:
                        U = g_scores[dir][neighbor] + g_scores[1 - dir][neighbor]
                        last_good_node = neighbor
                        changed = True
