from grid import Grid
from graph import Graph
import utils
import itertools
import copy
from typing import List, Tuple, Callable
import collections
import heapq
import math


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

    def __init__(self, heuristic, name=""):
        self.heuristic = heuristic
        self.name = name

    def __repr__(self):
        return self.name

    def solve(self, grid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
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
    def solve(self, grid: Grid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Performs a breadth-first search on a graph which is expanded gradually
        to compute an optimal solution. It then returns the sequence of swaps at the format
        [((i1, j1), (i2, j2)), ((i1', j1'), (i2', j2')), ...].
        """
        m, n = grid.m, grid.n
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
                    path = node[1] + [swap]
                    queue.appendleft((cpy, path))
                    seen.add(cpy)
                utils.make_swap(L, swap)
        return node[1]


class AStarSolver(HeuristicSolver):

    def solve(self, grid: Grid) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
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
    An incomplete implementation of Meet in the Middle for Unit Costs,
    a bidirectional search algorithm described in
    https://ojs.aaai.org/index.php/SOCS/article/download/18514/18305/22030.
    """

    def __init__(self, name, heuristic_f, heuristic_b, eps):
        self.name = name
        self.heuristic_f = heuristic_f
        self.heuristic_b = heuristic_b
        self.eps = eps

    def __repr__(self):
        return self.name

    def choose_direction(self, sets, U, changed, last_direction):
        raise NotImplementedError
        """min_node_f = sets[0]["queue"][0]
        min_node_b = sets[1]["queue"][0]
        pr_min_f = min_node_f[0]
        pr_min_b = min_node_b[0][0]
        if pr_min_f < pr_min_b:
            return 0
        if pr_min_f > pr_min_b:
            return 1
        if U == math.inf:
            if min_node_f[3] <= min_node_b[3]:
                return 0
            return 1
        if changed:
            if len(sets[0]["open_set"]) < len(sets[1]["open_set"]):
                return 1
            return 0
        return last_direction"""

    def solve(self, grid):
        raise NotImplementedError
        """src = grid.to_tuple()
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.all_swaps(grid.m, grid.n)
        counter = itertools.count()
        open_set_f, closed_set_f = {src: ()}, set()
        open_set_b, closed_set_b = {dst: ()}, set()
        queue_f = [(max(self.eps, self.heuristic_f(src)), -next(counter), src, 0, None)]
        queue_b = [(max(self.eps, self.heuristic_b(src)), -next(counter), dst, 0, None)]
        heapq.heapify(queue_f)
        heapq.heapify(queue_b)
        sets = [{"open_set": open_set_f, "closed_set": closed_set_f, "g_score": g_score_f, "heuristic": self.heuristic_f, "queue": queue_f},
                {"open_set": open_set_b, "closed_set": closed_set_b, "g_score": g_score_b, "heuristic": self.heuristic_b, "queue": queue_f}]
        
        U = math.inf
        last_direction = 0
        last_good_node = None
        f_min_f, f_min_b = self.heuristic_f(src), self.heuristic_b(src)
        g_min_f, g_min_b = 0, 0

        while open_set_f and open_set_b:
            min_node_f = queue_f[0]
            min_node_b = queue_b[0]
            pr_min_f = min_node_f[0]
            pr_min_b = min_node_b[0]
            C = min(pr_min_f, pr_min_b)
            if U <= max(C, f_min_f, f_min_b, g_min_f + g_min_b + self.eps):
                path = [last_good_node]
                node = closed_set_f[last_good_node]
                while node:
                    path.append(node)
                    node = closed_set_f[node]
                path.reverse()
                node = closed_set_b[last_good_node]
                while node:
                    path.append(node)
                    node = closed_set_b[node]
            direction = self.choose_direction(sets, last_direction)
            last_direction = direction
            
            _, _, node, node_g, parent = heapq.heappop(sets[direction]["queue"])
            if node in sets[direction]["closed_set"]:
                continue
            sets[direction]["closed_set"][node] = parent
            sets[direction]["open_set"].remove(node)
            tentative_g = node_g + 1
            list_ = list(node)
            for swap in all_swaps:
                utils.make_swap(list_, swap)
                neighbor = tuple(list_)
                utils.make_swap(list_, swap)
                if neighbor in sets[direction]["closed_set"] or neighbor in sets[direction]["open_set"] and \
                    open_set[]:
                    continue
                if neighbor in sets[direction]["open_set"]:
                    move_g, move_h = sets[direction]["open_set"][neighbor]
                    if move_g <= tentative_g:
                        continue
                else:
                    move_h = sets[direction]["heuristic"](neighbor)
                sets[direction]["open_set"][neighbor] = tentative_g, move_h
                heapq.heappush(
                    sets[direction]["queue"],
                    (max(2*tentative_g + self.eps, tentative_g + move_h), -next(counter), neighbor, tentative_g, node),
                )
                if neighbor in sets[1-direction]["open_set"]:
                    last_good_node = neighbor
                    U = min(U, open_set_b[neighbor][1] + open_set_f[1])"""
