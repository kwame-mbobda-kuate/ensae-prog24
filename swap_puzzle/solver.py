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
FOUND = 0
BWD = 1
MAX_LENGTH = 1000


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
        all_swaps = Grid.all_swaps(m, n)
        src = grid.to_tuple()
        dst = Grid(m, n).to_tuple()
        explored = {src: (None, None)}
        queue = collections.deque([src])
        nb_nodes = 1
        # The element of the queue are of the form (Tuple[int, ...], List[Tuple[Tuple[int, int], Tuple[int, int]]])
        # where the first element is tuple representation of a grid and the second the swaps needed to obtain it
        # starting from src.
        while queue:
            node = queue.pop()
            if node == dst:
                path = []
                while explored[node][0]:
                    path.append(explored[node][1])
                    node = explored[node][0]
                path.reverse()
                if debug:
                    return {"path": path, "nb_nodes": nb_nodes}
                return path
            L = list(node)
            for swap in all_swaps:
                utils.make_swap(L, swap)
                cpy = tuple(L)
                if cpy not in explored:
                    nb_nodes += 1
                    queue.appendleft(cpy)
                    explored[cpy] = (node, swap)
                utils.make_swap(L, swap)


class BidirectionnalBFSSolver(NaiveSolver):
    """
    Solves the swap puzzle by performing a bidirectionnal BFS i.e
    a BFS from the source and another from the destination.
    """
    def solve(
        self, grid: Grid, debug: bool = False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:

        m, n = grid.m, grid.n
        all_swaps = Grid.all_swaps(m, n)
        src = grid.to_tuple()
        dst = Grid(m, n).to_tuple()
        nb_nodes = 0
        explored = [{src: (None, None)}, {dst: (None, None)}]
        queues = [collections.deque([src]), collections.deque([dst])]
        dir = FWD
        while queues[dir]:
            node = queues[dir].pop()
            if node in explored[1 - dir]:
                cpy = node
                path = []
                while explored[FWD][node][0]:
                    path.append(explored[FWD][node][1])
                    node = explored[FWD][node][0]
                path.reverse()
                node = cpy
                while explored[BWD][node][0]:
                    path.append(explored[BWD][node][1])
                    node = explored[BWD][node][0]
                if debug:
                    return {"path": path, "nb_nodes": nb_nodes}
                return path
            L = list(node)
            for swap in all_swaps:
                utils.make_swap(L, swap)
                cpy = tuple(L)
                if cpy not in explored[dir]:
                    nb_nodes += 1
                    queues[dir].appendleft(cpy)
                    explored[dir][cpy] = (node, swap)
                utils.make_swap(L, swap)
            dir = 1 - dir


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
                neighbor = tuple(L)
                utils.make_swap(L, swap)
                path = seen[node] + [swap]
                if neighbor == src:
                    if debug:
                        print(f"{self.name}: {nb_nodes} nodes expanded")
                    return path[::-1]
                if neighbor not in seen:
                    nb_nodes += 1
                    queue.appendleft(neighbor)
                    seen[neighbor] = path
                sym = utils.compose(inv, neighbor)
                # We know a path from dst to cpy
                # Finding a path from cpy to src
                # amounts to finding a path from
                # src^(-1) o cpy to src o src^(-1)
                # i.e. from src^(-1) o cpy to dst
                if sym in seen:
                    if debug:
                        print(f"{self.name}: {nb_nodes} nodes expanded")
                    return (path + seen[sym])[::-1]


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

    def solve(
        self, grid: Grid, debug=False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Compute an optimal solution by using the A* algorithm.
        """
        nb_nodes = 0
        src = grid.to_tuple()
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.all_swaps(grid.m, grid.n)
        open_set, closed_set = {src: (0, self.heuristic(src))}, {src: None}
        # closed_set contains the parent from which we discovered the node
        # open_set contains (distance known, heuristic + distance)
        #                    (g, f)
        queue = [(0, src)]
        # [(heuristic + distance, index, node)]
        # index allows A* to work in a LIFO way across equal cost
        # paths
        heapq.heapify(queue)
        while queue:
            _, node = heapq.heappop(queue)
            if node == dst:
                path = []
                while node:
                    path.append(node)
                    node = closed_set[node]
                path.reverse()
                path = utils.reconstruct_path(path)
                if debug:
                    return {"nb_nodes": nb_nodes, "path": path}
                return path
            tentative_g = open_set[node][0] + 1
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
                    (move_h + tentative_g, neighbor),
                )
                nb_nodes += 1
                closed_set[neighbor] = node


class MDAStarSolver:
    """
    An A* solver optimized for Mahnattan Distance.
    """
    def __init__(self, name="", dbs={}):
        self.dbs = dbs
        self.name = name

    def __repr__(self):
        return self.name

    def compute(self, m: int, n: int):
        db = {}
        g = list(Grid(m, n).to_tuple())
        for swap in Grid.all_swaps(m, n):
            s1 = swap[0][0] * n + swap[0][1] + 2
            s2 = swap[1][0] * n + swap[1][1] + 2
            for i in range(1, m * n + 1):
                for j in range(1, m * n + 1):
                    g[s1] = i
                    g[s2] = j
                    db[(s1, s2, i, j)] = utils.update(g, swap)
        self.dbs[(m, n)] = db

    def solve(
        self, grid: Grid, debug=False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Compute an optimal solution by using the A* algorithm.
        """
        nb_nodes = 0
        m, n = grid.m, grid.n
        if (m, n) not in self.dbs:
            self.compute(m, n)
        db = self.dbs[(m, n)]
        src = grid.to_tuple()
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.alt_all_swaps(grid.m, grid.n)
        open_set, closed_set = {src: (0, 0)}, {src: None}
        # closed_set contains the parent from which we discovered the node
        # open_set contains (distance known, heuristic + distance)
        #                    (g, f)
        queue = [(0, src)]
        # [(heuristic + distance, index, node)]
        # index allows A* to work in a LIFO way across equal cost
        # paths
        heapq.heapify(queue)
        while queue:
            _, node = heapq.heappop(queue)
            if node == dst:
                path = []
                while node:
                    path.append(node)
                    node = closed_set[node]
                path.reverse()
                path = utils.reconstruct_path(path)
                if debug:
                    return {"nb_nodes": nb_nodes, "path": path}
                return path
            tentative_g, tentative_h = open_set[node]
            tentative_g += 1
            list_ = list(node)
            for swap in all_swaps:
                list_[swap[0]], list_[swap[1]] = list_[swap[1]], list_[swap[0]]
                neighbor = tuple(list_)
                list_[swap[0]], list_[swap[1]] = list_[swap[1]], list_[swap[0]]
                if neighbor in closed_set:
                    continue
                if neighbor in open_set:
                    move_g, move_h = open_set[neighbor]
                    if move_g <= tentative_g:
                        continue
                else:
                    move_h = (
                        tentative_h
                        + db[(swap[0], swap[1], node[swap[0]], node[swap[1]])]
                    )
                    # move_h = self.heuristic(neighbor)
                open_set[neighbor] = tentative_g, move_h
                heapq.heappush(
                    queue,
                    (move_h + tentative_g, neighbor),
                )
                nb_nodes += 1
                closed_set[neighbor] = node


class PerimeterAStarSolver(HeuristicSolver):

    def __init__(
        self,
        heuristic: Callable[[Tuple[int, ...], Tuple[int, ...]], float],
        d: int = 0,
        name: str = "",
    ):
        self.heuristic = heuristic
        self.d = d
        self.name = name
        self.p_d = set()
        self.a_d = {}

    def heuristic_d(self, grid):
        return min(self.heuristic(grid, node) for node in self.p_d) + self.d

    def solve(
        self, grid: Grid, debug=False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        nb_nodes = 0
        src = grid.to_tuple()
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.all_swaps(grid.m, grid.n)
        queue = collections.deque([dst])
        self.a_d = {dst: []}
        self.p_d = set()
        while queue:
            node = queue.pop()
            path = self.a_d[node]
            if len(path) == self.d:
                self.p_d.add(node)
            else:
                list_node = list(node)
                for swap in all_swaps:
                    utils.make_swap(list_node, swap)
                    neighbor = tuple(list_node)
                    utils.make_swap(list_node, swap)
                    if neighbor not in self.a_d:
                        self.a_d[neighbor] = path + [swap]
                        queue.appendleft(neighbor)
                        nb_nodes += 1
        if src in self.a_d:
            if debug:
                return {"nb_nodes": nb_nodes, "path": self.a_d[src]}
            return self.a_d[src]
        open_set, closed_set = {src: (0, self.heuristic_d(src))}, {src: None}
        queue = [(0, src)]
        heapq.heapify(queue)

        while queue:
            _, node = heapq.heappop(queue)
            if node in self.a_d:
                cpy = node
                path = []
                while node:
                    path.append(utils.compose(node, dst))
                    node = closed_set[node]
                path.reverse()
                path = self.a_d[cpy] + utils.reconstruct_path(path)
                if debug:
                    return {"nb_nodes": nb_nodes, "path": path}
                return path
            tentative_g = open_set[node][0] + 1
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
                    move_h = self.heuristic_d(neighbor)
                open_set[neighbor] = tentative_g, move_h
                heapq.heappush(
                    queue,
                    (move_h + tentative_g, neighbor),
                )
                nb_nodes += 1
                closed_set[neighbor] = node
