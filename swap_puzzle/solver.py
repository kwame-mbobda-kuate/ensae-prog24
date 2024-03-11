from grid import Grid
from graph import Graph
import utils
import itertools
import copy
from typing import List, Tuple, Callable
import collections
from solver_utils import (
    AStarNode,
    OneDimBucketList,
    TwoDimBucketList,
    NoTieBreakAStarNode,
)
import heapq
import math


FWD = 0
FOUND = 0
BWD = 1


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

    def solve(
        self, grid: Grid, debug=False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
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
        path = utils.reconstruct_path(path)
        if debug:
            return {"nb_nodes": len(graph.nodes), "path": path}
        return path


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


class BidirectionalBFSSolver(NaiveSolver):
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


class PseudoBidirectionalSolver(NaiveSolver):
    """
    Performs a breadth-first search on a graph starting from the sorted grid and expanding it gradually
    to compute an optimal solution.  When a grid is seen, it checks if its symetric has been seen. If so,
    it can deduce an optimal path..
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
                    path = (path + seen[sym])[::-1]
                    if debug:
                        return {"nb_nodes": nb_nodes, "path": path}
                    return path


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
    """
    The classic A* solver, implemented with
    binary heap and using tie-breaking based on
    g values between nodes of equal f values.
    """

    def solve(
        self, grid: Grid, debug=False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        nb_nodes = 0
        src = AStarNode(grid.to_tuple(), None, 0, self.heuristic(grid.to_tuple()))
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.alt_all_swaps(grid.m, grid.n)
        open_set, closed_set = {src: 0}, set()
        # open_set maps grids to their g value
        queue = [src]
        heapq.heapify(queue)
        while queue:
            node = heapq.heappop(queue)
            closed_set.add(node)
            if node.grid == dst:
                path = []
                while node:
                    path.append(node.grid)
                    node = node.parent
                path.reverse()
                path = utils.reconstruct_path(path)
                if debug:
                    return {"nb_nodes": nb_nodes, "path": path}
                return path
            list_grid = list(node.grid)
            neighbor_g = node.g + 1
            for swap in all_swaps:
                utils.alt_make_swap(list_grid, swap)
                neighbor_grid = tuple(list_grid)
                neighbor = AStarNode(neighbor_grid, node, neighbor_g, 0)
                utils.alt_make_swap(list_grid, swap)
                if neighbor in closed_set:
                    continue
                if open_set.get(neighbor, math.inf) <= neighbor_g:
                    continue
                neighbor.h = self.heuristic(neighbor_grid)
                neighbor.f = neighbor.g + neighbor.h
                open_set[neighbor] = neighbor_g
                heapq.heappush(queue, neighbor)
                nb_nodes += 1


class BucketAStarSolver(HeuristicSolver):
    """
    An implementation of A* with two dimensional
    bucket list to allow tie-breaking.
    """

    def solve(
        self, grid: Grid, debug=False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        nb_nodes = 0
        src = AStarNode(grid.to_tuple(), None, 0, self.heuristic(grid.to_tuple()))
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.alt_all_swaps(grid.m, grid.n)
        open_set, closed_set = {src: 0}, set()
        queue = TwoDimBucketList()
        queue.add(src)
        while not queue.empty():
            node = queue.pop()
            closed_set.add(node)
            if node.grid == dst:
                path = []
                while node:
                    path.append(node.grid)
                    node = node.parent
                path.reverse()
                path = utils.reconstruct_path(path)
                if debug:
                    return {"nb_nodes": nb_nodes, "path": path}
                return path
            list_grid = list(node.grid)
            neighbor_g = node.g + 1
            for swap in all_swaps:
                utils.alt_make_swap(list_grid, swap)
                neighbor_grid = tuple(list_grid)
                neighbor = AStarNode(neighbor_grid, node, neighbor_g, 0)
                utils.alt_make_swap(list_grid, swap)
                if neighbor in closed_set:
                    continue
                if neighbor not in open_set or neighbor_g < open_set[neighbor]:
                    neighbor.h = self.heuristic(neighbor_grid)
                    neighbor.f = neighbor.g + neighbor.h
                    open_set[neighbor] = neighbor_g
                    queue.add(neighbor)
                    nb_nodes += 1


class MDAStarSolver:
    """
    An A* solver optimized for Mahnattan Distance by
    computing it incrementally.
    """

    def __init__(self, name="", dbs={}):
        self.dbs = dbs
        self.name = name

    def __repr__(self):
        return self.name

    def compute(self, m: int, n: int):
        """
        Precomputes the variations of Manhattan
        distance.
        """
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
        nb_nodes = 0
        m, n = grid.m, grid.n
        if (m, n) not in self.dbs:
            self.compute(m, n)
        db = self.dbs[(m, n)]
        src = AStarNode(
            grid.to_tuple(), None, 0, utils.half_manhattan_distance(grid.to_tuple())
        )
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.alt_all_swaps(grid.m, grid.n)
        open_set, closed_set = {src: 0}, set()
        queue = [src]
        heapq.heapify(queue)
        while queue:
            node = heapq.heappop(queue)
            closed_set.add(node)
            if node.grid == dst:
                path = []
                while node:
                    path.append(node.grid)
                    node = node.parent
                path.reverse()
                path = utils.reconstruct_path(path)
                if debug:
                    return {"nb_nodes": nb_nodes, "path": path}
                return path
            list_grid = list(node.grid)
            neighbor_g = node.g + 1
            for swap in all_swaps:
                utils.alt_make_swap(list_grid, swap)
                neighbor_grid = tuple(list_grid)
                neighbor = AStarNode(neighbor_grid, node, neighbor_g, 0)
                utils.alt_make_swap(list_grid, swap)
                if neighbor in closed_set:
                    continue
                if open_set.get(neighbor, math.inf) <= neighbor_g:
                    continue
                neighbor.h = (
                    node.h
                    + db[(swap[0], swap[1], list_grid[swap[0]], list_grid[swap[1]])]
                )
                neighbor.f = neighbor.g + neighbor.h
                open_set[neighbor] = neighbor_g
                heapq.heappush(queue, neighbor)
                nb_nodes += 1


class PerimeterAStarSolver(HeuristicSolver):
    """
    An algorithm which performs a BFS from the destination then
    runs A* from the source with an improved heurisitc.
    See https://www.sciencedirect.com/science/article/pii/000437029490040X.
    """

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

        nb_nodes = 0
        src = AStarNode(grid.to_tuple(), None, 0, self.heuristic_d(grid.to_tuple()))
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.alt_all_swaps(grid.m, grid.n)
        open_set, closed_set = {src: 0}, set()
        queue = [src]
        heapq.heapify(queue)
        while queue:
            node = heapq.heappop(queue)
            closed_set.add(node)
            if node.grid in self.a_d:
                cpy = node.grid
                path = []
                while node:
                    path.append(node.grid)
                    node = node.parent
                path.reverse()
                path = self.a_d[cpy] + utils.reconstruct_path(path)
                path.reverse()
                if debug:
                    return {"nb_nodes": nb_nodes, "path": path}
                return path
            list_grid = list(node.grid)
            neighbor_g = node.g + 1
            for swap in all_swaps:
                utils.alt_make_swap(list_grid, swap)
                neighbor_grid = tuple(list_grid)
                neighbor = AStarNode(neighbor_grid, node, neighbor_g, 0)
                utils.alt_make_swap(list_grid, swap)
                if neighbor in closed_set:
                    continue
                if open_set.get(neighbor, math.inf) <= neighbor_g:
                    continue
                neighbor.h = self.heuristic_d(neighbor_grid)
                neighbor.f = neighbor.g + neighbor.h
                open_set[neighbor] = neighbor_g
                heapq.heappush(queue, neighbor)
                nb_nodes += 1


class BidirectionalAStarSolver:
    """
    A naive implementation of bidirectionnal A*.
    """

    def __init__(self, get_heurisitic_f, get_heurisitc_b, name=""):
        self.get_heuristic_f = get_heurisitic_f
        self.get_heuristic_b = get_heurisitc_b
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def solve(
        self, grid: Grid, debug=False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Compute an optimal solution by using the A* algorithm.
        """
        nb_nodes = 0
        dir = FWD
        all_swaps = Grid.all_swaps(grid.m, grid.n)
        grid_srcs = [grid.to_tuple(), Grid(grid.m, grid.n).to_tuple()]
        heuristics = [
            self.get_heuristic_f(grid_srcs[BWD]),
            self.get_heuristic_b(grid_srcs[FWD]),
        ]
        srcs = [
            AStarNode(grid_srcs[FWD], None, 0, heuristics[FWD](grid_srcs[FWD])),
            AStarNode(grid_srcs[BWD], None, 0, heuristics[BWD](grid_srcs[BWD])),
        ]
        open_sets = [{srcs[FWD]: 0}, {srcs[BWD]: 0}]
        closed_sets = [set(), set()]
        queues = [[srcs[FWD]], [srcs[BWD]]]
        heapq.heapify(queues[FWD])
        heapq.heapify(queues[BWD])

        while queues[BWD] and queues[FWD]:
            node = heapq.heappop(queues[dir])
            closed_sets[dir].add(node)
            if node == srcs[1 - dir]:
                path = []
                while node:
                    path.append(node.grid)
                    node = node.parent
                if dir == FWD:
                    path.reverse()
                path = utils.reconstruct_path(path)
                if debug:
                    return {"nb_nodes": nb_nodes, "path": path}
                return path
            if node in closed_sets[dir] and node in closed_sets[1 - dir]:
                # We found a node visited by both searches
                nodes = [None, None]
                nodes[dir] = node
                for other_node in closed_sets[1 - dir]:
                    if node == other_node and (
                        not nodes[1 - dir] or other_node.g < nodes[1 - dir].g
                    ):
                        # We take the closest to the srcs[1-dir]
                        nodes[1 - dir] = other_node
                node = nodes[FWD]
                path = []
                while node.parent:
                    path.append(node.parent.grid)
                    node = node.parent
                path.reverse()
                node = nodes[BWD]
                while node:
                    path.append(node.grid)
                    node = node.parent
                path = utils.reconstruct_path(path)
                if debug:
                    return {"path": path, "nb_nodes": nb_nodes}
                return path
            list_grid = list(node.grid)
            neighbor_g = node.g + 1
            for swap in all_swaps:
                utils.make_swap(list_grid, swap)
                neighbor_grid = tuple(list_grid)
                neighbor = AStarNode(neighbor_grid, node, neighbor_g, 0)
                utils.make_swap(list_grid, swap)
                if neighbor in closed_sets[dir]:
                    continue
                if open_sets[dir].get(neighbor, math.inf) <= neighbor_g:
                    continue
                neighbor.h = heuristics[dir](neighbor_grid)
                neighbor.f = neighbor.g + neighbor.h
                open_sets[dir][neighbor] = neighbor_g
                heapq.heappush(queues[dir], neighbor)
                nb_nodes += 1
            dir = 1 - dir


class DIBBSSolver:
    """
    Dynamically improved bounds bidirectional search :
    a bidirectional search algorithm based on A* that
    dynamically improves the bounds during its execution.
    See https://www.sciencedirect.com/science/article/pii/S0004370220301545.
    """

    def __init__(self, get_heurisitic_f, get_heurisitc_b, name=""):
        self.get_heuristic_f = get_heurisitic_f
        self.get_heuristic_b = get_heurisitc_b
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def solve(
        self, grid: Grid, debug=False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        nb_nodes = 0
        dir = FWD
        all_swaps = Grid.alt_all_swaps(grid.m, grid.n)
        grid_srcs = [grid.to_tuple(), Grid(grid.m, grid.n).to_tuple()]
        heuristics = [
            self.get_heuristic_f(grid_srcs[BWD]),
            self.get_heuristic_b(grid_srcs[FWD]),
        ]
        srcs = [
            AStarNode(grid_srcs[FWD], None, 0, heuristics[FWD](grid_srcs[FWD])),
            AStarNode(grid_srcs[BWD], None, 0, heuristics[BWD](grid_srcs[BWD])),
        ]
        open_sets = [
            {srcs[FWD]: [heuristics[FWD](grid_srcs[FWD]), 0]},
            {srcs[BWD]: [heuristics[BWD](grid_srcs[BWD]), 0]},
        ]
        closed_sets = [set(), set()]
        queues = [[srcs[FWD]], [srcs[BWD]]]
        last_good_node = None
        UB = math.inf
        F_min = [0, 0]

        while UB > sum(F_min) / 2:
            node = heapq.heappop(queues[dir])
            closed_sets[dir].add(node)
            list_grid = list(node.grid)
            neighbor_g = node.g + 1
            for swap in all_swaps:
                utils.alt_make_swap(list_grid, swap)
                neighbor_grid = tuple(list_grid)
                neighbor = AStarNode(neighbor_grid, node, neighbor_g, 0)
                utils.alt_make_swap(list_grid, swap)
                if neighbor in closed_sets[dir]:
                    continue
                if neighbor not in open_sets[dir] or open_sets[dir][neighbor][0] > 2 * (
                    node.g + 1
                ) + heuristics[dir](neighbor_grid) - heuristics[1 - dir](neighbor_grid):
                    neighbor.f = (
                        2 * (node.g + 1)
                        + heuristics[dir](neighbor_grid)
                        - heuristics[1 - dir](neighbor_grid)
                    )
                    neighbor.g = neighbor_g
                    open_sets[dir][neighbor] = (neighbor.f, neighbor.g)
                    neighbor.parent = node
                    heapq.heappush(queues[dir], neighbor)
                    nb_nodes += 1
                    if neighbor in open_sets[1 - dir]:
                        if neighbor.g + open_sets[1 - dir][neighbor][1] < UB:
                            UB = neighbor.g + open_sets[1 - dir][neighbor][1]
                            last_good_node = neighbor
            F_min[dir] = queues[dir][0].f
            dir = 1 - dir
        nodes = [None, None]
        nodes[1 - dir] = last_good_node
        for node in open_sets[dir]:
            if node == last_good_node:
                if not nodes[dir] or nodes[dir].g > node.g:
                    nodes[dir] = node
        for node in closed_sets[dir]:
            if node == last_good_node:
                if not nodes[dir] or nodes[dir].g > node.g:
                    nodes[dir] = node
        node = nodes[FWD]
        path = []
        while node.parent:
            path.append(node.parent.grid)
            node = node.parent
        path.reverse()
        node = nodes[BWD]
        while node:
            path.append(node.grid)
            node = node.parent
        path = utils.reconstruct_path(path)
        if debug:
            return {"path": path, "nb_nodes": nb_nodes}
        return path


class NoTieBreakAStarSolver(HeuristicSolver):
    """
    An implementation of A* with a binary heap and
    without tie-breaking.
    """

    def solve(
        self, grid: Grid, debug=False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Compute an optimal solution by using the A* algorithm.
        """
        nb_nodes = 0
        src = NoTieBreakAStarNode(
            grid.to_tuple(), None, 0, self.heuristic(grid.to_tuple())
        )
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.alt_all_swaps(grid.m, grid.n)
        open_set, closed_set = {src: 0}, set()
        # closed_set contains the parent from which we discovered the node
        # open_set contains (distance known, heuristic + distance)
        #                    (g, f)
        queue = [src]
        heapq.heapify(queue)
        while queue:
            node = heapq.heappop(queue)
            closed_set.add(node)
            if node.grid == dst:
                path = []
                while node:
                    path.append(node.grid)
                    node = node.parent
                path.reverse()
                path = utils.reconstruct_path(path)
                if debug:
                    return {"nb_nodes": nb_nodes, "path": path}
                return path
            list_grid = list(node.grid)
            neighbor_g = node.g + 1
            for swap in all_swaps:
                utils.alt_make_swap(list_grid, swap)
                neighbor_grid = tuple(list_grid)
                neighbor = NoTieBreakAStarNode(neighbor_grid, node, neighbor_g, 0)
                utils.alt_make_swap(list_grid, swap)
                if neighbor in closed_set:
                    continue
                if open_set.get(neighbor, math.inf) <= neighbor_g:
                    continue
                neighbor.h = self.heuristic(neighbor_grid)
                neighbor.f = neighbor.g + neighbor.h
                open_set[neighbor] = neighbor_g
                heapq.heappush(queue, neighbor)
                nb_nodes += 1


class NoTieBreakBucketAStarSolver(HeuristicSolver):
    """
    An implementation of A* with a bucket list and without tie breaking.
    """

    def solve(
        self, grid: Grid, debug=False
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Compute an optimal solution by using the A* algorithm.
        """
        nb_nodes = 0
        src = NoTieBreakAStarNode(
            grid.to_tuple(), None, 0, self.heuristic(grid.to_tuple())
        )
        dst = Grid(grid.m, grid.n).to_tuple()
        all_swaps = Grid.alt_all_swaps(grid.m, grid.n)
        open_set, closed_set = {src: 0}, set()
        # closed_set contains the parent from which we discovered the node
        # open_set contains (distance known, heuristic + distance)
        #                    (g, f)
        queue = OneDimBucketList()
        queue.add(src)
        while not queue.empty():
            node = queue.pop()
            closed_set.add(node)
            # open_set.pop(node)
            if node.grid == dst:
                path = []
                while node:
                    path.append(node.grid)
                    node = node.parent
                path.reverse()
                path = utils.reconstruct_path(path)
                if debug:
                    return {"nb_nodes": nb_nodes, "path": path}
                return path
            list_grid = list(node.grid)
            neighbor_g = node.g + 1
            for swap in all_swaps:
                utils.alt_make_swap(list_grid, swap)
                neighbor_grid = tuple(list_grid)
                neighbor = NoTieBreakAStarNode(neighbor_grid, node, neighbor_g, 0)
                utils.alt_make_swap(list_grid, swap)
                if neighbor in closed_set:
                    continue
                if neighbor not in open_set or neighbor_g < open_set[neighbor]:
                    neighbor.h = self.heuristic(neighbor_grid)
                    neighbor.f = neighbor.g + neighbor.h
                    open_set[neighbor] = neighbor_g
                    queue.add(neighbor)
                    nb_nodes += 1
