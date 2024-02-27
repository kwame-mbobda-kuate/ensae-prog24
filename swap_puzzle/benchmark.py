import solver
import time
import grid
import gadb
import utils
from apdb import DictAPDB
import apdb
from gadb import GADB


def benchmark(solvers, m, n, N):
    """
    Benchmarks a list of solvers on random grids.
    """
    grids = [grid.Grid.random_grid(m, n) for _ in range(N)]
    for solver in solvers:
        nb_nodes = 0
        t1 = time.perf_counter()
        for grid_ in grids:
            nb_nodes += solver.solve(grid_, debug=True)["nb_nodes"]
        print(f"{solver} : {((time.perf_counter() - t1) / N):.4f} seconds, {nb_nodes / N} expanded")

s1 = solver.BidirectionalBFSSolver("Bidirectional BFS Solver")
s2 = solver.BFSSolver("BFS Solver")
s3 = solver.AStarSolver(utils.half_manhattan_distance, "A* Solver with MD")
s4 = solver.AStarSolver(apdb.APDBList(lambda x: x[0], [DictAPDB.default_load(3, 3, [*range(1, 10)])]).get_heuristic(), "A* Solver with APDB")
benchmark([s1, s2, s3, s4], 3, 3, 10)