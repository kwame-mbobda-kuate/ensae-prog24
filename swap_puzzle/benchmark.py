import solver
import time
import grid
import gadb
import utils
from apdb import DictAPDB
from gadb import GADB


def benchmark(solvers, m, n, N):
    """
    Benchmarks a list of solvers on random grids.
    """
    grids = [grid.Grid.random_grid(m, n) for _ in range(N)]
    for solver in solvers:
        t1 = time.perf_counter()
        for grid_ in grids:
            solver.solve(grid_, debug=False)
        print(f"{solver} : {((time.perf_counter() - t1) / N):.4f} seconds")

gadb3 = GADB.default_load(4, 4, 3)
gadb2 = GADB.default_load(4, 4, 2)
h1 = gadb.GADBList([gadb2, gadb3]).get_heuristic()
h2 = utils.half_manhattan_distance
s1 = solver.AStarSolver(h1, "GADB")
s2 = solver.AStarSolver(h2, "MD")
benchmark([s1, s2], 4, 4, 10)