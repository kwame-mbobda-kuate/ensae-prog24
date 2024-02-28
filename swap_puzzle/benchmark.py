import solver
import time
import grid
import gadb
import utils
from apdb import ArrayAPDB
import apdb
from gadb import GADB


def solving_time(solvers, m, n, N):
    """
    Benchmarks a list of solvers on random grids.
    """
    grids = [grid.Grid.random_grid(m, n) for _ in range(N)]
    for solver in solvers:
        nb_nodes = 0
        t1 = time.perf_counter()
        for grid_ in grids:
            nb_nodes += solver.solve(grid_, debug=True)["nb_nodes"]
        print(
            f"{solver} : {((time.perf_counter() - t1) / N):.4f} seconds, {nb_nodes / N} expanded"
        )


def greatest_heuristic(heuristics, m, n, N):
    grids = [grid.Grid.random_grid(m, n).to_tuple() for _ in range(N)]
    means = [sum(heuristic(g) for g in grids) / N for heuristic in heuristics]
    print(means)
