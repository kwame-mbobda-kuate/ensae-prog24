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


def compare_heuristic(heuristics, m, n, N):
    grids = [grid.Grid.random_grid(m, n).to_tuple() for _ in range(N)]
    means = [sum(heuristic(g) for g in grids) / N for heuristic in heuristics]
    print(means)

s = solver.MDAStarSolver()
s.compute(4, 4)
h = gadb.GADBList([GADB.default_load(4, 4, 2), GADB.default_load(4, 4, 3)]).get_heuristic()
s2 = solver.AStarSolver(h)
solving_time([s, s2], 4, 4, 10)