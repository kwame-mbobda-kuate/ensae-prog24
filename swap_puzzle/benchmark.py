import time
import grid
from apdb import ArrayAPDB
from gadb import GADB


def benchmark_solvers(solvers, m, n, N):
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
            f"{solver} : {((time.perf_counter() - t1) / N):.4f} seconds, {nb_nodes / N:.2f} nodes expanded"
        )


def benchmark_heuristics(heuristics, m, n, N):
    """
    Benchmarks a list of heuristics on random grids.
    """
    grids = [grid.Grid.random_grid(m, n).to_tuple() for _ in range(N)]
    for heuristic in heuristics:
        sum_h = 0
        t1 = time.perf_counter()
        for g in grids:
            sum_h = sum_h + heuristic(g) / N
        t2 = time.perf_counter() - t1
        print(sum_h, t2 / N)
