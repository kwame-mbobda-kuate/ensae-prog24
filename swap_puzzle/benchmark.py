import solver
import time
import grid
import utils
import apdb


def benchmark(solvers, m, n, N):
    """
    Benchmarks a list of solvers on random grids.
    """
    grids = [grid.Grid.random_grid(m, n) for _ in range(N)]
    for solver in solvers:
        t1 = time.perf_counter()
        for grid_ in grids:
            solver.solve(grid_)
        print(f"{solver} : {((time.perf_counter() - t1) / N):.3f} seconds")


m, n = 4, 4
groups = [[1, 2, 3, 5, 6], [9, 10, 13, 14, 15, 16], [4, 7, 8, 11, 12]]
h = apdb.apdb_heuristic(m, n, groups)
s1 = solver.AStarSolver(h, "A* with APDB")
s2 = solver.AStarSolver(utils.halved_manhattan_distance, "A* with MD")
benchmark([s1, s2], m, n, 5)
