import solver
import time
import grid
import gadb
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
h1 = apdb.array_apdb_heuristic(m, n, groups)
h2 = apdb.dict_apdb_heuristic(m, n, groups)
h3 = gadb.gadb_heuristic(
    m,
    n,
    [
        (2, gadb.load_gadb(gadb.get_filename(m, n, 2))),
        (2, gadb.load_gadb(gadb.get_filename(m, n, 2))),
    ],
)
s1 = solver.AStarSolver(h1, "A* with APDB (np.array)")
s3 = solver.AStarSolver(h2, "A* with APDB (dict)")
s2 = solver.AStarSolver(utils.half_manhattan_distance, "A* with MD")
s4 = solver.AStarSolver(h3, "A* with GADB")
benchmark([s1, s2, s3, s4], m, n, 10)
