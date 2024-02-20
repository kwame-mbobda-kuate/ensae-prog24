import solver
import time
import grid
import utils


def benchmark(solvers, m, n, N):
    """
    Benchmarks a list of solvers on random grids.
    """
    grids = [grid.Grid.random_grid(m, n) for _ in range(N)]
    for solver in solvers:
        t1 = time.perf_counter()
        for grid_ in grids:
            solver.solve(grid_)
        print(solver, (time.perf_counter() - t1) / N * 1000)
