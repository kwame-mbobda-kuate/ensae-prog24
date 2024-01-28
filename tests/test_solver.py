# This will work if ran from the root folder ensae-prog24
import sys

sys.path.append("swap_puzzle/")

import unittest
from grid import Grid
import solver
import random
import itertools

solvers = [
    (solver.naive_solver, "Naive solver"),
    (solver.naive_bfs_solver, "Naive BFS solver"),
    (solver.bfs_solver, "BFS Solver"),
    (solver.manhattan_a_star_solver, "A* Solver with Manhattan heuristic"),
]
fast_and_exact_solvers = solvers[2:]


class TestSolver(unittest.TestCase):
    def test_is_sorted_solver(self):
        m, n = 2, 2
        grids = [Grid.random_grid(m, n) for _ in range(50)]
        for solver in solvers:
            print(solver[1])
            for grid in grids:
                swaps = solver[0](grid)
                grid.swap_seq(swaps)
                self.assertEqual(grid.is_sorted(), True)

    def test_is_optimal_solver(self):
        m, n = 3, 3
        grids = [Grid.random_grid(m, n) for _ in range(50)]
        length1, length2 = -1, -1
        for grid in grids:
            for i, solver in enumerate(fast_and_exact_solvers):
                length2 = length1
                length1 = len(solver[0](grid))
                if i > 0:
                    print(
                        fast_and_exact_solvers[i - 1][1],
                        "compared to",
                        fast_and_exact_solvers[i][1],
                    )
                    self.assertEqual(length1, length2)


if __name__ == "__main__":
    unittest.main()
