# This will work if ran from the root folder ensae-prog24
import sys

sys.path.append("swap_puzzle/")

import unittest
from grid import Grid
import solver
import utils

solvers = [
    solver.GreedySolver("Naive Solver"),
    solver.NaiveBFSSolver("Naive BFS solver"),
    solver.BFSSolver("BFS Solver"),
    solver.OptimizedBFSSolver("Optimized BFS Solver"),
    solver.AStarSolver(
        utils.halved_manhattan_distance, "A* Solver with Manhattan heuristic"
    ),
]
fast_and_exact_solvers = solvers[2:]


class TestSolver(unittest.TestCase):
    def test_is_sorted_solver(self):
        m, n = 2, 2
        grids = [Grid.random_grid(m, n) for _ in range(10)]
        for solver in solvers:
            print(solver)
            for grid in grids:
                swaps = solver.solve(grid)
                grid.swap_seq(swaps)
                self.assertEqual(grid.is_sorted(), True)

    def test_is_optimal_solver(self):
        m, n = 3, 3
        grids = [Grid.random_grid(m, n) for _ in range(10)]
        length1, length2 = -1, -1
        for grid in grids:
            for i, solver in enumerate(fast_and_exact_solvers):
                length2 = length1
                length1 = len(solver.solve(grid))
                if i > 0:
                    print(
                        fast_and_exact_solvers[i - 1],
                        "compared to",
                        fast_and_exact_solvers[i],
                    )
                    self.assertEqual(length1, length2)


if __name__ == "__main__":
    unittest.main()
