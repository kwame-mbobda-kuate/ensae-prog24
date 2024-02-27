# This will work if ran from the root folder ensae-prog24
import sys

sys.path.append("swap_puzzle/")

import unittest
from grid import Grid
import solver
import utils
from apdb import DictAPDB
from gadb import GADB
import gadb
import apdb


solvers = [
    solver.GreedySolver("Naive Solver"),
    solver.NaiveBFSSolver("Naive BFS solver"),
    solver.BFSSolver("BFS Solver"),
    solver.AStarSolver(
        utils.half_manhattan_distance, "A* Solver with Manhattan heuristic"
    ),
    solver.AStarSolver(gadb.GADBList([GADB.default_load(3, 3, 3), GADB.default_load(3, 3, 2)]).get_heuristic(), "GADB")
    #solver.AStarSolver(apdb.APDBList(utils.half_sum, [apdb.ArrayAPDB.default_load(3, 3, g) for g in [(1, 2, 3), (4, 5, 6), (7, 8, 9)]]).get_heuristic(), "DictAPDB")
]
fast_and_exact_solvers = solvers[2:]


class TestSolver(unittest.TestCase):
    def test_is_sorted_solver(self):
        m, n = 3, 3
        grids = [Grid.random_grid(m, n) for _ in range(10)]
        for solver in fast_and_exact_solvers:
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
                path = solver.solve(grid)
                print(path, solver)
                length1 = len(path)
                if i > 0:
                    print(
                        fast_and_exact_solvers[i - 1],
                        "compared to",
                        fast_and_exact_solvers[i],
                    )
                    self.assertEqual(length1, length2)
            print()


if __name__ == "__main__":
    unittest.main()
