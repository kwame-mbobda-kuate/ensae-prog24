# This will work if ran from the root folder ensae-prog24
import sys

sys.path.append("swap_puzzle/")

import unittest
import solver
import utils
from grid import Grid


class TestHeuristic(unittest.TestCase):

    def aux_test_admissibiliy(self, m: int, n: int, N: int, solver, heuristic):
        for _ in range(N):
            g = Grid.random_grid(m, n)
            self.assertTrue(len(solver.solve(g)) >= heuristic(g.to_tuple()))

    def test_inversion(self):
        s = solver.MDAStarSolver()
        self.aux_test_admissibiliy(3, 3, 1000, s, utils.inversion_distance)


if __name__ == "__main__":
    unittest.main()
