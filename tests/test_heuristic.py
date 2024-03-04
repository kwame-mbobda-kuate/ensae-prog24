# This will work if ran from the root folder ensae-prog24
import sys

sys.path.append("swap_puzzle/")

import unittest
import solver
import gadb
import utils
from gadb import GADB
from grid import Grid
from apdb import APDB
import apdb


class TestHeuristic(unittest.TestCase):

    def aux_test_admissibiliy(self, m: int, n: int, N: int, solver, heuristic):
        for _ in range(N):
            g = Grid.random_grid(m, n)
            self.assertTrue(len(solver.solve(g)) >= heuristic(g.to_tuple()))
    
    def test_gadb(self):
        h = gadb.List([GADB.default_load(3, 3, 3), GADB.default_load(3, 3, 2)]).get_heuristic()
        s = solver.AStarSolver(utils.half_manhattan_distance, "A*")
        self.aux_test_admissibiliy(3, 3, 1000, s, h)

if __name__ == "__main__":
    unittest.main()
