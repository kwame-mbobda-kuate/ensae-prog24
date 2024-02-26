# This will work if ran from the root folder ensae-prog24
import sys

sys.path.append("swap_puzzle/")

import unittest
from grid import Grid


class TestHeuristic(unittest.TestCase):

    def test_admissibiliy(self, m: int, n: int, N: int, solver, heuristic):
        for _ in range(N):
            g = Grid.random_grid(m, n)
            self.assertTrue(len(solver.solve(g)) >= heuristic(g.to_tuple()))
