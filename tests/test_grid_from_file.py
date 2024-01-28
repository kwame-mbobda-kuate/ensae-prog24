# This will work if ran from the root folder ensae-prog24
import sys

sys.path.append("swap_puzzle/")

import unittest
from grid import Grid


class TestGridLoading(unittest.TestCase):
    def test_grid0(self):
        g = Grid.grid_from_file("input/grid0.in")
        self.assertEqual(g.m, 2)
        self.assertEqual(g.n, 2)
        self.assertEqual(g.state, [[2, 4], [3, 1]])

    def test_grid1(self):
        g = Grid.grid_from_file("input/grid1.in")
        self.assertEqual(g.m, 4)
        self.assertEqual(g.n, 2)
        self.assertEqual(g.state, [[1, 2], [3, 4], [5, 6], [8, 7]])


if __name__ == "__main__":
    unittest.main()
