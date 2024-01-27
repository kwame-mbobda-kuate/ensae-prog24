# This will work if ran from the root folder ensae-prog24
import sys

sys.path.append("swap_puzzle/")

import unittest
from grid import Grid


class Test_Swap(unittest.TestCase):

    def test_allowed_swaps(self):
        g = Grid(5, 5)
        self.assertEquals(g.allowed_swaps((0, 0)), [(0, 1), (1, 0)])
        self.assertEquals(g.allowed_swaps((2, 2)), [(2, 3), (1, 2), (2, 1), (3, 2)])
        self.assertEquals(g.allowed_swaps((4, 2)), [(4, 3), (3, 2), (4, 1)])


if __name__ == "__main__":
    unittest.main()
