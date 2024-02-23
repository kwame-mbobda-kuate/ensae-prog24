# This will work if ran from the root folder ensae-prog24
import sys

sys.path.append("swap_puzzle/")

import unittest
from grid import Grid
from graph import Graph
import itertools
import apdb


class TestAPDB(unittest.TestCase):
    def aux_test_apdb(self, m, n, group):
        apdb1 = apdb.dict_loads_apdb(apdb.get_filename(m, n, group))
        apdb2 = apdb.array_loads_apdb(apdb.get_filename(m, n, group))
        for perm in itertools.permutations(range(m * n), len(group)):
            mapping = tuple(perm)
            self.assertEqual(apdb1[mapping], apdb2[mapping])

    def test_group1(self):
        return self.aux_test_apdb(4, 4, [1, 2, 3, 5, 6])

    def test_group2(self):
        return self.aux_test_apdb(4, 4, [9, 10, 13, 14, 15, 16])

    def test_group3(self):
        return self.aux_test_apdb(4, 4, [4, 7, 8, 11, 12])


if __name__ == "__main__":
    unittest.main()
